"""Runtime cache handle + ``runtime_cache()`` context manager.

The handle wraps a ``trt.IRuntimeCache`` plus optional disk-backing config.
Used by:

* The runtime ``cache()`` CM to attach a SHARED cache across one or more
  modules' engines.
* ``RuntimeSettings(runtime_cache=...)`` (string path ⇒ engine creates an
  implicit per-engine handle; ``RuntimeCacheHandle`` ⇒ external shared
  handle attached directly).

File I/O lives entirely on the Python side under a ``filelock`` (this module).
The C++-side ``torch.classes.tensorrt.RuntimeCacheHandle`` is a passive
shared_ptr wrapper used only to cross the Python/C++ boundary.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from typing import IO, Any, Optional, Protocol, Sequence, Union

import torch
import torch_tensorrt

logger = logging.getLogger(__name__)

_FILELOCK_TIMEOUT_S = 10.0


class _CacheBacking(Protocol):
    """Abstracts pybind vs torchbind serialization of a TensorRT IRuntimeCache."""

    def serialize_bytes(self) -> Optional[bytes]: ...
    def deserialize_bytes(self, data: bytes) -> None: ...
    def is_ready(self) -> bool: ...


class _PybindBacking:
    """Wraps a pybind ``trt.IRuntimeCache`` directly (Python rt path)."""

    def __init__(self, cache: Any) -> None:
        self._cache = cache

    def serialize_bytes(self) -> Optional[bytes]:
        host_mem = self._cache.serialize()
        if host_mem is None or host_mem.nbytes == 0:
            return None
        return bytes(memoryview(host_mem))

    def deserialize_bytes(self, data: bytes) -> None:
        self._cache.deserialize(data)

    def is_ready(self) -> bool:
        # If you have a pybind cache, it's materialized.
        return True


class _TorchbindBacking:
    """Wraps a torchbind ``RuntimeCacheHandle`` sibling (cpp rt path).

    The cpp engine materializes the underlying ``trt_handle`` lazily on first
    execute, so ``is_ready()`` only returns True after that has happened.
    """

    def __init__(self, tb: Any) -> None:
        self._tb = tb

    def serialize_bytes(self) -> Optional[bytes]:
        if not self._tb.has_cache():
            return None
        tensor = self._tb.serialize()
        if tensor.numel() == 0:
            return None
        return bytes(tensor.cpu().contiguous().numpy())

    def deserialize_bytes(self, data: bytes) -> None:
        if not self._tb.has_cache():
            return
        tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        self._tb.deserialize(tensor)

    def is_ready(self) -> bool:
        return self._tb.has_cache()

    @property
    def torchbind(self) -> Any:
        return self._tb


class RuntimeCacheHandle:
    """Wraps a ``trt.IRuntimeCache`` (or a torchbind sibling) + optional disk path.

    Three construction patterns differ in *who else holds a reference*, which
    drives the ``autosave_on_del`` flag:

    1. **Engine-implicit**: when an engine sees
       ``RuntimeSettings(runtime_cache="/path")``, the engine's
       ``TRTRuntimeConfig`` materializes a handle internally with
       ``autosave_on_del=True``. No other Python object holds the handle,
       so ``__del__`` writes the cache to disk on the engine's last release.

    2. **Runtime CM** (shared, multi-engine): the :func:`runtime_cache` CM
       constructs a handle with ``autosave_on_del=False`` and explicitly
       calls ``handle.save()`` on ``__exit__``. The handle's ``__del__``
       no-ops since the CM already saved.

    3. **User-constructed** (advanced): hand-built handles default to
       ``autosave_on_del=False`` so save timing stays under the user's
       control. Opt in with ``RuntimeCacheHandle(path=..., autosave_on_del=True)``
       for with-block-style autosave on hand-built handles.

    Backing is polymorphic via :class:`_CacheBacking`: either a
    :class:`_PybindBacking` (Python rt; wraps ``trt.IRuntimeCache``) or a
    :class:`_TorchbindBacking` (cpp rt; wraps the torchbind sibling). Exactly
    one is set at any time; the "exactly one" invariant is enforced at
    construction.

    **Equality is identity-based.** Two handles wrap distinct underlying
    ``IRuntimeCache`` instances even when they share a path, and the runtime
    treats them as such (separate ``IRuntimeConfig`` slots, no
    kernel-specialization sharing).
    """

    def __init__(
        self,
        cache: Any = None,
        path: str = "",
        autosave_on_del: bool = False,
        torchbind_handle: Any = None,
    ) -> None:
        # Mutually exclusive at construction -- enforced, not just documented.
        if cache is not None and torchbind_handle is not None:
            raise ValueError(
                "RuntimeCacheHandle: specify ``cache`` OR ``torchbind_handle``, not both"
            )
        if cache is not None:
            self._backing: Optional[_CacheBacking] = _PybindBacking(cache)
        elif torchbind_handle is not None:
            self._backing = _TorchbindBacking(torchbind_handle)
        else:
            self._backing = None
        self.path = path
        self.autosave_on_del = autosave_on_del
        self._lock = threading.Lock()

    # Back-compat properties for callers that previously inspected the raw
    # fields. ``RuntimeCacheHandle`` is in the public API surface and the
    # ``runtime_settings`` setter / ``_to_torchbind_handle`` helper read
    # these.
    @property
    def _cache(self) -> Any:
        if isinstance(self._backing, _PybindBacking):
            return self._backing._cache
        return None

    @property
    def _torchbind(self) -> Any:
        if isinstance(self._backing, _TorchbindBacking):
            return self._backing.torchbind
        return None

    @property
    def cache(self) -> Any:
        """The underlying Python pybind ``trt.IRuntimeCache``. ``None`` if not yet materialized or if backed by a torchbind sibling."""
        return self._cache

    def ensure_cache(self, runtime_config: Any) -> Any:
        """Idempotent. First caller materializes via ``runtime_config.create_runtime_cache()``.

        Only meaningful for the Python-runtime path. The C++ runtime
        materializes its cache inside the engine and exposes bytes through
        the torchbind sibling.
        """
        with self._lock:
            if self._backing is None:
                self._backing = _PybindBacking(runtime_config.create_runtime_cache())
            assert isinstance(self._backing, _PybindBacking)
            return self._backing._cache

    def _read_bytes(self) -> Optional[bytes]:
        return self._backing.serialize_bytes() if self._backing is not None else None

    def _write_bytes(self, data: bytes) -> None:
        if self._backing is not None and self._backing.is_ready():
            self._backing.deserialize_bytes(data)

    def load_from_stream(self, stream: IO[bytes]) -> int:
        """Read bytes from ``stream`` and deserialize into the underlying cache.

        Returns the number of bytes consumed. ``0`` means "nothing to load" --
        either the handle has no backing cache yet or the stream had no bytes
        (first-run case, mirroring path-mode's missing-file early-return).
        IO errors from the stream itself (closed handle, ``UnsupportedOperation``
        on a write-only sink, etc.) propagate -- the caller passed something
        wrong and should hear about it. Path-mode :meth:`load` delegates here
        once the file is opened.
        """
        if self._backing is None:
            return 0
        data = stream.read()
        if not data:
            return 0
        self._write_bytes(data)
        logger.debug(f"Loaded runtime cache from stream ({len(data)} bytes)")
        return len(data)

    def save_to_stream(self, stream: IO[bytes]) -> int:
        """Serialize the underlying cache and write bytes to ``stream``.

        Returns the number of bytes written. ``0`` means "nothing to save" --
        the cache hadn't picked up any entries yet. IO errors from the stream
        (closed handle, ``UnsupportedOperation`` on a read-only sink, etc.)
        propagate. Path-mode :meth:`save` delegates here once the tmp file is
        opened and uses the return value to decide whether to promote the tmp
        to the destination (avoids writing a zero-byte cache file).
        """
        data = self._read_bytes()
        if not data:
            return 0
        stream.write(data)
        logger.debug(f"Saved runtime cache to stream ({len(data)} bytes)")
        return len(data)

    def load(self, path: Optional[str] = None) -> None:
        """Read bytes from disk and deserialize into the underlying cache.

        No-op if no cache backing is present, the resolved path is empty, or
        the file doesn't exist (first run). Caller must ensure no enqueue is
        concurrently writing (the CM enforces this by ordering load before
        engine attach; ``ensure_cache`` is called inside engine setup).
        """
        target = path if path is not None else self.path
        if not target:
            return
        if self._backing is None:
            return
        if not os.path.exists(target):
            return  # first run; nothing to load
        from filelock import FileLock

        with FileLock(target + ".lock").acquire(timeout=_FILELOCK_TIMEOUT_S):
            with open(target, "rb") as f:
                self.load_from_stream(f)

    def save(self, path: Optional[str] = None) -> None:
        """Serialize the underlying cache and write to disk under a filelock.

        No-op if path is empty or the cache wasn't materialized. Caller must
        ensure no enqueue is concurrently writing (the CM detaches the cache
        from all engines before calling save in ``__exit__``).
        """
        target = path if path is not None else self.path
        if not target:
            return
        from filelock import FileLock

        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = target + ".tmp"
        with FileLock(target + ".lock").acquire(timeout=_FILELOCK_TIMEOUT_S):
            with open(tmp, "wb") as f:
                wrote = self.save_to_stream(f)
            # If the cache had nothing to serialize, drop the empty tmp file
            # rather than promote a zero-byte cache file over the destination.
            if wrote:
                shutil.move(tmp, target)
            else:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def __del__(self) -> None:
        # Best-effort autosave for engine-implicit handles. The CM disables
        # this (``autosave_on_del=False``) since it saves on ``__exit__``;
        # user-constructed handles default to disabled so save timing stays
        # under the user's control. ``__del__`` can fire during interpreter
        # shutdown when imports/filesystem ops fail unpredictably -- swallow.
        if self.autosave_on_del and self.path:
            try:
                self.save()
            except Exception:
                pass

    def __repr__(self) -> str:
        return (
            f"RuntimeCacheHandle(path={self.path!r}, "
            f"autosave_on_del={self.autosave_on_del}, "
            f"materialized={self._backing is not None})"
        )


class _RuntimeCacheContextManager:
    """``with runtime_cache(target, path) as rc:`` -- shared cache CM.

    Bootstraps an ``IRuntimeCache`` from one of the engines under target,
    wraps it in a :class:`RuntimeCacheHandle`, loads from disk (or a
    user-provided stream), attaches to all engines under all listed targets
    for the duration of the block, and saves on exit if ``autosave``.

    The ``path`` slot is overloaded:

    * ``str`` / ``os.PathLike`` -> file-backed (load from + atomic-write to
      disk under a ``filelock``).
    * file-like (anything with ``.read`` or ``.write``) -> stream-backed
      (``.read()`` once on enter, ``.write(bytes)`` once on exit). The caller
      owns open/close/flush via their own ``with open(...)`` block. Useful
      for ``io.BytesIO``, gzip streams, or any other in-memory sink.
    * ``""`` (default) -> in-memory only.
    """

    def __init__(
        self,
        target_or_targets: Union["torch.nn.Module", Sequence["torch.nn.Module"]],
        path: Union[str, "os.PathLike[str]", IO[bytes], Any] = "",
        autosave: bool = True,
    ) -> None:
        if isinstance(target_or_targets, torch.nn.Module):
            self._targets: tuple[torch.nn.Module, ...] = (target_or_targets,)
        else:
            self._targets = tuple(target_or_targets)

        # Resolve the IO source. ``self.path`` stays a real string so that
        # ``rc.path`` keeps the same shape for callers; stream-mode reports
        # ``""`` here.
        if isinstance(path, (str, os.PathLike)):
            self.path = os.fspath(path) if path else ""
            self._stream: Optional[IO[bytes]] = None
        elif hasattr(path, "read") or hasattr(path, "write"):
            self.path = ""
            self._stream = path
        else:
            raise TypeError(
                "runtime_cache(): 'path' must be str, os.PathLike, or a "
                f"file-like object (with .read or .write); got {type(path).__name__}"
            )

        self.autosave = autosave
        self.handle: Optional[RuntimeCacheHandle] = None
        self._inner_cm: Any = None

    def _load_into(self, handle: "RuntimeCacheHandle") -> None:
        """Apply our IO source to the handle's load path.

        Path-mode -> filelocked read from disk. Stream-mode -> single
        ``.read()`` from the caller-owned file-like. In-memory
        (``path=''``) -> no-op.
        """
        if self._stream is not None:
            handle.load_from_stream(self._stream)
        elif self.path:
            handle.load()
        # else: in-memory, nothing to load

    def _save_from(self, handle: "RuntimeCacheHandle") -> None:
        """Apply our IO source to the handle's save path.

        Path-mode -> filelocked atomic-rename write. Stream-mode -> single
        ``.write(bytes)`` to the caller-owned sink. In-memory -> no-op.
        """
        if self._stream is not None:
            handle.save_to_stream(self._stream)
        elif self.path:
            handle.save()
        # else: in-memory, nothing to save

    def __enter__(self) -> RuntimeCacheHandle:
        # Defer imports to avoid a circular dependency:
        # _runtime_cache -> _runtime_config -> _TorchTensorRTModule -> (indirect) _runtime_cache.
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )
        from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine
        from torch_tensorrt.runtime._runtime_config import runtime_config

        # 1. Find any TorchTensorRTModule under the targets; first one wins.
        bootstrap_module = None
        for target in self._targets:
            for _, mod in target.named_modules():
                if isinstance(mod, TorchTensorRTModule):
                    bootstrap_module = mod
                    break
            if bootstrap_module is not None:
                break
        if bootstrap_module is None:
            raise RuntimeError(
                "runtime_cache() requires at least one TorchTensorRTModule "
                "under the target(s)."
            )

        # 2. Build the handle. The underlying ``IRuntimeCache`` materializes
        # differently per runtime:
        #
        # - Python rt: ``IRuntimeConfig.create_runtime_cache()`` returns a pybind
        #   ``IRuntimeCache`` directly; we materialize up front and load before
        #   attach.
        # - Cpp rt: the torchbind sibling carries a ``shared_ptr<IRuntimeCache>``
        #   that the C++ ``TRTRuntimeConfig::ensure_initialized`` populates on
        #   first attach; we attach first and load after.
        #
        # ``autosave_on_del=False`` in both cases -- the CM saves explicitly on
        # ``__exit__``; letting ``__del__`` also save would double-write when
        # ``rc`` falls out of scope after the with-block.
        if isinstance(bootstrap_module.engine, TRTEngine):
            cache_obj = bootstrap_module.engine.runtime_config.create_runtime_cache()
            self.handle = RuntimeCacheHandle(
                cache=cache_obj, path=self.path, autosave_on_del=False
            )
            self._load_into(self.handle)
            self._inner_cm = runtime_config(
                list(self._targets), runtime_cache=self.handle
            )
            self._inner_cm.__enter__()
        else:
            tb = torch.classes.tensorrt.RuntimeCacheHandle(self.path)
            self.handle = RuntimeCacheHandle(
                torchbind_handle=tb, path=self.path, autosave_on_del=False
            )
            # Attach first -- the C++ engine materializes ``tb.trt_handle`` in
            # the process of applying ``runtime_settings``. After that the load
            # path can write into a live ``IRuntimeCache``.
            self._inner_cm = runtime_config(
                list(self._targets), runtime_cache=self.handle
            )
            self._inner_cm.__enter__()
            self._load_into(self.handle)
        return self.handle

    def __exit__(self, *args: Any) -> None:
        if self._inner_cm is not None:
            self._inner_cm.__exit__(*args)
        if self.autosave and self.handle is not None:
            # Wait for in-flight enqueues on the still-attached cache before
            # ``_inner_cm.__exit__`` detached it; otherwise a concurrent
            # ``compiled(*inputs)`` on another thread can land kernels into a
            # cache that's about to be read here.
            try:
                torch.cuda.synchronize()
            except Exception as e:
                logger.debug(f"torch.cuda.synchronize() before save failed: {e}")
            # Swallow save errors at exit so a transient filesystem failure
            # (full disk, filelock timeout, permission denied) doesn't mask
            # the user's actual exception when the ``with`` block is already
            # exiting via raise. Mirrors the engine-implicit handle's
            # ``__del__`` semantics.
            try:
                self._save_from(self.handle)
            except Exception as e:
                logger.warning(f"Failed to autosave runtime cache on CM exit: {e}")


def runtime_cache(
    target_or_targets: Union["torch.nn.Module", Sequence["torch.nn.Module"]],
    path: Union[str, "os.PathLike[str]", IO[bytes], Any] = "",
    autosave: bool = True,
) -> _RuntimeCacheContextManager:
    """Context manager that attaches a shared runtime cache to all engines
    under ``target_or_targets`` for the duration of the ``with`` block.

    ``path`` accepts a filesystem path (``str``/``os.PathLike``) or a
    file-like object (anything with ``.read`` / ``.write``, e.g. an opened
    file handle or ``io.BytesIO``). Streams are read once on enter and
    written once on exit; ownership of open/close stays with the caller.

    Yields the :class:`RuntimeCacheHandle` for inspection or explicit
    ``handle.save()`` calls (e.g., for mid-block checkpointing -- caller is
    responsible for ``torch.cuda.synchronize()`` first).
    """
    return _RuntimeCacheContextManager(target_or_targets, path, autosave)


# When the C++ Torch-TensorRT runtime is loaded, we ALSO expose
# ``torch.classes.tensorrt.RuntimeCacheHandle`` as the canonical
# cross-language handle. The Python class above is the user-facing API;
# at dispatch time the Python module converts to/from the torchbind class as
# needed (see ``TorchTensorRTModule.runtime_settings`` setter).
def _to_torchbind_handle(
    rc: Union[None, str, "RuntimeCacheHandle", Any],
) -> Any:
    """Convert a Python-side ``runtime_cache`` value to a torchbind handle
    suitable for ``torch.classes.tensorrt.Engine.update_runtime_settings(...)``.

    Returns ``None`` if no runtime cache is requested. Raises if the C++
    runtime isn't loaded (caller shouldn't dispatch to a C++ engine in that
    case anyway). Already-torchbind handles (``torch.ScriptObject``) are passed
    through unchanged so callers can pre-stash a handle on the module and
    share it across dispatch calls.
    """
    if rc is None:
        return None
    if not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
        raise RuntimeError(
            "torch_tensorrt C++ runtime is not available; cannot construct "
            "torch.classes.tensorrt.RuntimeCacheHandle"
        )
    if isinstance(rc, torch.ScriptObject):
        return rc
    if isinstance(rc, RuntimeCacheHandle):
        # Reuse an existing torchbind sibling so the C++ engine sees the
        # same underlying pointer across calls. Falling through to construct
        # a fresh torchbind would orphan the existing one.
        if isinstance(rc._backing, _TorchbindBacking):
            return rc._backing.torchbind
        # Mixed-runtime hazard: a pybind-backed handle crossing into the cpp
        # runtime would silently drop its materialized ``IRuntimeCache``
        # (the cpp engine would attach a fresh torchbind below). Reject so
        # the user picks a deliberate fix.
        if isinstance(rc._backing, _PybindBacking):
            raise RuntimeError(
                "Cannot attach a Python-side RuntimeCacheHandle (with a live "
                "pybind IRuntimeCache) to a C++ runtime engine: the cache would "
                "be orphaned. Reconstruct the handle on the C++ side, or "
                "serialize/deserialize the cache bytes explicitly."
            )
        # No backing yet: synthesize a torchbind from the (possibly empty) path.
        return torch.classes.tensorrt.RuntimeCacheHandle(rc.path) if rc.path else None
    # Truthy-string check: ``""`` would construct a no-op torchbind handle.
    if isinstance(rc, str) and rc:
        return torch.classes.tensorrt.RuntimeCacheHandle(rc)
    return None
