"""Runtime cache facade + per-runtime handle + ``runtime_cache()`` CM.

User-facing :class:`RuntimeCache` is a thin facade that wraps an inner
runtime-cache handle. The inner handle is either:

* :class:`_RuntimeCacheHandle` (python rt) -- this module's python port of
  the cpp ``RuntimeCacheHandle`` class declared in
  ``core/runtime/RuntimeSettings.h``. Same public surface, same semantics,
  including deferred materialization (pending bytes drained on first
  ``ensure_materialized``).
* ``torch.classes.tensorrt.RuntimeCacheHandle`` (cpp rt) -- the torchbind
  cross-language sibling, used directly. The cpp side already implements
  pending-bytes semantics and is interface-compatible with the python
  class, so no wrapper class is needed on the cpp-rt path.

Both implementations satisfy :class:`_RuntimeCacheHandleProtocol`. The
facade forwards straight through with no isinstance branching.

File I/O lives entirely on the Python side under a ``filelock`` (this
module).
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import threading
import weakref
from functools import partial
from typing import (
    IO,
    Any,
    Callable,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

import torch
import torch_tensorrt

logger = logging.getLogger(__name__)

_FILELOCK_TIMEOUT_S = 10.0


@runtime_checkable
class _RuntimeCacheHandleProtocol(Protocol):
    """Common interface for python-rt and cpp-rt runtime-cache handles.

    Both the python class :class:`_RuntimeCacheHandle` and the torchbind
    class ``torch.classes.tensorrt.RuntimeCacheHandle`` satisfy this
    Protocol, letting :class:`RuntimeCache` forward to either without
    branching.

    Note: ``ensure_materialized`` is C++-public on the cpp class but NOT
    registered with torchbind (see ``core/runtime/register_jit_hooks.cpp``).
    The cpp engine calls it directly from cpp ``TRTRuntimeConfig::ensure_initialized``;
    python only ever invokes it on the python-rt :class:`_RuntimeCacheHandle`.
    The Protocol method is structurally present but unreachable on cpp rt.
    """

    path: str

    def serialize(self) -> torch.Tensor: ...
    def deserialize(self, data: torch.Tensor) -> None: ...
    def has_cache(self) -> bool: ...
    def ensure_materialized(self, runtime_config: Any) -> Optional[Any]: ...


class _RuntimeCacheHandle:
    """Python-rt port of cpp ``RuntimeCacheHandle``. Module-private.

    Same public surface as the cpp class. Two states:

    * **pending**: ``_cache is None``; ``_pending_warm_bytes`` may hold
      bytes stashed from a pre-materialize ``deserialize``. ``serialize``
      returns an empty tensor; ``has_cache`` returns ``False``.
    * **materialized**: ``_cache`` is a live ``trt.IRuntimeCache``.
      ``deserialize`` feeds bytes into the cache; ``serialize`` reads
      cache bytes back as a uint8 tensor; ``has_cache`` returns ``True``.

    ``ensure_materialized(runtime_config)`` is the state transition:
    creates ``_cache = runtime_config.create_runtime_cache()`` and drains
    ``_pending_warm_bytes`` into it. Idempotent.

    Needed for ``ensure_materialized``'s check-then-set across the
    GIL-releasing ``create_runtime_cache`` C call when a single handle is
    shared across N engines whose forwards run on different threads.
    """

    def __init__(self, path: str = "") -> None:
        self.path: str = path
        # Internal storage uses python ``bytes`` (matches what
        # ``trt.IRuntimeCache.deserialize(blob)`` takes). The public
        # ``serialize`` / ``deserialize`` use ``torch.Tensor`` to mirror
        # the cpp signature; conversion is at the API boundary.
        self._cache: Optional[Any] = None
        self._pending_warm_bytes: Optional[bytes] = None
        self._lock = threading.Lock()

    def _serialize_unlocked(self) -> torch.Tensor:
        """Lock-free helper used by ``serialize`` and ``__getstate__``;
        caller must hold ``self._lock``."""
        if self._cache is None:
            return torch.empty(0, dtype=torch.uint8)
        host_mem = self._cache.serialize()
        if host_mem is None or host_mem.nbytes == 0:
            return torch.empty(0, dtype=torch.uint8)
        return torch.frombuffer(
            bytearray(bytes(memoryview(host_mem))), dtype=torch.uint8
        )

    def serialize(self) -> torch.Tensor:
        with self._lock:
            return self._serialize_unlocked()

    def deserialize(self, data: torch.Tensor) -> None:
        with self._lock:
            if data.numel() == 0:
                return
            data_bytes = bytes(data.cpu().contiguous().numpy())
            if self._cache is None:
                self._pending_warm_bytes = data_bytes
                return
            self._cache.deserialize(data_bytes)

    def has_cache(self) -> bool:
        return self._cache is not None

    def ensure_materialized(self, runtime_config: Any) -> Any:
        """Idempotently create the underlying ``trt.IRuntimeCache`` against
        ``runtime_config`` and drain any pending warm-load bytes.

        Returns the live cache. Safe to call concurrently from multiple
        engines sharing this handle (the lock guards check-then-set across
        the GIL-releasing ``create_runtime_cache`` C call).
        """
        with self._lock:
            if self._cache is None:
                self._cache = runtime_config.create_runtime_cache()
                if self._pending_warm_bytes is not None:
                    self._cache.deserialize(self._pending_warm_bytes)
                    self._pending_warm_bytes = None
            return self._cache

    def __getstate__(self) -> dict[str, Any]:
        """Pickle as ``(path, bytes)`` mirroring the cpp ``def_pickle``
        contract. The bytes blob carries either the live ``IRuntimeCache``
        contents (when materialized) or the pending warm bytes
        (pre-materialize window), so the loading side gets a hot handle on
        the first engine without an extra ``handle.load()`` from disk.

        The lock and the live ``_cache`` are per-process and never cross
        the pickle boundary; ``__setstate__`` rebuilds them fresh.
        """
        with self._lock:
            if self._cache is not None:
                blob = self._serialize_unlocked()
            elif self._pending_warm_bytes is not None:
                blob = torch.frombuffer(
                    bytearray(self._pending_warm_bytes), dtype=torch.uint8
                )
            else:
                blob = torch.empty(0, dtype=torch.uint8)
            path = self.path
        return {"path": path, "bytes": blob}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.path = state["path"]
        self._cache = None
        self._pending_warm_bytes = None
        self._lock = threading.Lock()
        blob = state.get("bytes")
        if blob is not None and blob.numel() > 0:
            # Stashes into ``_pending_warm_bytes``; first engine that calls
            # ``ensure_materialized`` drains it into the live cache.
            self.deserialize(blob)


def _autosave_at_exit(ref: "weakref.ref[RuntimeCache]") -> None:
    """Module-level so the atexit closure only holds a weakref, not a bound
    method (see :meth:`RuntimeCache.__init__` for full rationale)."""
    rc = ref()
    if rc is not None:
        rc._autosave_if_enabled()


class RuntimeCache:
    """User-facing handle for the TensorRT-RTX runtime kernel cache.

    Three construction patterns differ in *who else holds a reference*,
    which drives the ``autosave_on_del`` flag:

    1. **Engine-implicit**: when an engine sees
       ``RuntimeSettings(runtime_cache="/path")``, the engine's
       ``TRTRuntimeConfig`` materializes a :class:`RuntimeCache` internally
       with ``autosave_on_del=True``. No other Python object holds it, so
       two non-owning hooks back the save: ``__del__`` for mid-program GC,
       and a ``weakref``-based ``atexit`` hook for interpreter shutdown
       (where ``__del__`` is unsafe because lazy imports like ``filelock``
       may already be torn down). Whichever fires first flips
       ``autosave_on_del`` off so the other no-ops.

    2. **Runtime CM** (shared, multi-engine): the :func:`runtime_cache` CM
       constructs a :class:`RuntimeCache` with ``autosave_on_del=False``
       and explicitly calls ``handle.save()`` on ``__exit__``. The
       ``__del__`` no-ops since the CM already saved.

    3. **User-constructed** (advanced): hand-built handles default to
       ``autosave_on_del=False`` so save timing stays under the user's
       control. Opt in with
       ``RuntimeCache(path=..., autosave_on_del=True)`` for with-block-style
       autosave on hand-built handles.

    The internal ``_handle`` is the cpp-rt torchbind
    ``torch.classes.tensorrt.RuntimeCacheHandle`` when
    ``ENABLED_FEATURES.torch_tensorrt_runtime`` is set, else the
    pure-Python :class:`_RuntimeCacheHandle`; both satisfy
    :class:`_RuntimeCacheHandleProtocol` interface which this class uses.

    **Identity-based equality.** Two handles wrap distinct underlying
    ``IRuntimeCache`` instances even when they share a path, and the
    runtime treats them as such.
    """

    def __init__(
        self,
        path: str = "",
        autosave_on_del: bool = False,
    ) -> None:
        # Set the atexit-token slot first so ``__del__`` can safely read it
        # even if a later step in ``__init__`` raises and leaves the object
        # partially constructed.
        self._atexit_token: Optional[Callable[..., None]] = None

        # Pick the backing that matches the active runtime. The torchbind
        # class ``torch.classes.tensorrt.RuntimeCacheHandle`` is registered by
        # the C++ shared library; if the .so isn't loaded
        # (``ENABLED_FEATURES.torch_tensorrt_runtime is False``) it doesn't
        # exist as an attribute, so we fall back to the pure-Python
        # ``_RuntimeCacheHandle``. Both satisfy
        # :class:`_RuntimeCacheHandleProtocol`, so the facade methods forward
        # without branching on which backing won.
        if torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
            self._handle: _RuntimeCacheHandleProtocol = (
                torch.classes.tensorrt.RuntimeCacheHandle(path)
            )
        else:
            self._handle = _RuntimeCacheHandle(path=path)
        self.autosave_on_del = autosave_on_del

        # Engine-implicit handles must save before ``sys.meta_path`` is torn
        # down at interpreter exit -- ``__del__`` then hits ``ImportError``
        # from the torchbind property and the lazy ``filelock`` import.
        # ``atexit`` fires before that teardown.
        if autosave_on_del:
            self._register_atexit_autosave()

    @property
    def path(self) -> str:
        """The disk path the handle is anchored to. Single source of truth
        is ``_handle.path``."""
        return self._handle.path

    def is_cpp_runtime(self) -> bool:
        """``True`` iff this handle is backed by the cpp runtime (wraps the
        torchbind ``torch.classes.tensorrt.RuntimeCacheHandle``). ``False``
        for the python runtime (wraps :class:`_RuntimeCacheHandle`)."""
        return not isinstance(self._handle, _RuntimeCacheHandle)

    def has_cache(self) -> bool:
        """Forwards to ``_handle.has_cache()``. ``True`` once the underlying
        ``trt.IRuntimeCache`` has been materialized."""
        return self._handle.has_cache()

    def ensure_cache(self, runtime_config: Any) -> Optional[Any]:
        """Engine-internal: materialize the cache against ``runtime_config``
        and return it. Called only from python
        :meth:`TRTRuntimeConfig._apply_settings`, which only runs on python
        rt. On cpp rt the engine materializes via the cpp side directly;
        this method is structurally unreachable.
        """
        if isinstance(self._handle, _RuntimeCacheHandle):
            return self._handle.ensure_materialized(runtime_config)
        # Unreachable in practice: _apply_settings doesn't run on cpp rt.
        return None

    def load_from_stream(self, stream: IO[bytes]) -> int:
        """Read bytes from ``stream`` and deserialize into the underlying
        cache. Returns the number of bytes consumed; ``0`` means the
        stream had no bytes.
        """
        data = stream.read()
        if not data:
            return 0
        tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        self._handle.deserialize(tensor)
        logger.debug(f"Loaded runtime cache from stream ({len(data)} bytes)")
        return len(data)

    def save_to_stream(self, stream: IO[bytes]) -> int:
        """Serialize the underlying cache and write bytes to ``stream``.
        Returns the number of bytes written; ``0`` means the cache had
        nothing to serialize (no entries, or not yet materialized).
        """
        tensor = self._handle.serialize()
        if tensor.numel() == 0:
            return 0
        data = bytes(tensor.cpu().contiguous().numpy())
        stream.write(data)
        logger.debug(f"Saved runtime cache to stream ({len(data)} bytes)")
        return len(data)

    def load(self, path: Optional[str] = None) -> None:
        """Read bytes from disk and deserialize into the underlying cache.

        No-op if the resolved path is empty or the file doesn't exist
        (first run). Caller must ensure no concurrent writer (the CM
        enforces this by ordering load before engine attach).
        """
        target = path if path is not None else self.path
        if not target:
            return
        if not os.path.exists(target):
            return  # first run; nothing to load
        from filelock import FileLock

        with FileLock(target + ".lock").acquire(timeout=_FILELOCK_TIMEOUT_S):
            with open(target, "rb") as f:
                self.load_from_stream(f)

    def save(self, path: Optional[str] = None) -> None:
        """Serialize the underlying cache and write to disk under a
        filelock.

        No-op if path is empty or the cache wasn't materialized. Caller
        must ensure no concurrent writer (the CM detaches the cache from
        all engines before calling save in ``__exit__``).
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

    def _autosave_if_enabled(self) -> None:
        """Idempotent autosave shared by ``__del__`` and the atexit hook.
        Flips ``autosave_on_del`` off; swallows any exception so a
        shutdown-time ``ImportError`` never leaks as
        ``Exception ignored in __del__``.
        """
        try:
            if self.autosave_on_del and self.path:
                self.autosave_on_del = False
                self.save()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, Any]:
        """Strip the unpicklable atexit token from the pickle stream.

        The token is a ``partial`` over a ``weakref`` -- both of which are
        per-process artifacts and ``weakref`` is unpicklable. The pickled
        state carries only ``_handle`` (cpp torchbind persists path-only;
        see ``register_jit_hooks.cpp``) and ``autosave_on_del``;
        ``__setstate__`` reconstructs a fresh atexit hook in the loading
        process if autosave was enabled.
        """
        state = self.__dict__.copy()
        state["_atexit_token"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        # Re-register atexit autosave in the loading process if it was
        # active in the saving one. The fresh ``weakref.ref(self)`` is
        # bound to the *new* instance, so the loading-process GC behavior
        # mirrors what ``__init__`` would have set up directly.
        if self.autosave_on_del:
            self._register_atexit_autosave()

    def _register_atexit_autosave(self) -> None:
        """Idempotently arm the atexit autosave hook for this handle.

        Shared by ``__init__`` (initial wiring when ``autosave_on_del=True``)
        and ``__setstate__`` (rewiring after pickle round-trip drops the
        token). ``partial`` over a ``weakref`` keeps the registration
        non-owning, so mid-program GC still runs ``__del__`` normally; the
        ``_atexit_token`` slot check makes a second call a no-op.
        """
        if self._atexit_token is None:
            self._atexit_token = atexit.register(
                partial(_autosave_at_exit, weakref.ref(self))
            )

    def __del__(self) -> None:
        # Mid-program GC path. The companion ``_autosave_at_exit`` hook
        # covers the case where the handle survives until interpreter exit;
        # whichever path runs first flips ``autosave_on_del`` off so the
        # other no-ops.
        self._autosave_if_enabled()

        # Drop our atexit hook so the registry does not accumulate dead
        # entries across many engine-implicit handles in long-running
        # processes (each entry holds a now-dead weakref).
        #
        # Use ``getattr`` rather than direct attribute access: protocols
        # like ``copy.deepcopy`` can crash mid-state-copy on an unrelated
        # field (e.g. a pre-existing ``threading.Lock`` somewhere else in
        # the object graph) and leave the new instance with only some of
        # its attributes set. ``__init__`` never ran on that object, so
        # ``self._atexit_token`` may simply not exist when ``__del__``
        # fires -- ``getattr`` with a default makes that case a no-op
        # instead of raising ``AttributeError`` to ``sys.unraisablehook``.
        token = getattr(self, "_atexit_token", None)
        if token is not None:
            try:
                atexit.unregister(token)
            except Exception:
                pass

    def __repr__(self) -> str:
        return (
            f"RuntimeCache(path={self.path!r}, "
            f"autosave_on_del={self.autosave_on_del}, "
            f"materialized={self.has_cache()})"
        )


class _RuntimeCacheContextManager:
    """``with runtime_cache(target, path) as rc:`` -- shared cache CM.

    Builds a fresh :class:`RuntimeCache`, loads from disk (or a
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
        self.handle: Optional[RuntimeCache] = None
        self._inner_cm: Any = None

    def _load_into(self, handle: "RuntimeCache") -> None:
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

    def _save_from(self, handle: "RuntimeCache") -> None:
        """Apply our IO source to the handle's save path.

        Path-mode -> filelocked atomic-rename write. Stream-mode -> single
        ``.write(bytes)`` to the caller-owned sink. In-memory -> no-op.
        """
        if self._stream is not None:
            handle.save_to_stream(self._stream)
        elif self.path:
            handle.save()
        # else: in-memory, nothing to save

    def __enter__(self) -> RuntimeCache:
        # Defer imports to avoid a circular dependency:
        # _runtime_cache -> _runtime_config -> _TorchTensorRTModule -> (indirect) _runtime_cache.
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )
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

        # 2. Build the handle in its pending state on both runtimes. The
        # facade auto-picks the torchbind sibling on cpp rt and the
        # pure-Python handle on python-only rt. Disk bytes go through
        # ``handle.load*`` into pending storage; the first engine's
        # ``_apply_settings`` (python rt) or cpp ``ensure_initialized``
        # (cpp rt) materializes the underlying ``IRuntimeCache`` and drains
        # the pending bytes atomically.
        #
        # ``autosave_on_del=False`` -- the CM saves explicitly on
        # ``__exit__``; letting ``__del__`` also save would double-write when
        # ``rc`` falls out of scope after the with-block.
        self.handle = RuntimeCache(path=self.path, autosave_on_del=False)
        self._load_into(self.handle)
        self._inner_cm = runtime_config(list(self._targets), runtime_cache=self.handle)
        self._inner_cm.__enter__()
        return self.handle

    def __exit__(self, *args: Any) -> None:
        if self._inner_cm is not None:
            self._inner_cm.__exit__(*args)
        if self.autosave and self.handle is not None:
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

    Yields the :class:`RuntimeCache` for inspection or explicit
    ``handle.save()`` calls (e.g., for mid-block checkpointing).
    """
    return _RuntimeCacheContextManager(target_or_targets, path, autosave)


# When the C++ Torch-TensorRT runtime is loaded, we ALSO expose
# ``torch.classes.tensorrt.RuntimeCacheHandle`` as the canonical
# cross-language handle. The Python :class:`RuntimeCache` above is the
# user-facing API; at dispatch time the Python module converts to/from
# the torchbind class as needed (see ``TorchTensorRTModule.runtime_settings``
# setter).
def _to_torchbind_handle(
    rc: Union[None, str, "RuntimeCache", Any],
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
    if isinstance(rc, RuntimeCache):
        # The facade auto-picks the torchbind sibling on cpp rt (see
        # ``RuntimeCache.__init__``), so any ``RuntimeCache`` constructed in
        # this process is already cpp-backed. Unwrap directly.
        return rc._handle
    # Truthy-string check: ``""`` would construct a no-op torchbind handle.
    if isinstance(rc, str) and rc:
        return torch.classes.tensorrt.RuntimeCacheHandle(rc)
    return None
