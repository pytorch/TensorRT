# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""``low_memory``: construct a model on ``meta`` and map its weights from a checkpoint (or offload an already-loaded
model to disk) in one block, with the settings that keep compilation small."""

import contextlib
import errno
import itertools
import logging
import os
import platform
import sys
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType, MappingProxyType, TracebackType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.nn.modules.module import register_module_parameter_registration_hook
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils.weak import WeakIdKeyDictionary
from torch_tensorrt._utils import _cache_root
from torch_tensorrt.dynamo._weight_release import parameter_storages
from torch_tensorrt.dynamo.tools.weight_mapping import (
    MappedWeights,
    checkpoint_source,
    delete_file_when_unmapped,
    map_weights,
    release_mapped_storage,
    save_weights_for_mapping,
)
from torch_tensorrt.dynamo.utils import release_host_and_device_memory

logger = logging.getLogger(__name__)


@dataclass
class _MetaParameterContext:
    replacements: WeakIdKeyDictionary = field(default_factory=WeakIdKeyDictionary)
    warned_classes: Set[Type[Any]] = field(default_factory=set)


_meta_parameter_context: ContextVar[Optional[_MetaParameterContext]] = ContextVar(
    "torch_tensorrt_low_memory_meta_parameters", default=None
)


def _warn_if_called_from_pretrained(
    module: torch.nn.Module, ctx: _MetaParameterContext
) -> None:
    """Warn once per root loader class when ``from_pretrained`` runs under ``model_weights_on_meta=True``."""
    frame: Optional[FrameType] = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_name == "from_pretrained":
            cls = frame.f_locals.get("cls")
            key: Type[Any] = cls if isinstance(cls, type) else type(module)
            if key not in ctx.warned_classes:
                ctx.warned_classes.add(key)
                logger.warning(
                    f"{key.__name__}.from_pretrained(...) was called inside low_memory(model_weights_on_meta=True). "
                    "from_pretrained still loads checkpoint weights into host memory before they are "
                    "discarded onto meta, so model_weights_on_meta=True is not effective. Initialize from a config "
                    f"instead (for example AutoModel.from_config(config) or {key.__name__}(config)), then "
                    "call low_mem.map_weights(model)."
                )
            return
        frame = frame.f_back


def _place_parameter_on_meta(
    module: torch.nn.Module, name: str, param: torch.nn.Parameter
) -> Optional[torch.nn.Parameter]:
    ctx = _meta_parameter_context.get()
    if ctx is None or param.is_meta:
        # Returning None makes PyTorch skip `param = output` and keep the original param untouched.
        return None
    _warn_if_called_from_pretrained(module, ctx)
    if param not in ctx.replacements:
        ctx.replacements[param] = torch.nn.Parameter(
            param.to(device="meta"), requires_grad=param.requires_grad
        )
    return ctx.replacements[param]


# Installed once so entering and leaving low_memory never mutates PyTorch's global hook dict concurrently.
register_module_parameter_registration_hook(_place_parameter_on_meta)


def stop_placing_parameters_on_meta() -> None:
    """Turn off ``low_memory``'s ``meta`` parameter-registration hook in the current context."""
    _meta_parameter_context.set(None)


def _warn_if_loaded_via_from_pretrained(model: torch.nn.Module) -> None:
    """Warn when a pre-loaded model passed to ``map_weights`` was already loaded via ``from_pretrained``."""
    if hasattr(model, "_weight_conversions") or (
        isinstance(getattr(model, "name_or_path", None), str)
        and bool(getattr(model, "name_or_path", ""))
    ):
        cls_name = type(model).__name__
        logger.warning(
            f"{cls_name} appears to have been loaded from a checkpoint (e.g. from_pretrained), which "
            "already materialized all weights in memory before map_weights offloaded them. To avoid peak "
            "weight memory during model initialization, pass model_weights_on_meta=True to low_memory(...) and "
            f"construct the model from its config (e.g. {cls_name}.from_config(config) or {cls_name}(config)) "
            "before calling low_mem.map_weights(model)."
        )


class low_memory:
    """Compile a model with low memory, streaming weights from a checkpoint without loading them into RAM.

    To avoid allocating weights during both model initialization and compilation, pass ``model_weights_on_meta=True``
    and **construct the model from its config** (e.g. ``AutoModelForCausalLM.from_config(config)`` or
    ``MyModel(config)`` — **do not** use ``from_pretrained``, which loads the full checkpoint into memory
    first). Entering the block places newly registered ``nn.Parameter`` tensors on ``meta`` while keeping
    buffers real (such as rotary frequencies computed in ``__init__``), and :meth:`map_weights` streams
    the checkpoint in 64 MiB chunks into a memory-mapped cache file::

        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        with torch_tensorrt.dynamo.low_memory(model_weights_on_meta=True) as low_mem:
            model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).eval()
            low_mem.map_weights(model)  # reads checkpoint from model.config._name_or_path
            exported = torch.export.export(model, (example,))
            trt_model = torch_tensorrt.dynamo.compile(
                exported, inputs=[example.cuda()], **low_mem.compile_settings
            )

    When the model's config does not record its source, or its checkpoint uses different tensor names, pass
    ``checkpoint`` (a safetensors file, sharded ``model.safetensors.index.json``, local folder, or Hugging
    Face Hub repo id) and optional ``checkpoint_key`` to :meth:`map_weights`::

        with torch_tensorrt.dynamo.low_memory(model_weights_on_meta=True) as low_mem:
            policy = PI05Policy(config).eval()
            low_mem.map_weights(
                policy, "lerobot/pi05_base", checkpoint_key=pi05_checkpoint_key
            )
            ...

    By default (``model_weights_on_meta=False``), entering the block leaves model construction untouched so
    ``low_memory()`` can be used purely for ``low_mem.compile_settings`` without :meth:`map_weights`::

        with torch_tensorrt.dynamo.low_memory() as low_mem:
            trt_model = torch_tensorrt.dynamo.compile(
                exported, inputs=[example.cuda()], **low_mem.compile_settings
            )

    :meth:`map_weights` can also offload an already-allocated model to disk when ``checkpoint`` is omitted,
    freeing its tensors before compilation — though loading via ``from_pretrained`` still pays full model
    memory during initialization (and logs a warning recommending ``from_config`` + ``model_weights_on_meta=True``).
    Anything outside a loaded model that still holds its tensors (an autograd graph, ``state_dict`` copy,
    or optimizer) keeps them in memory, and a warning says how much, from 1 MiB.

    The file is deleted once the model's weights no longer use it, or when Python exits. Its directory must
    be on a disk, not a RAM-backed tmpfs, where the file would take the memory this frees: set ``TMPDIR``
    to choose it. A file a killed process left behind is deleted on the next mapping. :meth:`map_weights`
    is not supported on Windows, which can't delete a memory-mapped file.

    Args:
        model_weights_on_meta: place newly registered ``nn.Parameter`` tensors on ``meta`` upon entering the block
            until :meth:`map_weights` or compilation runs. ``False`` by default; pass ``True`` when
            constructing a model from its config without allocating weights before :meth:`map_weights`.
        offload_module_to_cpu: move the model's weights to the CPU before the engines are built, and move
            the weights of PyTorch fallback partitions to the GPU afterwards. On by default, and always on
            with ``model_weights_on_meta=True``, which ignores ``False`` with a warning: mapped weights are
            CPU tensors, so a fallback partition runs on the GPU only once this moves its weights there.
        enable_resource_partitioning: split TensorRT subgraphs into engines small enough to build within
            ``cpu_memory_budget``. On by default.
        cpu_memory_budget: host memory budget in bytes for resource partitioning.
        max_partition_size: weight bytes per TensorRT engine, which gives the same split on every run
            instead of one derived from free memory. Set it or ``cpu_memory_budget``, not both.
        release_weights_after_build: what to free once the last engine that reads a weight is built.
            ``None`` (default) frees only weights mapped by :meth:`map_weights`, by dropping their pages;
            the file gives them back if the weight is used again. ``True`` also frees unmapped weights, so
            the model can't use those afterwards; when every weight the engines read comes from
            :meth:`map_weights`, it frees nothing more. ``False`` frees nothing, and keeps the mapped pages
            resident when the block exits, where ``None`` and ``True`` drop them.

    Attributes:
        model: the last model passed to :meth:`map_weights`; ``None`` until :meth:`map_weights` runs.
        weights_file: the file holding ``model``'s weights; ``None`` until :meth:`map_weights` runs.
        compile_settings: the settings to pass to ``torch_tensorrt.dynamo.compile``.
        report: what :meth:`map_weights` did (bytes mapped); ``None`` until :meth:`map_weights` runs.
    """

    def __init__(
        self,
        *,
        model_weights_on_meta: bool = False,
        offload_module_to_cpu: Optional[bool] = None,
        enable_resource_partitioning: Optional[bool] = None,
        cpu_memory_budget: Optional[int] = None,
        max_partition_size: Optional[int] = None,
        release_weights_after_build: Optional[bool] = None,
    ) -> None:
        offload = True if offload_module_to_cpu is None else bool(offload_module_to_cpu)
        if model_weights_on_meta and not offload:
            logger.warning(
                "low_memory(model_weights_on_meta=True) keeps offload_module_to_cpu on: map_weights maps the "
                "weights as CPU tensors, and offload_module_to_cpu is what moves the weights of the parts that "
                "run in PyTorch to the GPU once the engines are built. Without it those parts keep CPU weights, "
                "and calling the compiled model with GPU inputs fails with a device mismatch."
            )
            offload = True
        settings = {
            "offload_module_to_cpu": offload,
            "enable_resource_partitioning": (
                True
                if enable_resource_partitioning is None
                else enable_resource_partitioning
            ),
            "cpu_memory_budget": cpu_memory_budget,
            "max_partition_size": max_partition_size,
            "release_weights_after_build": (
                None
                if release_weights_after_build is None
                else bool(release_weights_after_build)
            ),
        }
        self._model_weights_on_meta = bool(model_weights_on_meta)
        self._mapped_models: List[torch.nn.Module] = []
        self.compile_settings: Mapping[str, Any] = MappingProxyType(
            {k: v for k, v in settings.items() if v is not None}
        )
        self.model: Optional[torch.nn.Module] = None
        self.weights_file: Optional[Path] = None
        self.report: Optional[MappedWeights] = None

    def __enter__(self) -> "low_memory":
        if self._model_weights_on_meta:
            _meta_parameter_context.set(_MetaParameterContext())
        return self

    def map_weights(
        self,
        model: torch.nn.Module,
        checkpoint: Optional[Union[str, "os.PathLike[str]"]] = None,
        *,
        checkpoint_key: Optional[Callable[[str], str]] = None,
    ) -> MappedWeights:
        """Write ``model``'s weights to a cache file (from ``checkpoint`` or from ``model`` itself), point
        ``model``'s tensors at views of that file, and turn ``meta`` parameter creation off for compilation.

        Args:
            model: the ``nn.Module`` whose parameters and buffers should be mapped.
            checkpoint: optional safetensors file, sharded ``model.safetensors.index.json``, folder holding
                one of them, or Hugging Face Hub repo id. If omitted for a ``meta`` model, uses
                ``model.config._name_or_path`` when present.
            checkpoint_key: maps a model tensor's name to the checkpoint's name when they differ; the name
                itself by default.
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"map_weights takes an nn.Module, not {type(model).__name__}"
            )
        stop_placing_parameters_on_meta()
        if platform.system() == "Windows":
            raise NotImplementedError(
                "low_memory.map_weights keeps the model's weights in a memory-mapped file, which Windows "
                "can't delete; map a weights file you keep with torch_tensorrt.dynamo.tools.map_weights "
                "instead, or use low_memory(model_weights_on_meta=False) for compile_settings only."
            )

        tensors = list(itertools.chain(model.named_parameters(), model.named_buffers()))
        on_meta = [name for name, tensor in tensors if tensor.is_meta]
        if not on_meta:
            _warn_if_loaded_via_from_pretrained(model)
        effective_checkpoint = checkpoint
        if effective_checkpoint is None and on_meta:
            effective_checkpoint = _get_checkpoint_from_model_config(model)
        if checkpoint_key is not None and effective_checkpoint is None:
            raise ValueError(
                "checkpoint_key names the checkpoint's tensors; pass checkpoint= too"
            )

        source = None
        if effective_checkpoint is not None:
            source = checkpoint_source(effective_checkpoint, checkpoint_key)
        elif on_meta:
            raise ValueError(
                f"{len(on_meta)} of the model's tensors are on the meta device, with no values: "
                f"{on_meta[:8]}. Pass the checkpoint path or Hub repo id (checkpoint=) to "
                "low_mem.map_weights(...), or build from a Hugging Face config loaded with "
                "AutoConfig.from_pretrained(...) via AutoModel.from_config(config)."
            )
        loaded = _weak_refs_to_storages(tensor for _, tensor in tensors)

        path = _new_weights_file_path()
        try:
            save_weights_for_mapping(model, path, source=source)
        except OSError as e:
            # Reading the checkpoint can fail too; only a full disk is TMPDIR's.
            if e.errno not in (errno.ENOSPC, errno.EDQUOT, errno.EFBIG):
                raise
            raise OSError(
                e.errno,
                f"Could not write the model's weights to {path} ({e.strerror}). low_memory keeps them in a "
                "file while the model uses them; set TMPDIR to a disk with room for them.",
            ) from e
        try:
            report = map_weights(model, path)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        delete_file_when_unmapped(path)
        release_host_and_device_memory()
        held = sum(nbytes for ref, nbytes in loaded.values() if not ref.expired())
        # A view keeps the tensor it views, so an expanded buffer, such as SigLIP's position ids, keeps a
        # few bytes of its own.
        if held >= 2**20:
            logger.warning(
                f"{held / 2**20:.1f} MiB of the model's loaded weights are still in memory: something "
                "outside the model holds them, such as the autograd graph of a forward pass run with "
                "gradients, a state_dict copy or an optimizer. Drop it to free them."
            )
        self.model = model
        self.weights_file = path
        self.report = report
        self._mapped_models.append(model)
        return report

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        if self._model_weights_on_meta:
            stop_placing_parameters_on_meta()
        if self.compile_settings.get("release_weights_after_build") is not False:
            for mapped_model in self._mapped_models:
                for storage in parameter_storages(mapped_model).values():
                    release_mapped_storage(storage)


def _get_checkpoint_from_model_config(model: torch.nn.Module) -> Optional[str]:
    """The checkpoint path or Hub repo id recorded by Hugging Face on ``model.config``, if any."""
    config = getattr(model, "config", None)
    for candidate in (
        getattr(config, "_name_or_path", None),
        getattr(model, "name_or_path", None),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _weak_refs_to_storages(
    tensors: Iterable[torch.Tensor],
) -> Dict[int, Tuple[StorageWeakRef, int]]:
    """A weak reference to each distinct storage of ``tensors``, with its size, keyed by its address."""
    found: Dict[int, Tuple[StorageWeakRef, int]] = {}
    for tensor in tensors:
        if (
            tensor.is_meta
            or tensor.layout != torch.strided
            or is_traceable_wrapper_subclass(tensor)
        ):
            continue
        storage = tensor.untyped_storage()
        if storage.nbytes() and storage.data_ptr() not in found:
            found[storage.data_ptr()] = (StorageWeakRef(storage), storage.nbytes())
    return found


def _new_weights_file_path() -> Path:
    """A new path in the cache directory for a weights file, named after this process."""
    # Not the engine cache's directory: its eviction removes whatever it finds there.
    directory: Path = _cache_root() / "offloaded_weights"
    directory.mkdir(parents=True, exist_ok=True)
    _remove_files_of_dead_processes(directory)
    return directory / f"weights_{os.getpid()}_{os.urandom(8).hex()}.safetensors"


def _remove_files_of_dead_processes(directory: Path) -> None:
    """Delete the weights files, whole or partly written, of processes that no longer run. A killed process
    can't delete its file, and each is as large as a model."""
    for path in directory.glob("weights_*"):
        pid = path.name.split("_")[1]
        if pid.isdigit() and not _is_running(int(pid)):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
                logger.info(f"Deleted {path}, left behind by process {pid}")


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # signal 0 only checks that the process exists
    except ProcessLookupError:
        return False
    except PermissionError:  # it exists, as another user's
        return True
    return True
