# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""``low_memory``: compile a loaded model from a file of its weights, with the settings that keep compilation small."""

import contextlib
import itertools
import logging
import os
import platform
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Type

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch_tensorrt._utils import _cache_root
from torch_tensorrt.dynamo._weight_release import parameter_storages
from torch_tensorrt.dynamo.tools.weight_mapping import (
    MappedWeights,
    delete_file_when_unmapped,
    map_weights,
    release_mapped_storage,
    save_weights_for_mapping,
)
from torch_tensorrt.dynamo.utils import release_host_and_device_memory

logger = logging.getLogger(__name__)


class low_memory:
    """Move a loaded model's weights into a file, with the compile settings that keep its compilation small.

    Entering the block writes the model's parameters and buffers to a new file in Torch-TensorRT's cache
    directory, then points the model's own tensors at zero-copy views of it, which frees the loaded copy:
    the model runs as before, on the CPU, and ``model`` is the same object. Export it with CPU inputs and
    compile it with ``torch_tensorrt.dynamo.compile``, passing ``compile_settings``; each weight's pages
    are read while its engine is built and dropped once its last engine is built. (``torch_tensorrt.compile(
    model, inputs=...)`` traces with the inputs on the GPU, which a CPU model can't take.)::

        model = MyModel.from_pretrained(checkpoint).eval()
        with torch_tensorrt.dynamo.low_memory(model) as low_mem:
            exported = torch.export.export(low_mem.model, (example,))
            trt_model = torch_tensorrt.dynamo.compile(
                exported, inputs=[example.cuda()], **low_mem.compile_settings
            )

    The model has to fit in memory once, while it loads; the compilation then holds about one engine's
    weights at a time instead. To compile a model that doesn't fit, build it on ``meta`` and map a weights
    file with ``torch_tensorrt.dynamo.tools.map_weights``. Anything else that still holds the loaded tensors,
    such as the autograd graph of a forward pass run with gradients, a ``state_dict`` copy or an optimizer,
    keeps them in memory, and a warning says how much.

    The file is deleted once the model's weights no longer use it, or when Python exits. Its directory must
    be on a disk, not a RAM-backed tmpfs, where the file would take the memory this frees: set ``TMPDIR``
    to choose it. A file a killed process left behind is deleted on the next entry. Not supported on
    Windows, which can't delete a memory-mapped file.

    The keyword arguments are the ``torch_tensorrt.dynamo.compile`` settings that lower memory during
    compilation. ``compile_settings`` holds the ones given, with ``offload_module_to_cpu`` and
    ``enable_resource_partitioning`` on unless set; a setting left at ``None`` otherwise isn't in it, so
    ``compile``'s default applies. The settings apply only to the calls they're passed to.

    On leaving, the pages of the model's parameters are dropped, also on an exception, unless
    ``release_weights_after_build=False``; buffers, which the model may update, keep theirs.

    Args:
        model: the loaded model, on any device.
        offload_module_to_cpu: move the model's weights to the CPU before the engines are built, so
            peak GPU memory drops to about the engines' size while the CPU holds the weights, and move
            the weights of the parts that run in PyTorch to the GPU afterwards. On by default: the mapped
            weights are on the CPU already, and a PyTorch part that reads one needs it on the GPU.
        enable_resource_partitioning: split TensorRT subgraphs into engines small enough to build
            within ``cpu_memory_budget``. On by default; without a budget, engines are sized to the
            system's available memory, so a model that fits isn't split.
        cpu_memory_budget: the host memory, in bytes, the compiling process may use, counting what it
            already uses. Read only with ``enable_resource_partitioning=True``; without a budget, the
            partitioner sizes engines to the system's available memory.
        release_weights_after_build: ``True`` also frees weights that aren't mapped, which the source
            model then can't use; ``False`` releases nothing, and leaving the block then keeps the
            pages too. ``compile``'s default drops only mapped weights' pages, once their last engine
            is built.

    Attributes:
        model: the model, its weights in ``weights_file``; ``None`` until the block is entered.
        weights_file: the file holding the model's weights; ``None`` until the block is entered.
        compile_settings: the settings to pass to ``torch_tensorrt.dynamo.compile``.
        report: what ``map_weights`` did (bytes mapped); ``None`` until the block is entered.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        offload_module_to_cpu: Optional[bool] = None,
        enable_resource_partitioning: Optional[bool] = None,
        cpu_memory_budget: Optional[int] = None,
        release_weights_after_build: Optional[bool] = None,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"low_memory takes the loaded model, not {type(model).__name__}. Load it as usual and pass "
                "it; to map a weights file into a model built on meta, use "
                "torch_tensorrt.dynamo.tools.map_weights."
            )
        settings = {
            "offload_module_to_cpu": (
                True if offload_module_to_cpu is None else offload_module_to_cpu
            ),
            "enable_resource_partitioning": (
                True
                if enable_resource_partitioning is None
                else enable_resource_partitioning
            ),
            "cpu_memory_budget": cpu_memory_budget,
            "release_weights_after_build": (
                None
                if release_weights_after_build is None
                else bool(release_weights_after_build)
            ),
        }
        self._loaded_model = model
        self.compile_settings: Mapping[str, Any] = MappingProxyType(
            {k: v for k, v in settings.items() if v is not None}
        )
        self.model: Optional[torch.nn.Module] = None
        self.weights_file: Optional[Path] = None
        self.report: Optional[MappedWeights] = None

    def __enter__(self) -> "low_memory":
        if platform.system() == "Windows":
            raise NotImplementedError(
                "low_memory keeps the model's weights in a memory-mapped file, which Windows can't delete; "
                "map a weights file you keep with torch_tensorrt.dynamo.tools.map_weights instead."
            )
        model = self._loaded_model
        tensors = list(itertools.chain(model.named_parameters(), model.named_buffers()))
        on_meta = [name for name, tensor in tensors if tensor.is_meta]
        if on_meta:
            raise ValueError(
                f"low_memory takes a loaded model, and {len(on_meta)} of its tensors are on the meta device, "
                f"with no values: {on_meta[:8]}. Load them, or map a weights file into the model with "
                "torch_tensorrt.dynamo.tools.map_weights."
            )
        loaded = _weak_refs_to_storages(tensor for _, tensor in tensors)

        path = _new_weights_file_path()
        try:
            save_weights_for_mapping(model, path)
        except OSError as e:
            raise OSError(
                e.errno,
                f"Could not write the model's weights to {path} ({e.strerror}). low_memory keeps them in a "
                "file while the model uses them; set TMPDIR to a disk with room for them.",
            ) from e
        try:
            self.report = map_weights(model, path)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        delete_file_when_unmapped(path)
        release_host_and_device_memory()
        held = sum(nbytes for ref, nbytes in loaded.values() if not ref.expired())
        if held:
            logger.warning(
                f"{held / 2**20:.1f} MiB of the model's loaded weights are still in memory: something "
                "outside the model holds them, such as the autograd graph of a forward pass run with "
                "gradients, a state_dict copy or an optimizer. Drop it to free them."
            )
        self.model = model
        self.weights_file = path
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        if (
            self.model is not None
            and self.compile_settings.get("release_weights_after_build") is not False
        ):
            for storage in parameter_storages(self.model).values():
                release_mapped_storage(storage)


def _weak_refs_to_storages(
    tensors: Iterable[torch.Tensor],
) -> Dict[int, Tuple[StorageWeakRef, int]]:
    """A weak reference to each distinct storage of ``tensors``, with its size, keyed by its address."""
    found: Dict[int, Tuple[StorageWeakRef, int]] = {}
    for tensor in tensors:
        if tensor.layout != torch.strided or is_traceable_wrapper_subclass(tensor):
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
