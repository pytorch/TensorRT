# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""``low_memory``: compile a model straight from its checkpoint, with the settings that keep compilation small."""

import os
from types import MappingProxyType, TracebackType
from typing import Any, Callable, Mapping, Optional, Type, Union

import torch
from torch_tensorrt.dynamo._weight_release import parameter_storages
from torch_tensorrt.dynamo.tools.weight_mapping import (
    MappedWeights,
    map_weights,
    release_mapped_storage,
)


class low_memory:
    """Build a model from its weights file without loading it, with the compile settings that keep its
    compilation small.

    Entering the block builds the model on ``meta``, so it allocates no weights, maps its weights from the
    safetensors file ``weights`` with ``map_weights``, and returns this object, whose ``model`` is the
    model, on the CPU. Every tensor the model reads must be in the file (``save_weights_for_mapping``
    writes them all, non-persistent buffers included): a missing parameter or persistent buffer raises, and
    a buffer left on ``meta`` is named in a warning. Export the model with CPU inputs and compile it
    with ``torch_tensorrt.dynamo.compile``, passing ``compile_settings``; each weight's pages are read while
    its engine is built and dropped once its last engine is built. (``torch_tensorrt.compile(model,
    inputs=...)`` traces with the inputs on the GPU, which a CPU model can't take.)::

        with torch_tensorrt.dynamo.low_memory(lambda: MyModel(config), "weights.safetensors") as low_mem:
            exported = torch.export.export(low_mem.model, (example,))
            trt_model = torch_tensorrt.dynamo.compile(
                exported, inputs=[example.cuda()], **low_mem.compile_settings
            )

    The keyword arguments are the ``torch_tensorrt.dynamo.compile`` settings that lower memory during
    compilation. ``compile_settings`` holds the ones given, with ``offload_module_to_cpu`` and
    ``enable_resource_partitioning`` on unless set; a setting left at ``None`` otherwise isn't in it, so
    ``compile``'s default applies. The settings apply only to the calls they're passed to.

    On leaving, the pages of the model's mapped parameters are dropped, also on an exception, unless
    ``release_weights_after_build=False``; buffers, which the model may update, keep theirs. The file
    stays mapped: the model still reads it, so keep it in place, unchanged, while the model is in use. A
    write to a mapped weight survives on Linux 5.4 and later; older kernels discard it when its pages are
    dropped.

    Args:
        model_fn: builds the model, without moving it to a device.
        weights: the safetensors file to map, in the model's names, dtypes and shapes
            (``save_weights_for_mapping`` writes one, non-persistent buffers included).
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
        model: the model, its weights mapped; ``None`` until the block is entered.
        compile_settings: the settings to pass to ``torch_tensorrt.dynamo.compile``.
        report: what ``map_weights`` did (bytes mapped, names missing); ``None`` until the block is
            entered.
    """

    def __init__(
        self,
        model_fn: Callable[[], torch.nn.Module],
        weights: Union[str, "os.PathLike[str]"],
        *,
        offload_module_to_cpu: Optional[bool] = None,
        enable_resource_partitioning: Optional[bool] = None,
        cpu_memory_budget: Optional[int] = None,
        release_weights_after_build: Optional[bool] = None,
    ) -> None:
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
        self._model_fn = model_fn
        self._weights = weights
        self.compile_settings: Mapping[str, Any] = MappingProxyType(
            {k: v for k, v in settings.items() if v is not None}
        )
        self.model: Optional[torch.nn.Module] = None
        self.report: Optional[MappedWeights] = None

    def __enter__(self) -> "low_memory":
        with torch.device("meta"):
            model = self._model_fn()
        self.report = map_weights(model, self._weights, strict=True)
        self.model = model
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
