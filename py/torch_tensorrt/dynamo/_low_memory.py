# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""``low_memory``: compile a model straight from its checkpoint, with compile defaults for a block."""

import functools
import itertools
import os
from contextvars import ContextVar, Token
from types import MappingProxyType, TracebackType
from typing import Any, Callable, Mapping, Optional, Type, TypeVar, Union, cast

import torch
from torch_tensorrt.dynamo.tools.weight_mapping import (
    MappedWeights,
    map_weights,
    release_mapped_storage,
)

# The compile settings of the enclosing ``low_memory`` blocks.
_compile_defaults: ContextVar[Mapping[str, Any]] = ContextVar(
    "low_memory_compile_defaults", default=MappingProxyType({})
)

F = TypeVar("F", bound=Callable[..., Any])


def with_low_memory_defaults(compile_fn: F) -> F:
    """Fill in the settings the caller doesn't pass from the enclosing ``low_memory`` blocks."""

    @functools.wraps(compile_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return compile_fn(*args, **{**_compile_defaults.get(), **kwargs})

    return cast(F, wrapper)


class low_memory:
    """Compile a model from its weights file without loading it, and set compile defaults for a block.

    With ``model_fn`` and ``weights``, entering the block builds the model on ``meta``, so it allocates
    no weights, maps its weights from the safetensors file ``weights`` with ``map_weights``, and returns
    it, on the CPU. Every tensor must be in the file. Export it with CPU inputs and compile it with
    ``torch_tensorrt.dynamo.compile``; each weight's pages are read while its engine is built and dropped
    once its last engine is built::

        with torch_tensorrt.dynamo.low_memory(lambda: MyModel(config), "weights.safetensors") as model:
            exported = torch.export.export(model, (example,))
            trt_model = torch_tensorrt.dynamo.compile(exported, inputs=[example.cuda()])

    The other arguments are the ``torch_tensorrt.dynamo.compile`` settings that lower memory during
    compilation. The block uses them as defaults for every call to it inside the block, also in code you
    don't own, and so for ``torch_tensorrt.compile`` with ``ir="dynamo"``.
    ``torch.compile(backend="tensorrt")``, ``cross_compile_for_windows`` and
    ``convert_exported_program_to_serialized_trt_engine`` don't use them. A setting left at ``None``
    isn't set, so an outer block's value or ``compile``'s default applies. An argument passed to
    ``compile`` wins over the block's, and an inner block's setting wins over an outer one's. The
    defaults are a ``contextvars`` context: leaving the block restores the previous ones, and a thread
    sees them only if it copies the context::

        with torch_tensorrt.dynamo.low_memory(
            enable_resource_partitioning=True, cpu_memory_budget=8 << 30
        ):
            run_export_script()

    On leaving, the pages of the model's mapped weights are dropped, also on an exception, unless the
    block sets ``release_weights_after_build=False``. The file stays mapped: the model still reads it, so
    keep it in place, unchanged, while the model is in use. A write to a mapped weight is lost when its
    pages are dropped.

    Args:
        model_fn: builds the model, without moving it to a device. Pass with ``weights``.
        weights: the safetensors file to map, in the model's names, dtypes and shapes
            (``save_weights_for_mapping`` writes one, non-persistent buffers included).
        offload_module_to_cpu: move the model's weights to the CPU before the engines are built, so
            peak GPU memory drops to about the engines' size while the CPU holds the weights. Mapped
            weights are on the CPU already.
        enable_resource_partitioning: split TensorRT subgraphs into engines small enough to build
            within ``cpu_memory_budget``.
        cpu_memory_budget: the host memory, in bytes, the compiling process may use, counting what it
            already uses. Read only with ``enable_resource_partitioning=True``; without a budget, the
            partitioner sizes engines to the system's available memory.
        release_weights_after_build: ``True`` also frees weights that aren't mapped, which the source
            model then can't use; ``False`` releases nothing, and leaving the block then keeps the
            pages too. ``compile``'s default drops only mapped weights' pages, once their last engine
            is built.

    Attributes:
        report: what ``map_weights`` did (bytes mapped, names missing); ``None`` without a model.
    """

    def __init__(
        self,
        model_fn: Optional[Callable[[], torch.nn.Module]] = None,
        weights: Optional[Union[str, "os.PathLike[str]"]] = None,
        *,
        offload_module_to_cpu: Optional[bool] = None,
        enable_resource_partitioning: Optional[bool] = None,
        cpu_memory_budget: Optional[int] = None,
        release_weights_after_build: Optional[bool] = None,
    ) -> None:
        if (model_fn is None) != (weights is None):
            raise ValueError("Pass model_fn and weights together, or neither")
        settings = {
            "offload_module_to_cpu": offload_module_to_cpu,
            "enable_resource_partitioning": enable_resource_partitioning,
            "cpu_memory_budget": cpu_memory_budget,
            "release_weights_after_build": release_weights_after_build,
        }
        self._model_fn = model_fn
        self._weights = weights
        self._settings = {k: v for k, v in settings.items() if v is not None}
        self._model: Optional[torch.nn.Module] = None
        self._token: Optional[Token[Mapping[str, Any]]] = None
        self.report: Optional[MappedWeights] = None

    def __enter__(self) -> Optional[torch.nn.Module]:
        if self._model_fn is not None:
            assert self._weights is not None
            with torch.device("meta"):
                model = self._model_fn()
            self.report = map_weights(model, self._weights, strict=True)
            on_meta = [
                name
                for name, t in itertools.chain(
                    model.named_parameters(), model.named_buffers()
                )
                if t.is_meta
            ]
            if on_meta:
                raise ValueError(
                    f"{len(on_meta)} tensors are not in {os.fspath(self._weights)}, so they have no "
                    f"values: {on_meta[:8]}. Write the file with save_weights_for_mapping, which "
                    "includes non-persistent buffers."
                )
            self._model = model
        self._token = _compile_defaults.set(
            MappingProxyType({**_compile_defaults.get(), **self._settings})
        )
        return self._model

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        assert self._token is not None
        release = (
            _compile_defaults.get().get("release_weights_after_build") is not False
        )
        _compile_defaults.reset(self._token)
        if self._model is not None and release:
            for t in itertools.chain(self._model.parameters(), self._model.buffers()):
                release_mapped_storage(t.untyped_storage())
