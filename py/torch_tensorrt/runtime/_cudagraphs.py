# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import operator
import sys
from typing import Any, Optional, Union

import torch

import torch_tensorrt
from torch_tensorrt.dynamo.runtime._CudaGraphsTorchTensorRTModule import (
    CudaGraphsTorchTensorRTModule,
)


class CudaGraphsMode:
    # No cuda graphs
    STANDARD = 0
    # Cuda graphs is applied to TRT module
    SUBGRAPH_CUDAGRAPHS = 1
    # Internal mode to apply cuda graphs for wrapped runtime module
    WHOLE_GRAPH_CUDAGRAPHS = 2


if torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
    _PY_RT_CUDAGRAPHS = torch.ops.tensorrt.get_cudagraphs_mode()
else:
    _PY_RT_CUDAGRAPHS = CudaGraphsMode.STANDARD


logger = logging.getLogger(__name__)


def set_cudagraphs_mode(mode: bool) -> None:
    # Set new cudagraphs mode for Python
    global _PY_RT_CUDAGRAPHS
    _PY_RT_CUDAGRAPHS = (
        CudaGraphsMode.SUBGRAPH_CUDAGRAPHS if mode else CudaGraphsMode.STANDARD
    )

    # Set new mode for C++
    if torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
        torch.ops.tensorrt.set_cudagraphs_mode(_PY_RT_CUDAGRAPHS)

    logger.info(f"Set Cudagraphs usage to {mode}")


def get_whole_cudagraphs_mode() -> bool:
    # check if whole cudagraphs mode is enabled or not
    global _PY_RT_CUDAGRAPHS
    if _PY_RT_CUDAGRAPHS == CudaGraphsMode.WHOLE_GRAPH_CUDAGRAPHS:
        return True
    else:
        return False


def get_cudagraphs_mode() -> bool:
    # Get cudagraphs mode for Python
    global _PY_RT_CUDAGRAPHS
    if _PY_RT_CUDAGRAPHS == CudaGraphsMode.SUBGRAPH_CUDAGRAPHS:
        return True
    else:
        return False


class _CudagraphsContextManager(object):
    """Helper class used in conjunction with `enable_cudagraphs`

    Used to enable cudagraphs as a context manager
    """

    def __init__(
        self,
        compiled_module: torch.nn.Module,
        cuda_graph_strategy: Optional[str] = None,
    ) -> None:
        global _PY_RT_CUDAGRAPHS
        self.old_mode = _PY_RT_CUDAGRAPHS
        self.compiled_module = compiled_module
        self._cuda_graph_strategy = cuda_graph_strategy
        self._inner_cm: Any = None
        self.cudagraphs_module: Optional[CudaGraphsTorchTensorRTModule] = None
        self.old_module = None
        self._active = False

    def __enter__(self) -> Union[torch.nn.Module, torch.fx.GraphModule]:
        if self._active:
            raise RuntimeError("A CUDA graph context cannot re-enter itself")
        self._active = True
        # Snapshot on entry: this context may have been constructed before a
        # surrounding CUDA-graph context was entered.
        self.old_mode = _PY_RT_CUDAGRAPHS
        try:
            # Apply the RTX strategy before warmup materializes the engine's
            # execution context, and undo it if wrapper construction fails.
            if self._cuda_graph_strategy is not None:
                from torch_tensorrt.runtime._runtime_config import runtime_config

                inner_cm = runtime_config(
                    self.compiled_module,
                    cuda_graph_strategy=self._cuda_graph_strategy,
                )
                inner_cm.__enter__()
                self._inner_cm = inner_cm

            mutable = isinstance(
                self.compiled_module, torch_tensorrt.MutableTorchTensorRTModule
            )
            if mutable:
                self.old_module = self.compiled_module.gm
                wrapped = get_cuda_graph_module(self.old_module)
                self.compiled_module.gm = wrapped
            else:
                wrapped = get_cuda_graph_module(self.compiled_module)
            if isinstance(wrapped, CudaGraphsTorchTensorRTModule):
                self.cudagraphs_module = wrapped
            return self.compiled_module if mutable else wrapped
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *args: Any) -> None:
        self._active = False
        # Restore the exact integer mode (0=STANDARD, 1=SUBGRAPH, 2=WHOLE_GRAPH).
        # Calling set_cudagraphs_mode(bool) would coerce 2 → 1 and can't restore
        # WHOLE_GRAPH_CUDAGRAPHS, so we write the global and C++ op directly.
        global _PY_RT_CUDAGRAPHS
        _PY_RT_CUDAGRAPHS = self.old_mode
        if torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
            torch.ops.tensorrt.set_cudagraphs_mode(self.old_mode)
        try:
            # __del__ is not predictable, so release the captured graph here.
            if self.cudagraphs_module is not None:
                self.cudagraphs_module._reset_captured_graph()
                self.cudagraphs_module = None
        finally:
            if self.old_module is not None:  # MutableTorchTRTModule
                self.compiled_module.gm = self.old_module
                self.old_module = None
            # Restore strategy after detaching the wrapper, even if graph
            # cleanup failed. A failed __enter__ also follows this path.
            if self._inner_cm is not None:
                inner_cm, self._inner_cm = self._inner_cm, None
                inner_cm.__exit__(*args)


def get_cuda_graph_module(
    compiled_module: torch.fx.GraphModule,
) -> Union[torch.nn.Module, torch.fx.GraphModule]:
    global _PY_RT_CUDAGRAPHS

    num_torch_module = 0
    num_trt_module = 0
    for name, module in compiled_module.named_children():
        # need to disable cudagraphs if any model requires output allocator
        if (
            hasattr(module, "requires_output_allocator")
            and module.requires_output_allocator
        ):
            raise RuntimeError(
                "The model contains submodules that require a dynamic output allocator at runtime, which is incompatible with CUDA Graphs. Please disable CUDA Graphs."
            )
        if "_run_on_acc" in name:
            num_trt_module += 1
        elif "_run_on_gpu" in name:
            num_torch_module += 1

    # The global partitioner leaves PyTorch work in the parent graph rather
    # than naming a fallback child _run_on_gpu. Explicit regions also retain
    # their own child names, including when the whole model stays in PyTorch.
    if isinstance(compiled_module, torch.fx.GraphModule):
        from torch_tensorrt.dynamo.regions import get_region_records

        if any(
            record.route == "torch" for record in get_region_records(compiled_module)
        ) or any(
            node.op in ("call_function", "call_method")
            and node.target is not operator.getitem
            for node in compiled_module.graph.nodes
        ):
            num_torch_module += 1

    if num_torch_module > 0:
        # Set whole cudagraphs mode and returns wrapped module
        _PY_RT_CUDAGRAPHS = CudaGraphsMode.WHOLE_GRAPH_CUDAGRAPHS
        # Set new mode for C++
        if torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
            torch.ops.tensorrt.set_cudagraphs_mode(_PY_RT_CUDAGRAPHS)

        logger.debug(
            "Found pytorch subgraphs in module, wrapping module in CudaGraphsTorchTensorRTModule"
        )
        return CudaGraphsTorchTensorRTModule(compiled_module)
    else:
        if num_trt_module > 0:
            logger.debug("No graph breaks detected, using runtime cudagraphs mode")
        else:
            logger.debug(
                "Please consider dynamo if there is graph breaks. Using runtime cudagraphs mode"
            )
        # Enable cudagraphs for TRT submodule
        set_cudagraphs_mode(True)
        return compiled_module


def enable_cudagraphs(
    compiled_module: Union[torch.fx.GraphModule, torch.nn.Module],
    *,
    cuda_graph_strategy: Optional[str] = None,
) -> _CudagraphsContextManager:
    """Wrap ``compiled_module`` for outer torch.cuda.CUDAGraph capture/replay.

    ``cuda_graph_strategy`` (TRT-RTX-only) collapses the formerly-nested
    ``runtime_config(..., cuda_graph_strategy=...)`` + ``enable_cudagraphs``
    pair into a single context manager. The strategy is applied state-only
    before the wrapper's ``warm_up`` materializes the engine's
    ``IExecutionContext`` and restored on exit -- one
    ``createExecutionContext`` call total.
    """
    if (
        cuda_graph_strategy is not None
        and not torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx
    ):
        raise RuntimeError(
            "`cuda_graph_strategy` is TRT-RTX-only; this is a non-RTX build. "
            "Drop the kwarg or build against TensorRT-RTX."
        )
    return _CudagraphsContextManager(
        compiled_module, cuda_graph_strategy=cuda_graph_strategy
    )
