import logging
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

    def __enter__(self) -> Union[torch.nn.Module, torch.fx.GraphModule]:
        # Apply the RTX cuda-graph strategy BEFORE the wrapper's ``warm_up``
        # materializes the engine's ``IExecutionContext``. Open the
        # ``runtime_config`` CM here so the strategy is live for the next
        # ``get_cuda_graph_module`` call.
        if self._cuda_graph_strategy is not None:
            from torch_tensorrt.runtime._runtime_config import runtime_config

            self._inner_cm = runtime_config(
                self.compiled_module,
                cuda_graph_strategy=self._cuda_graph_strategy,
            )
            self._inner_cm.__enter__()

        if isinstance(self.compiled_module, torch_tensorrt.MutableTorchTensorRTModule):
            self.old_module = self.compiled_module.gm
            self.compiled_module.gm = get_cuda_graph_module(self.compiled_module.gm)
            return self.compiled_module
        else:
            return get_cuda_graph_module(self.compiled_module)

    def __exit__(self, *args: Any) -> None:
        # Restore the exact integer mode (0=STANDARD, 1=SUBGRAPH, 2=WHOLE_GRAPH).
        # Calling set_cudagraphs_mode(bool) would coerce 2 → 1 and can't restore
        # WHOLE_GRAPH_CUDAGRAPHS, so we write the global and C++ op directly.
        global _PY_RT_CUDAGRAPHS
        _PY_RT_CUDAGRAPHS = self.old_mode
        if torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
            torch.ops.tensorrt.set_cudagraphs_mode(self.old_mode)
        # __del__ is not entirely predictable, so we reset cudagraph here
        if self.cudagraphs_module:
            self.cudagraphs_module._reset_captured_graph()
        if self.old_module:  # MutableTorchTRTModule
            self.compiled_module.gm = self.old_module
        # Restore prior strategy state on the engines (after the wrapper is
        # gone). Reverse order would leave the wrapper attached to engines
        # whose settings have already been restored.
        if self._inner_cm is not None:
            self._inner_cm.__exit__(*args)


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
