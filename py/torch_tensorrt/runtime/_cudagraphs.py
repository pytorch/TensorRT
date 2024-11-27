import logging
from typing import Any, Optional

import torch
import torch_tensorrt
from torch_tensorrt.dynamo.runtime._WrapperTorchTensorRTModule import (
    WrapperTorchTensorRTModule,
)


class CudaGraphsMode:
    STANDARD = 0
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

    def __init__(self, compiled_module: Optional[torch.nn.Module]) -> None:
        global _PY_RT_CUDAGRAPHS
        self.old_mode = _PY_RT_CUDAGRAPHS
        self.compiled_module = compiled_module

    def __enter__(self) -> "_CudagraphsContextManager":
        global _PY_RT_CUDAGRAPHS
        if self.compiled_module:
            _PY_RT_CUDAGRAPHS = CudaGraphsMode.WHOLE_GRAPH_CUDAGRAPHS
            # Set new mode for C++
            if torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
                torch.ops.tensorrt.set_cudagraphs_mode(_PY_RT_CUDAGRAPHS)
            return WrapperTorchTensorRTModule(self.compiled_module)
        else:
            # Enable cudagraphs
            set_cudagraphs_mode(True)
            return self

    def __exit__(self, *args: Any) -> None:
        # Set cudagraphs back to old mode
        set_cudagraphs_mode(self.old_mode)


def enable_cudagraphs(
    compiled_module: Optional[torch.nn.Module] = None,
) -> _CudagraphsContextManager:
    return _CudagraphsContextManager(compiled_module)
