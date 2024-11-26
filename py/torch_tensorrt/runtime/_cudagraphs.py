import logging
from typing import Any, Optional

import torch
import torch_tensorrt
from torch_tensorrt.dynamo.runtime._WrapperTorchTensorRTModule import (
    WrapperTorchTensorRTModule,
)

if torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
    _PY_RT_CUDAGRAPHS = torch.ops.tensorrt.get_cudagraphs_mode()
else:
    _PY_RT_CUDAGRAPHS = False


logger = logging.getLogger(__name__)


def set_cudagraphs_mode(mode: bool) -> None:
    # Set new cudagraphs mode for Python
    global _PY_RT_CUDAGRAPHS
    _PY_RT_CUDAGRAPHS = mode

    # Set new mode for C++
    if torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
        torch.ops.tensorrt.set_cudagraphs_mode(mode)

    logger.info(f"Set Cudagraphs usage to {mode}")


def get_cudagraphs_mode() -> bool:
    # Get cudagraphs mode for Python
    global _PY_RT_CUDAGRAPHS
    return _PY_RT_CUDAGRAPHS  # type: ignore


class _CudagraphsContextManager(object):
    """Helper class used in conjunction with `enable_cudagraphs`

    Used to enable cudagraphs as a context manager
    """

    def __init__(self, module_to_wrap: Optional[torch.nn.Module]) -> None:
        global _PY_RT_CUDAGRAPHS
        self.old_mode = _PY_RT_CUDAGRAPHS
        self.module_to_wrap = module_to_wrap

    def __enter__(self) -> "_CudagraphsContextManager":
        # Enable cudagraphs
        set_cudagraphs_mode(True)
        if self.module_to_wrap:
            return WrapperTorchTensorRTModule(self.module_to_wrap)
        else:
            return self

    def __exit__(self, *args: Any) -> None:
        # Set cudagraphs back to old mode
        set_cudagraphs_mode(self.old_mode)


def enable_cudagraphs(
    module_to_wrap: Optional[torch.nn.Module] = None,
) -> _CudagraphsContextManager:
    return _CudagraphsContextManager(module_to_wrap)
