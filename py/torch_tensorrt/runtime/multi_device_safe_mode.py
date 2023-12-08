import logging
from importlib.util import find_spec
from typing import Any

import torch

if find_spec("torch_tensorrt._C") is not None:
    _PY_RT_MULTI_DEVICE_SAFE_MODE = torch.ops.tensorrt.get_multi_device_safe_mode()
else:
    _PY_RT_MULTI_DEVICE_SAFE_MODE = False


logger = logging.getLogger(__name__)


class _MultiDeviceSafeModeContextManager(object):
    """Helper class used in conjunction with `set_multi_device_safe_mode`

    Used to enable `set_multi_device_safe_mode` as a dual-purpose context manager
    """

    def __init__(self, old_mode: bool) -> None:
        self.old_mode = old_mode

    def __enter__(self) -> "_MultiDeviceSafeModeContextManager":
        return self

    def __exit__(self, *args: Any) -> None:
        # Set multi-device safe mode back to old mode in Python
        global _PY_RT_MULTI_DEVICE_SAFE_MODE
        _PY_RT_MULTI_DEVICE_SAFE_MODE = self.old_mode

        # Set multi-device safe mode back to old mode in C++
        if find_spec("torch_tensorrt._C") is not None:
            torch.ops.tensorrt.set_multi_device_safe_mode(self.old_mode)


def set_multi_device_safe_mode(mode: bool) -> _MultiDeviceSafeModeContextManager:
    # Fetch existing safe mode and set new mode for Python
    global _PY_RT_MULTI_DEVICE_SAFE_MODE
    old_mode = _PY_RT_MULTI_DEVICE_SAFE_MODE
    _PY_RT_MULTI_DEVICE_SAFE_MODE = mode

    # Set new mode for C++
    if find_spec("torch_tensorrt._C") is not None:
        torch.ops.tensorrt.set_multi_device_safe_mode(mode)

    logger.info(f"Set multi-device safe mode to {mode}")

    # Return context manager in case the function is used in a `with` call
    return _MultiDeviceSafeModeContextManager(old_mode)
