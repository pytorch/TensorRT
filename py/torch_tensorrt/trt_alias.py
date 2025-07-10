import importlib
import importlib.util
import os
import sys
from types import ModuleType
from typing import Any


def is_rtx_gpu() -> bool:
    try:
        import torch

        return "RTX" in torch.cuda.get_device_name(0)
    except ImportError:
        # fallback to tensorrt
        return False


# TensorRTProxyModule is a proxy module that allows us to use the tensorrt_rtx package if rtx gpu is detected
class TensorRTProxyModule(ModuleType):
    def __init__(self, target_module: ModuleType) -> None:
        spec = importlib.util.spec_from_loader("tensorrt", loader=None)
        self.__spec__ = spec
        self.__package__ = target_module.__package__
        self.__path__ = target_module.__path__
        self.__file__ = target_module.__file__
        self.__loader__ = target_module.__loader__
        self.__version__ = target_module.__version__
        self._target_module = target_module
        self._nested_module = None
        self._package_name: str = ""

        # For RTX: tensorrt.tensorrt -> tensorrt_rtx.tensorrt_rtx
        # For standard: tensorrt.tensorrt -> tensorrt.tensorrt (no change)
        if hasattr(target_module, "tensorrt_rtx"):
            self._nested_module = target_module.tensorrt_rtx
        elif hasattr(target_module, "tensorrt"):
            self._nested_module = target_module.tensorrt

        # Set up the nested module structure
        if self._nested_module:
            self.tensorrt = self._nested_module

    # __getattr__ is used to get the attribute from the target module
    def __getattr__(self, name: str) -> Any:
        # First try to get from the target module
        try:
            return getattr(self._target_module, name)
        except AttributeError:
            print(f"AttributeError: {name}")
            # For nested modules like tensorrt.tensorrt
            if name == "tensorrt" and self._nested_module:
                return self._nested_module
            raise

    def __dir__(self) -> list[str]:
        return dir(self._target_module)


def alias_tensorrt() -> None:
    # Determine package name with env override support for easy testing with tensorrt or tensorrt_rtx
    # eg: FORCE_TENSORRT_RTX=1 python test.py
    # eg: FORCE_TENSORRT_STD=1 python test.py
    use_rtx = False
    if os.environ.get("FORCE_TENSORRT_RTX", "0") == "1":
        use_rtx = True
    elif os.environ.get("FORCE_TENSORRT_STD", "0") == "1":
        use_rtx = False
    else:
        use_rtx = is_rtx_gpu()

    # Import the appropriate package
    try:
        if use_rtx:
            target = importlib.import_module("tensorrt_rtx")
        else:
            target = importlib.import_module("tensorrt")
    except ImportError:
        # Fallback to standard tensorrt if RTX version not available
        print(f"import error when {use_rtx=}, fallback to standard tensorrt")
        try:
            target = importlib.import_module("tensorrt")
            # since we are using the standard tensorrt, we need to set the use_rtx to True
            use_rtx = True
        except ImportError:
            raise RuntimeError("TensorRT package not found")

    proxy = TensorRTProxyModule(target)
    proxy._package_name = "tensorrt" if use_rtx else "tensorrt_rtx"

    sys.modules["tensorrt"] = proxy


alias_tensorrt()
