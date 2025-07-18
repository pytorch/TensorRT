import importlib
import importlib.util
import os
import sys
from types import ModuleType
from typing import Any

tensorrt_package_name = ""
tensorrt_package_imported = False


# TensorRTProxyModule is a proxy module that allows us to register the tensorrt or tensorrt-rtx package
# since tensorrt-rtx is the drop-in replacement for tensorrt, we can use the same interface to use tensorrt-rtx
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
    global tensorrt_package_imported
    # tensorrt package has been imported, no need to alias again
    if tensorrt_package_imported:
        return

    # in order not to break or change the existing behavior, we only build and run with tensorrt by default, tensorrt-rtx is for experiment only
    # if we want to test with tensorrt-rtx, we have to build the wheel with --use-rtx and test with FORCE_TENSORRT_RTX=1
    # eg: FORCE_TENSORRT_RTX=1 python test.py
    # in future, we can do dynamic linking either to tensorrt or tensorrt-rtx based on the gpu type
    use_rtx = False
    if os.environ.get("FORCE_TENSORRT_RTX", "0") == "1":
        use_rtx = True

    global tensorrt_package_name
    tensorrt_package_name = "tensorrt_rtx" if use_rtx else "tensorrt"
    # Import the appropriate package
    try:
        target_module = importlib.import_module(tensorrt_package_name)
        proxy = TensorRTProxyModule(target_module)
        proxy._package_name = tensorrt_package_name
        sys.modules["tensorrt"] = proxy
        tensorrt_package_imported = True
    except ImportError as e:
        # Fallback to standard tensorrt if RTX version not available
        print(f"import error when try to import {tensorrt_package_name=} got error {e}")
        print(
            f"make sure tensorrt lib is in the LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}"
        )
        raise Exception(
            f"import error when try to import {tensorrt_package_name=} got error {e}"
        )


alias_tensorrt()
