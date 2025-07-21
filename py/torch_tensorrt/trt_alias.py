import ctypes
import importlib
import importlib.util
import os
import platform
import sys
from types import ModuleType
from typing import Any, Dict, List

package_imported = False
package_name = ""


def _parse_semver(version: str) -> Dict[str, str]:
    split = version.split(".")
    if len(split) < 3:
        split.append("")

    return {"major": split[0], "minor": split[1], "patch": split[2]}


def _find_lib(name: str, paths: List[str]) -> str:
    for path in paths:
        libpath = os.path.join(path, name)
        if os.path.isfile(libpath):
            return libpath

    raise FileNotFoundError(f"Could not find {name}\n  Search paths: {paths}")


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
    global package_imported
    global package_name
    # tensorrt package has been imported, no need to alias again
    if package_imported:
        return

    # in order not to break or change the existing behavior, we only build and run with tensorrt by default, tensorrt-rtx is for experiment only
    # if we want to test with tensorrt-rtx, we have to build the wheel with --use-rtx and test with FORCE_TENSORRT_RTX=1
    # eg: FORCE_TENSORRT_RTX=1 python test.py
    # in future, we can do dynamic linking either to tensorrt or tensorrt-rtx based on the gpu type
    use_rtx = False
    if os.environ.get("FORCE_TENSORRT_RTX", "0") == "1":
        use_rtx = True
    package_name = "tensorrt_rtx" if use_rtx else "tensorrt"
    # Import the appropriate package
    try:
        target_module = importlib.import_module(package_name)
        proxy = TensorRTProxyModule(target_module)
        proxy._package_name = package_name
        sys.modules["tensorrt"] = proxy
        package_imported = True
    except ImportError as e:
        # Fallback to standard tensorrt if RTX version not available
        print(f"import error when try to import {package_name=} got error {e}")
        print(
            f"make sure tensorrt lib is in the LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}"
        )
        if use_rtx:
            from torch_tensorrt import __tensorrt_rtx_version__

            tensorrt_version = _parse_semver(__tensorrt_rtx_version__)
            tensorrt_major = tensorrt_version["major"]
            tensorrt_minor = tensorrt_version["minor"]
            tensorrt_lib = {
                "win": [
                    f"tensorrt_rtx_{tensorrt_major}_{tensorrt_minor}.dll",
                ],
                "linux": [
                    f"libtensorrt_rtx.so.{tensorrt_major}",
                ],
            }
        else:
            from torch_tensorrt import __tensorrt_version__

            tensorrt_version = _parse_semver(__tensorrt_version__)
            tensorrt_major = tensorrt_version["major"]
            tensorrt_minor = tensorrt_version["minor"]
            tensorrt_lib = {
                "win": [
                    f"nvinfer_{tensorrt_major}.dll",
                    f"nvinfer_plugin_{tensorrt_major}.dll",
                ],
                "linux": [
                    f"libnvinfer.so.{tensorrt_major}",
                    f"libnvinfer_plugin.so.{tensorrt_major}",
                ],
            }

        from torch_tensorrt import __cuda_version__

        if sys.platform.startswith("win"):
            WIN_LIBS = tensorrt_lib["win"]
            WIN_PATHS = os.environ["PATH"].split(os.path.pathsep)
            for lib in WIN_LIBS:
                ctypes.CDLL(_find_lib(lib, WIN_PATHS))

        elif sys.platform.startswith("linux"):
            LINUX_PATHS = [
                f"/usr/local/cuda-{__cuda_version__}/lib64",
                "/usr/lib",
                "/usr/lib64",
            ]
            if "LD_LIBRARY_PATH" in os.environ:
                LINUX_PATHS += os.environ["LD_LIBRARY_PATH"].split(os.path.pathsep)
            if platform.uname().processor == "x86_64":
                LINUX_PATHS += [
                    "/usr/lib/x86_64-linux-gnu",
                ]
            elif platform.uname().processor == "aarch64":
                LINUX_PATHS += ["/usr/lib/aarch64-linux-gnu"]
            LINUX_LIBS = tensorrt_lib["linux"]
            for lib in LINUX_LIBS:
                ctypes.CDLL(_find_lib(lib, LINUX_PATHS))


alias_tensorrt()
