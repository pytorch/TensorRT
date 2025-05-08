import ctypes
import platform
import sys
from typing import Any

import torch


def sanitized_torch_version() -> Any:
    return (
        torch.__version__
        if ".nv" not in torch.__version__
        else torch.__version__.split(".nv")[0]
    )


def check_cross_compile_trt_win_lib():
    # cross compile feature is only available on linux
    # build engine on linux and run on windows
    import dllist

    if sys.platform.startswith("linux"):
        loaded_libs = dllist.dllist()
        target_lib = "libnvinfer_builder_resource_win.so.*"
        if target_lib in loaded_libs:
            return True
    return False
