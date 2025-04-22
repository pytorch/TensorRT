import ctypes
import platform
import sys
from typing import Any

import torch
from torch_tensorrt import _find_lib


def sanitized_torch_version() -> Any:
    return (
        torch.__version__
        if ".nv" not in torch.__version__
        else torch.__version__.split(".nv")[0]
    )


def check_cross_compile_trt_win_lib():
    if sys.platform.startswith("linux"):
        LINUX_PATHS = ["/usr/local/cuda-12.8/lib64", "/usr/lib", "/usr/lib64"]

    if platform.uname().processor == "x86_64":
        LINUX_PATHS += [
            "/usr/lib/x86_64-linux-gnu",
        ]
    elif platform.uname().processor == "aarch64":
        LINUX_PATHS += ["/usr/lib/aarch64-linux-gnu"]

    LINUX_LIBS = [
        f"libnvinfer_builder_resource_win.so.*",
    ]

    for lib in LINUX_LIBS:
        try:
            ctypes.CDLL(_find_lib(lib, LINUX_PATHS))
            return True
        except:
            continue
    return False
