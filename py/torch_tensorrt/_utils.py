import sys
from typing import Any

import torch


def sanitized_torch_version() -> Any:
    return (
        torch.__version__
        if ".nv" not in torch.__version__
        else torch.__version__.split(".nv")[0]
    )


def check_cross_compile_trt_win_lib() -> bool:
    # cross compile feature is only available on linux
    # build engine on linux and run on windows
    if sys.platform.startswith("linux"):
        import re

        import dllist

        loaded_libs = dllist.dllist()
        target_lib = ".*libnvinfer_builder_resource_win.so.*"
        return any(re.match(target_lib, lib) for lib in loaded_libs)
    return False
