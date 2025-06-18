import re
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
        import dllist

        pattern = re.compile(r".*libnvinfer_builder_resource_win\.so(\.\d+)*$")

        loaded_libs = dllist.dllist()
        for lib in loaded_libs:
            if pattern.match(lib):
                return True
    return False
