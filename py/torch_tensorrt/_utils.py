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


def is_tensorrt_version_supported(min_version: str = "10.8.0") -> bool:
    """
    Check if the installed TensorRT version supports the specified minimum version.
    Args:
        min_version (str): Minimum required TensorRT version (default: "10.8.0" for FP4 support)
    Returns:
        bool: True if TensorRT version is >= min_version, False otherwise
    Example:
        >>> if is_tensorrt_version_supported("10.8.0"):
        ...     # Use FP4 features
        ...     pass
    """
    try:
        from importlib import metadata

        from packaging.version import Version

        return bool(Version(metadata.version("tensorrt")) >= Version(min_version))
    except (ImportError, ValueError):
        # If tensorrt is not installed or version cannot be determined
        return False
