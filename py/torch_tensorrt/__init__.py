import ctypes
import os
import platform
import sys
from typing import Dict, List

from torch_tensorrt._version import (  # noqa: F401
    __cuda_version__,
    __cudnn_version__,
    __tensorrt_version__,
    __version__,
)

from packaging import version

if sys.version_info < (3,):
    raise Exception(
        "Python 2 has reached end-of-life and is not supported by Torch-TensorRT"
    )


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


try:
    import tensorrt  # noqa: F401
except ImportError:
    cuda_version = _parse_semver(__cuda_version__)
    cudnn_version = _parse_semver(__cudnn_version__)
    tensorrt_version = _parse_semver(__tensorrt_version__)

    CUDA_MAJOR = cuda_version["major"]
    CUDNN_MAJOR = cudnn_version["major"]
    TENSORRT_MAJOR = tensorrt_version["major"]

    if sys.platform.startswith("win"):
        WIN_LIBS = [
            "nvinfer.dll",
            "nvinfer_plugin.dll",
        ]

        WIN_PATHS = os.environ["PATH"].split(os.path.pathsep)

        for lib in WIN_LIBS:
            ctypes.CDLL(_find_lib(lib, WIN_PATHS))

    elif sys.platform.startswith("linux"):
        LINUX_PATHS = ["/usr/local/cuda-12.1/lib64", "/usr/lib", "/usr/lib64"]

        if "LD_LIBRARY_PATH" in os.environ:
            LINUX_PATHS += os.environ["LD_LIBRARY_PATH"].split(os.path.pathsep)

        if platform.uname().processor == "x86_64":
            LINUX_PATHS += [
                "/usr/lib/x86_64-linux-gnu",
            ]

        elif platform.uname().processor == "aarch64":
            LINUX_PATHS += ["/usr/lib/aarch64-linux-gnu"]

        LINUX_LIBS = [
            f"libnvinfer.so.{TENSORRT_MAJOR}",
            f"libnvinfer_plugin.so.{TENSORRT_MAJOR}",
        ]

        for lib in LINUX_LIBS:
            ctypes.CDLL(_find_lib(lib, LINUX_PATHS))

import torch
from torch_tensorrt._compile import *  # noqa: F403
from torch_tensorrt._Device import Device  # noqa: F401
from torch_tensorrt._enums import *  # noqa: F403
from torch_tensorrt._Input import Input  # noqa: F401
from torch_tensorrt._utils import *  # noqa: F403
from torch_tensorrt._utils import sanitized_torch_version
from torch_tensorrt.logging import *
from torch_tensorrt.ptq import *
from torch_tensorrt.runtime import *  # noqa: F403

if version.parse(sanitized_torch_version()) >= version.parse("2.1.dev"):
    from torch_tensorrt.dynamo import backend  # noqa: F401

    from torch_tensorrt import dynamo  # noqa: F401


def _register_with_torch() -> None:
    trtorch_dir = os.path.dirname(__file__)
    torch.ops.load_library(trtorch_dir + "/lib/libtorchtrt.so")


_register_with_torch()
