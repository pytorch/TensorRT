import ctypes
import glob
import os
import sys
import platform
import warnings
from torch_tensorrt._version import (
    __version__,
    __cuda_version__,
    __cudnn_version__,
    __tensorrt_version__,
)

if sys.version_info < (3,):
    raise Exception(
        "Python 2 has reached end-of-life and is not supported by Torch-TensorRT"
    )


def _parse_semver(version):
    split = version.split(".")
    if len(split) < 3:
        split.append("")

    return {"major": split[0], "minor": split[1], "patch": split[2]}


def _find_lib(name, paths):
    for path in paths:
        libpath = os.path.join(path, name)
        if os.path.isfile(libpath):
            return libpath

    raise FileNotFoundError(f"Could not find {name}\n  Search paths: {paths}")


try:
    import tensorrt
except:
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
        LINUX_PATHS = [
            "/usr/local/cuda/lib64",
        ]

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

from torch_tensorrt._compile import *
from torch_tensorrt._util import *
from torch_tensorrt import ts
from torch_tensorrt import ptq
from torch_tensorrt._enums import *
from torch_tensorrt import logging
from torch_tensorrt._Input import Input
from torch_tensorrt._Device import Device

from torch_tensorrt import fx


def _register_with_torch():
    trtorch_dir = os.path.dirname(__file__)
    torch.ops.load_library(trtorch_dir + "/lib/libtorchtrt.so")


_register_with_torch()
