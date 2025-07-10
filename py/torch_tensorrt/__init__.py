import ctypes
import logging
import os
import platform
import sys
from typing import Dict, List

import torch
from torch_tensorrt._version import (  # noqa: F401
    __cuda_version__,
    __tensorrt_rtx_version__,
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


import logging

import torch

try:
    import tensorrt  # noqa: F401

    from . import trt_alias  # noqa: F401

    print(f"You are using {tensorrt.__name__=} {tensorrt.__version__=}")
except ImportError:
    tensorrt_version = _parse_semver(__tensorrt_version__)
    tensorrt_rtx_version = _parse_semver(__tensorrt_rtx_version__)

    TENSORRT_MAJOR = tensorrt_version["major"]
    TENSORRT_MINOR = tensorrt_version["minor"]
    TENSORRT_RTX_MAJOR = tensorrt_rtx_version["major"]
    TENSORRT_RTX_MINOR = tensorrt_rtx_version["minor"]

    trt_lib = {
        "tensorrt_rtx": {
            "win": [
                f"tensorrt_rtx_{TENSORRT_RTX_MAJOR}_{TENSORRT_RTX_MINOR}.dll",
                # TODO: lan to verify, comment out for now, as torch-tensorrt don't require onnx parser
                # f"tensorrt_onnxparser_rtx_{TENSORRT_RTX_MAJOR}_{TENSORRT_RTX_MINOR}.dll",
            ],
            "linux": [
                f"libtensorrt_rtx.so.{TENSORRT_RTX_MAJOR}",
                # TODO: lan to verify, comment out for now, as torch-tensorrt don't require onnx parser
                # f"libtensorrt_onnxparser_rtx.so.{TENSORRT_RTX_MAJOR}",
            ],
        },
        "tensorrt": {
            "win": [
                f"nvinfer_{TENSORRT_MAJOR}.dll",
                f"nvinfer_plugin_{TENSORRT_MAJOR}.dll",
            ],
            "linux": [
                f"libnvinfer.so.{TENSORRT_MAJOR}",
                f"libnvinfer_plugin.so.{TENSORRT_MAJOR}",
            ],
        },
    }
    if sys.platform.startswith("win"):
        WIN_LIBS = trt_lib[tensorrt.__name__]["win"]
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
        LINUX_LIBS = trt_lib[tensorrt.__name__]["linux"]
        for lib in LINUX_LIBS:
            ctypes.CDLL(_find_lib(lib, LINUX_PATHS))


def _register_with_torch() -> None:
    trtorch_dir = os.path.dirname(__file__)
    linked_file = os.path.join(
        "lib", ("torchtrt.dll" if sys.platform.startswith("win") else "libtorchtrt.so")
    )
    linked_file_runtime = os.path.join(
        "lib",
        (
            "torchtrt_runtime.dll"
            if sys.platform.startswith("win")
            else "libtorchtrt_runtime.so"
        ),
    )
    linked_file_full_path = os.path.join(trtorch_dir, linked_file)
    linked_file_runtime_full_path = os.path.join(trtorch_dir, linked_file_runtime)

    if os.path.isfile(linked_file_full_path):
        assert ENABLED_FEATURES.torchscript_frontend
        assert ENABLED_FEATURES.torch_tensorrt_runtime
        torch.ops.load_library(linked_file_full_path)

    elif os.path.isfile(linked_file_runtime_full_path):
        assert ENABLED_FEATURES.torch_tensorrt_runtime
        torch.ops.load_library(linked_file_runtime_full_path)


# note: trt_alias must be imported before enabled features, because enabled features will check tensorrt.plugin availability
from torch_tensorrt._features import ENABLED_FEATURES, _enabled_features_str

_LOGGER = logging.getLogger(__name__)
_LOGGER.debug(_enabled_features_str())


_register_with_torch()

from torch_tensorrt._Device import Device  # noqa: F401
from torch_tensorrt._enums import (  # noqa: F401
    DeviceType,
    EngineCapability,
    Platform,
    dtype,
    memory_format,
)
from torch_tensorrt._Input import Input  # noqa: F401
from torch_tensorrt.runtime import *  # noqa: F403

if ENABLED_FEATURES.torchscript_frontend:
    from torch_tensorrt import ts

if ENABLED_FEATURES.fx_frontend:
    from torch_tensorrt import fx

if ENABLED_FEATURES.dynamo_frontend:
    from torch_tensorrt.dynamo import backend  # noqa: F401
    from torch_tensorrt import dynamo  # noqa: F401

from torch_tensorrt._compile import *  # noqa: F403
from torch_tensorrt.dynamo.runtime._MutableTorchTensorRTModule import (
    MutableTorchTensorRTModule,
)
