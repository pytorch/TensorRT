import ctypes
import logging
import os
import platform
import sys
import tempfile
from typing import Dict, List

import torch
from torch_tensorrt._version import (  # noqa: F401
    __cuda_version__,
    __version__,
)

from packaging import version

if sys.version_info < (3,):
    raise Exception(
        "Python 2 has reached end-of-life and is not supported by Torch-TensorRT"
    )

import logging

_LOGGER = logging.getLogger(__name__)

import torch


def is_capture_tensorrt_api_recording_enabled() -> bool:
    if os.environ.get("TORCHTRT_ENABLE_TENSORRT_API_CAPTURE") == "1":
        if not sys.platform.startswith("linux"):
            _LOGGER.warning(
                f"Capturing TensorRT API calls is only supported on Linux, therefore ignoring the capture_tensorrt_api_recording setting for {sys.platform}"
            )
            os.environ.pop("TORCHTRT_ENABLE_TENSORRT_API_CAPTURE")
            return False
        if os.environ.get("USE_TRT_RTX", "False").lower() == "true":
            _LOGGER.warning(
                "Capturing TensorRT API calls is only supported on TensorRT, therefore ignoring the capture_tensorrt_api_recording setting for TensorRT-RTX"
            )
            os.environ.pop("TORCHTRT_ENABLE_TENSORRT_API_CAPTURE")
            return False
        return True
    return False


if is_capture_tensorrt_api_recording_enabled():
    linux_lib_path = []
    if "LD_LIBRARY_PATH" in os.environ:
        linux_lib_path.extend(os.environ["LD_LIBRARY_PATH"].split(os.path.pathsep))

    if platform.uname().processor == "x86_64":
        linux_lib_path.append("/usr/lib/x86_64-linux-gnu")
    elif platform.uname().processor == "aarch64":
        linux_lib_path.append("/usr/lib/aarch64-linux-gnu")

    tensorrt_lib_path = None
    for path in linux_lib_path:
        try:
            ctypes.CDLL(
                os.path.join(path, "libtensorrt_shim.so"), mode=ctypes.RTLD_GLOBAL
            )
            tensorrt_lib_path = path
            break
        except Exception as e:
            continue

    if tensorrt_lib_path is None:
        _LOGGER.error(
            "Capturing TensorRT API calls is enabled, but libtensorrt_shim.so is not found, make sure TensorRT lib is in the LD_LIBRARY_PATH, therefore ignoring the capture_tensorrt_api_recording setting"
        )
        os.environ.pop("TORCHTRT_ENABLE_TENSORRT_API_CAPTURE")
    else:
        os.environ["TRT_SHIM_NVINFER_LIB_NAME"] = os.path.join(
            tensorrt_lib_path, "libnvinfer.so"
        )

        import pwd

        current_user = pwd.getpwuid(os.getuid())[0]
        shim_temp_dir = os.path.join(
            tempfile.gettempdir(), f"torch_tensorrt_{current_user}/shim"
        )
        os.makedirs(shim_temp_dir, exist_ok=True)
        os.environ["TRT_SHIM_OUTPUT_JSON_FILE"] = os.path.join(
            shim_temp_dir, "shim.json"
        )
        _LOGGER.debug("capture_shim feature is enabled")
else:
    _LOGGER.info("capture_shim feature is disabled")

tensorrt_package_name = ""

try:
    # note: _TensorRTProxyModule must be imported before any import tensorrt

    from . import _TensorRTProxyModule  # noqa: F401

    tensorrt_package_name = _TensorRTProxyModule.package_name
    _LOGGER.info(f"You are using {_TensorRTProxyModule.package_name=} ")

except Exception as e:
    print(f"import error when try to import _TensorRTProxyModule, got error {e}")
    print(
        f"make sure tensorrt lib is in the LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}"
    )
    raise Exception(
        f"import error when try to import _TensorRTProxyModule, got error {e}"
    )


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


# note: _TensorRTProxyModule must be imported before enabled features, because enabled features will check tensorrt.plugin availability
from torch_tensorrt._features import ENABLED_FEATURES, _enabled_features_str

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
