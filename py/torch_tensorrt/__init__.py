import ctypes
import logging
import os
import platform
import sys
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
