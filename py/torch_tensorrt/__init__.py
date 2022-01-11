import os
import sys

if sys.version_info < (3,):
    raise Exception("Python 2 has reached end-of-life and is not supported by Torch-TensorRT")

import ctypes
import torch

from torch_tensorrt._version import __version__
from torch_tensorrt._compile import *
from torch_tensorrt._util import *
from torch_tensorrt import ts
from torch_tensorrt import ptq
from torch_tensorrt._enums import *
from torch_tensorrt import logging
from torch_tensorrt._Input import Input
from torch_tensorrt._Device import Device


def _register_with_torch():
    trtorch_dir = os.path.dirname(__file__)
    torch.ops.load_library(trtorch_dir + '/lib/libtorchtrt.so')


_register_with_torch()
