import os
import sys

if sys.version_info < (3,):
    raise Exception("Python 2 has reached end-of-life and is not supported by TRTorch")

import ctypes
import torch

from trtorch._version import __version__
from trtorch._compiler import *
from trtorch._compile_spec import TensorRTCompileSpec
from trtorch import ptq
from trtorch._types import *
from trtorch import logging
from trtorch.Input import Input
from trtorch.Device import Device


def _register_with_torch():
    trtorch_dir = os.path.dirname(__file__)
    torch.ops.load_library(trtorch_dir + '/lib/libtrtorch.so')


_register_with_torch()
