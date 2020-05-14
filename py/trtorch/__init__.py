import os
import sys

if sys.version_info < (3,):
    raise Exception("Python 2 has reached end-of-life and is not supported by TRTorch")

import ctypes
import torch

def _load_trtorch_lib():
    lib_name = 'libtrtorch.so'
    here = os.path.abspath(__file__)
    lib_path = os.path.join(os.path.dirname(here), 'lib', lib_name)
    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)

_load_trtorch_lib()

from .version import __version__
from trtorch import _C
from trtorch.compiler import *
from trtorch.types import *

def test(mod, data):
    _C._test(mod._c, data)
