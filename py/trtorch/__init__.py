import os
import sys

if sys.version_info < (3,):
    raise Exception("Python 2 has reached end-of-life and is not supported by TRTorch")

import ctypes
import torch

from trtorch._version import __version__
from trtorch._compiler import *
from trtorch._types import *
from trtorch import logging
