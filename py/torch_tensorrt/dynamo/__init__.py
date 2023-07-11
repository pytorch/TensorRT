import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse("2.1.dev"):
    from torch_tensorrt.dynamo import fx_ts_compat
    from .backend import compile
