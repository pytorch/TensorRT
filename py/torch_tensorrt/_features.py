import os
from packaging import version
from collections import namedtuple

from torch_tensorrt._version import __version__
from torch_tensorrt._utils import sanitized_torch_version

FeatureSet = namedtuple('FeatureSet', [
    "torchscript_frontend",
    "torch_tensorrt_runtime",
    "dynamo_frontend",
    "fx_frontend"
])

_TS_FE_AVAIL = os.path.isfile(os.path.dirname(__file__) + "/lib/libtorchtrt.so")
_TORCHTRT_RT_AVAIL = (_TS_FE_AVAIL or os.path.isfile(os.path.dirname(__file__) + "/lib/libtorchtrt_runtime.so"))
_DYNAMO_FE_AVAIL = version.parse(sanitized_torch_version()) >= version.parse("2.1.dev")
_FX_FE_AVAIL = True

ENABLED_FEATURES = FeatureSet(
    _TS_FE_AVAIL,
    _TORCHTRT_RT_AVAIL,
    _DYNAMO_FE_AVAIL,
    _FX_FE_AVAIL
)

def _enabled_features_str() -> str:
    enabled = lambda x: "ENABLED" if x else "DISABLED"
    return f"Enabled Features:\n  - Dynamo Frontend: {enabled(_DYNAMO_FE_AVAIL)}\n  - Torch-TensorRT Runtime: {enabled(_TORCHTRT_RT_AVAIL)}\n  - FX Frontend: {enabled(_FX_FE_AVAIL)}\n  - TorchScript Frontend: {enabled(_TS_FE_AVAIL)}\n"