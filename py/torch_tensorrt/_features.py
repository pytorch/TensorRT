import os
import sys
from collections import namedtuple

from torch_tensorrt._utils import sanitized_torch_version

from packaging import version

FeatureSet = namedtuple(
    "FeatureSet",
    [
        "torchscript_frontend",
        "torch_tensorrt_runtime",
        "dynamo_frontend",
        "fx_frontend",
    ],
)

trtorch_dir = os.path.dirname(__file__)
linked_file = os.path.join(
    "lib", "torchtrt.dll" if sys.platform.startswith("win") else "libtorchtrt.so"
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

_TS_FE_AVAIL = os.path.isfile(linked_file_full_path)
_TORCHTRT_RT_AVAIL = _TS_FE_AVAIL or os.path.isfile(linked_file_runtime_full_path)
_DYNAMO_FE_AVAIL = version.parse(sanitized_torch_version()) >= version.parse("2.1.dev")
_FX_FE_AVAIL = True

ENABLED_FEATURES = FeatureSet(
    _TS_FE_AVAIL, _TORCHTRT_RT_AVAIL, _DYNAMO_FE_AVAIL, _FX_FE_AVAIL
)


def _enabled_features_str() -> str:
    enabled = lambda x: "ENABLED" if x else "DISABLED"
    out_str: str = (
        f"Enabled Features:\n  - Dynamo Frontend: {enabled(_DYNAMO_FE_AVAIL)}\n  - Torch-TensorRT Runtime: {enabled(_TORCHTRT_RT_AVAIL)}\n  - FX Frontend: {enabled(_FX_FE_AVAIL)}\n  - TorchScript Frontend: {enabled(_TS_FE_AVAIL)}\n"  # type: ignore[no-untyped-call]
    )
    return out_str
