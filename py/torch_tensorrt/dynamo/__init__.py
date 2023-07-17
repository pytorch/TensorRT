from packaging import version
from torch_tensorrt._utils import sanitized_torch_version

if version.parse(sanitized_torch_version()) >= version.parse("2.1.dev"):
    from torch_tensorrt.dynamo import fx_ts_compat
    from .backend import compile
