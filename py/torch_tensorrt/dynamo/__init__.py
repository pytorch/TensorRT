from torch_tensorrt._utils import sanitized_torch_version

from packaging import version

if version.parse(sanitized_torch_version()) >= version.parse("2.1.dev"):
    from ._settings import *
    from ._SourceIR import SourceIR
    from .aten_tracer import trace
    from .compile import compile
    from .conversion import *
    from .conversion.converter_registry import (
        DYNAMO_CONVERTERS,
        dynamo_tensorrt_converter,
    )
