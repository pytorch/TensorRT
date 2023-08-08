from torch_tensorrt._utils import sanitized_torch_version

from packaging import version

if version.parse(sanitized_torch_version()) >= version.parse("2.1.dev"):
    from ._settings import *  # noqa: F403
    from ._SourceIR import SourceIR  # noqa: F403
    from .aten_tracer import trace  # noqa: F403
    from .compile import compile  # noqa: F403
    from .conversion import *  # noqa: F403
    from .conversion.converter_registry import DYNAMO_CONVERTERS  # noqa: F403
    from .conversion.converter_registry import dynamo_tensorrt_converter  # noqa: F403
