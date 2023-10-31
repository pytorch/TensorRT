import logging

from torch_tensorrt._utils import sanitized_torch_version

from packaging import version

logger = logging.getLogger(__name__)

if version.parse(sanitized_torch_version()) >= version.parse("2.1.dev"):
    from ._compiler import compile
    from ._exporter import export
    from ._settings import CompilationSettings
    from ._SourceIR import SourceIR
    from ._tracer import trace
