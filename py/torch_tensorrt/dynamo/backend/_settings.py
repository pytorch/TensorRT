from dataclasses import dataclass

from torch_tensorrt.fx.utils import LowerPrecision
from torch_tensorrt.dynamo.backend._defaults import (
    PRECISION,
    DEBUG,
    MAX_WORKSPACE_SIZE,
    MAX_NUM_TRT_ENGINES,
)


@dataclass(frozen=True)
class CompilationSettings:
    precision: LowerPrecision = PRECISION
    debug: bool = DEBUG
    workspace_size: int = MAX_WORKSPACE_SIZE
    max_num_trt_engines: int = MAX_NUM_TRT_ENGINES
