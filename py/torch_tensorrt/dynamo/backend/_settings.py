from dataclasses import dataclass, field
from typing import Sequence

from torch_tensorrt.fx.utils import LowerPrecision
from torch_tensorrt.dynamo.backend._defaults import (
    PRECISION,
    DEBUG,
    MAX_WORKSPACE_SIZE,
    MIN_BLOCK_SIZE,
    PASS_THROUGH_BUILD_FAILURES,
    TRUNCATE_LONG_AND_DOUBLE,
)


@dataclass(frozen=True)
class CompilationSettings:
    precision: LowerPrecision = PRECISION
    debug: bool = DEBUG
    workspace_size: int = MAX_WORKSPACE_SIZE
    min_block_size: int = MIN_BLOCK_SIZE
    torch_executed_ops: Sequence[str] = field(default_factory=set)
    pass_through_build_failures: bool = PASS_THROUGH_BUILD_FAILURES
    truncate_long_and_double: bool = TRUNCATE_LONG_AND_DOUBLE
