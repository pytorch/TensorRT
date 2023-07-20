from dataclasses import dataclass, field
from typing import Optional, Sequence
import torch
from torch_tensorrt.dynamo._defaults import (
    PRECISION,
    DEBUG,
    WORKSPACE_SIZE,
    MIN_BLOCK_SIZE,
    PASS_THROUGH_BUILD_FAILURES,
    MAX_AUX_STREAMS,
    VERSION_COMPATIBLE,
    OPTIMIZATION_LEVEL,
    USE_PYTHON_RUNTIME,
)


@dataclass
class CompilationSettings:
    precision: torch.dtype = PRECISION
    debug: bool = DEBUG
    workspace_size: int = WORKSPACE_SIZE
    min_block_size: int = MIN_BLOCK_SIZE
    torch_executed_ops: Sequence[str] = field(default_factory=set)
    pass_through_build_failures: bool = PASS_THROUGH_BUILD_FAILURES
    max_aux_streams: Optional[int] = MAX_AUX_STREAMS
    version_compatible: bool = VERSION_COMPATIBLE
    optimization_level: Optional[int] = OPTIMIZATION_LEVEL
    use_python_runtime: Optional[bool] = USE_PYTHON_RUNTIME
