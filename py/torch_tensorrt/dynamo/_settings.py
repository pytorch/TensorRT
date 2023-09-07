from dataclasses import dataclass, field
from typing import Optional, Set

import torch
from torch_tensorrt._Device import Device
from torch_tensorrt.dynamo._defaults import (
    DEBUG,
    ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    FALLBACK_TO_INDUCTOR,
    MAX_AUX_STREAMS,
    MIN_BLOCK_SIZE,
    OPTIMIZATION_LEVEL,
    PASS_THROUGH_BUILD_FAILURES,
    PRECISION,
    REQUIRE_FULL_COMPILATION,
    TRUNCATE_LONG_AND_DOUBLE,
    USE_FAST_PARTITIONER,
    USE_PYTHON_RUNTIME,
    VERSION_COMPATIBLE,
    WORKSPACE_SIZE,
    default_device,
)


@dataclass
class CompilationSettings:
    """Compilation settings for Torch-TensorRT Dynamo Paths

    Args:
        precision (torch.dtype): Model Layer precision
        debug (bool): Whether to print out verbose debugging information
        workspace_size (int): Workspace TRT is allowed to use for the module (0 is default)
        min_block_size (int): Minimum number of operators per TRT-Engine Block
        torch_executed_ops (Sequence[str]): Sequence of operations to run in Torch, regardless of converter coverage
        pass_through_build_failures (bool): Whether to fail on TRT engine build errors (True) or not (False)
        max_aux_streams (Optional[int]): Maximum number of allowed auxiliary TRT streams for each engine
        version_compatible (bool): Provide version forward-compatibility for engine plan files
        optimization_level (Optional[int]): Builder optimization 0-5, higher levels imply longer build time,
            searching for more optimization options. TRT defaults to 3
        use_python_runtime (Optional[bool]): Whether to strictly use Python runtime or C++ runtime. To auto-select a runtime
            based on C++ dependency presence (preferentially choosing C++ runtime if available), leave the
            argument as None
        truncate_long_and_double (bool): Truncate int64/float64 TRT engine inputs or weights to int32/float32
        enable_experimental_decompositions (bool): Whether to enable all core aten decompositions
            or only a selected subset of them
        fallback_to_inductor (bool): Whether to fallback to inductor on Torch-TRT Compilation Errors.
            Is overridden by pass_through_build_failures.
    """

    precision: torch.dtype = PRECISION
    debug: bool = DEBUG
    workspace_size: int = WORKSPACE_SIZE
    min_block_size: int = MIN_BLOCK_SIZE
    torch_executed_ops: Set[str] = field(default_factory=set)
    pass_through_build_failures: bool = PASS_THROUGH_BUILD_FAILURES
    max_aux_streams: Optional[int] = MAX_AUX_STREAMS
    version_compatible: bool = VERSION_COMPATIBLE
    optimization_level: Optional[int] = OPTIMIZATION_LEVEL
    use_python_runtime: Optional[bool] = USE_PYTHON_RUNTIME
    truncate_long_and_double: bool = TRUNCATE_LONG_AND_DOUBLE
    use_fast_partitioner: bool = USE_FAST_PARTITIONER
    enable_experimental_decompositions: bool = ENABLE_EXPERIMENTAL_DECOMPOSITIONS
    device: Device = field(default_factory=default_device)
    require_full_compilation: bool = REQUIRE_FULL_COMPILATION
    fallback_to_inductor: bool = FALLBACK_TO_INDUCTOR
