from dataclasses import dataclass, field
from typing import Collection, Optional

import torch
from tensorrt import EngineCapability
from torch.fx.node import Target
from torch_tensorrt._Device import Device
from torch_tensorrt.dynamo._defaults import (
    DEBUG,
    DISABLE_TF32,
    DLA_GLOBAL_DRAM_SIZE,
    DLA_LOCAL_DRAM_SIZE,
    DLA_SRAM_SIZE,
    ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    ENGINE_CAPABILITY,
    MAX_AUX_STREAMS,
    MIN_BLOCK_SIZE,
    NUM_AVG_TIMING_ITERS,
    OPTIMIZATION_LEVEL,
    OUTPUT_FORMAT,
    PASS_THROUGH_BUILD_FAILURES,
    PRECISION,
    REFIT,
    REQUIRE_FULL_COMPILATION,
    SPARSE_WEIGHTS,
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
        torch_executed_ops (Collection[Target]): Collection of operations to run in Torch, regardless of converter coverage
        pass_through_build_failures (bool): Whether to fail on TRT engine build errors (True) or not (False)
        max_aux_streams (Optional[int]): Maximum number of allowed auxiliary TRT streams for each engine
        version_compatible (bool): Provide version forward-compatibility for engine plan files
        optimization_level (Optional[int]): Builder optimization 0-5, higher levels imply longer build time,
            searching for more optimization options. TRT defaults to 3
        use_python_runtime (Optional[bool]): Whether to strictly use Python runtime or C++ runtime. To auto-select a runtime
            based on C++ dependency presence (preferentially choosing C++ runtime if available), leave the
            argument as None
        truncate_long_and_double (bool): Whether to truncate int64/float64 TRT engine inputs or weights to int32/float32
        use_fast_partitioner (bool): Whether to use the fast or global graph partitioning system
        enable_experimental_decompositions (bool): Whether to enable all core aten decompositions
            or only a selected subset of them
        device (Device): GPU to compile the model on
        require_full_compilation (bool): Whether to require the graph is fully compiled in TensorRT.
            Only applicable for `ir="dynamo"`; has no effect for `torch.compile` path
        disable_tf32 (bool): Whether to disable TF32 computation for TRT layers
        sparse_weights (bool): Whether to allow the builder to use sparse weights
        refit (bool): Whether to build a refittable engine
        engine_capability (trt.EngineCapability): Restrict kernel selection to safe gpu kernels or safe dla kernels
        num_avg_timing_iters (int): Number of averaging timing iterations used to select kernels
        dla_sram_size (int): Fast software managed RAM used by DLA to communicate within a layer.
        dla_local_dram_size (int): Host RAM used by DLA to share intermediate tensor data across operations
        dla_global_dram_size (int): Host RAM used by DLA to store weights and metadata for execution
        output_format (str): Output format of the result of TRT compilation. Options include "exported_program" (or) "ep" | "torchscript" (or) "ts" | "graph_module" (or) "fx". Default is "exported_program"
    """

    precision: torch.dtype = PRECISION
    debug: bool = DEBUG
    workspace_size: int = WORKSPACE_SIZE
    min_block_size: int = MIN_BLOCK_SIZE
    torch_executed_ops: Collection[Target] = field(default_factory=set)
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
    disable_tf32: bool = DISABLE_TF32
    sparse_weights: bool = SPARSE_WEIGHTS
    refit: bool = REFIT
    engine_capability: EngineCapability = ENGINE_CAPABILITY
    num_avg_timing_iters: int = NUM_AVG_TIMING_ITERS
    dla_sram_size: int = DLA_SRAM_SIZE
    dla_local_dram_size: int = DLA_LOCAL_DRAM_SIZE
    dla_global_dram_size: int = DLA_GLOBAL_DRAM_SIZE
    output_format: str = OUTPUT_FORMAT
