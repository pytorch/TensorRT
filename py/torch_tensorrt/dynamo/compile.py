from __future__ import annotations

import collections.abc
import logging
from typing import Any, List, Optional, Set, Tuple

import torch
import torch_tensorrt
from torch.fx.passes.pass_manager import PassManager
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import (  # TODO: Should probabably be the TRT EngineCapability Enum
    EngineCapability,
)
from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo._defaults import (
    DEBUG,
    ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    MAX_AUX_STREAMS,
    MIN_BLOCK_SIZE,
    OPTIMIZATION_LEVEL,
    PASS_THROUGH_BUILD_FAILURES,
    PRECISION,
    TRUNCATE_LONG_AND_DOUBLE,
    USE_FAST_PARTITIONER,
    USE_PYTHON_RUNTIME,
    VERSION_COMPATIBLE,
    WORKSPACE_SIZE,
)
from torch_tensorrt.dynamo.backend.backends import _compile_module
from torch_tensorrt.dynamo.lowering._fusers import (
    fuse_permute_linear,
    fuse_permute_matmul,
)
from torch_tensorrt.dynamo.utils import prepare_device, prepare_inputs

logger = logging.getLogger(__name__)


def compile(
    gm: Any,
    inputs: Any,
    *,
    device: Device = Device._current_device(),
    disable_tf32: bool = False,
    sparse_weights: bool = False,
    enabled_precisions: Set[torch.dtype] | Tuple[torch.dtype] = (torch.float32,),
    refit: bool = False,
    debug: bool = DEBUG,
    capability: EngineCapability = EngineCapability.default,
    num_avg_timing_iters: int = 1,
    workspace_size: int = WORKSPACE_SIZE,
    dla_sram_size: int = 1048576,
    dla_local_dram_size: int = 1073741824,
    dla_global_dram_size: int = 536870912,
    calibrator: object = None,
    truncate_long_and_double: bool = TRUNCATE_LONG_AND_DOUBLE,
    require_full_compilation: bool = False,
    min_block_size: int = MIN_BLOCK_SIZE,
    torch_executed_ops: Optional[List[str]] = None,
    torch_executed_modules: Optional[List[str]] = None,
    pass_through_build_failures: bool = PASS_THROUGH_BUILD_FAILURES,
    max_aux_streams: Optional[int] = MAX_AUX_STREAMS,
    version_compatible: bool = VERSION_COMPATIBLE,
    optimization_level: Optional[int] = OPTIMIZATION_LEVEL,
    use_python_runtime: bool = USE_PYTHON_RUNTIME,
    use_fast_partitioner: bool = USE_FAST_PARTITIONER,
    enable_experimental_decompositions: bool = ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    **kwargs: Any,
) -> torch.fx.GraphModule:
    if debug:
        logger.setLevel(logging.DEBUG)

    enabled_precisions = set(enabled_precisions)

    logger.warning(
        "The Dynamo backend is an experimental feature, for which only the "
        "following arguments are supported: "
        "{enabled_precisions, debug, workspace_size, min_block_size, "
        "torch_executed_ops, pass_through_build_failures, use_fast_partitioner, "
        "enable_experimental_decompositions}"
    )

    if not isinstance(inputs, collections.abc.Sequence):
        inputs = [inputs]

    _, torch_inputs = prepare_inputs(inputs, prepare_device(device))

    if (
        torch.float16 in enabled_precisions
        or torch_tensorrt.dtype.half in enabled_precisions
    ):
        precision = torch.float16
    elif (
        torch.float32 in enabled_precisions
        or torch_tensorrt.dtype.float in enabled_precisions
    ):
        precision = torch.float32
    elif len(enabled_precisions) == 0:
        logger.info(f"No precision specified, defaulting to {PRECISION}")
        precision = PRECISION
    else:
        raise ValueError(
            f"Precision {enabled_precisions} not supported in the Dynamo Path"
        )

    compilation_options = {
        "precision": precision,
        "debug": debug,
        "workspace_size": workspace_size,
        "min_block_size": min_block_size,
        "torch_executed_ops": torch_executed_ops
        if torch_executed_ops is not None
        else [],
        "pass_through_build_failures": pass_through_build_failures,
        "max_aux_streams": max_aux_streams,
        "version_compatible": version_compatible,
        "optimization_level": optimization_level,
        "use_python_runtime": use_python_runtime,
        "truncate_long_and_double": truncate_long_and_double,
        "use_fast_partitioner": use_fast_partitioner,
        "enable_experimental_decompositions": enable_experimental_decompositions,
    }

    settings = CompilationSettings(**compilation_options)

    return _compile_module(gm, torch_inputs, settings)


def lower_model(
    model: torch.nn.Module, inputs: Any, **kwargs: Any
) -> torch.fx.GraphModule:
    graph_optimization_pm = PassManager.build_from_passlist(
        [fuse_permute_matmul, fuse_permute_linear]
    )
    lowered_model: torch.fx.GraphModule = graph_optimization_pm(model)
    # if isinstance(lowered_model, torch.fx.GraphModule):
    #     ShapeProp(lowered_model).propagate(*inputs)

    return lowered_model
