from __future__ import annotations

import logging
import warnings
from typing import Any, Optional, Sequence, Set, Tuple

import numpy as np
import tensorrt as trt
import torch
from torch.export import ExportedProgram
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import EngineCapability, dtype
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo import _defaults
from torch_tensorrt.dynamo.conversion import CompilationSettings
from torch_tensorrt.dynamo.conversion._conversion import infer_module_output_dtypes
from torch_tensorrt.dynamo.conversion._TRTInterpreter import TRTInterpreter
from torch_tensorrt.dynamo.lowering import apply_lowering_passes, get_decompositions
from torch_tensorrt.dynamo.utils import (
    get_torch_inputs,
    prepare_inputs,
    to_torch_tensorrt_device,
)
from torch_tensorrt.logging import TRT_LOGGER

logger = logging.getLogger(__name__)


def construct_refit_mapping(
    module: torch.fx.GraphModule,
    inputs: Sequence[Input],
    settings: CompilationSettings = CompilationSettings(),
) -> dict[str, np.ndarray]:
    """Interpret an FX module to a TRTInterpreterResult
    Args:
        module: FX GraphModule to interpret
        inputs: Sequence of Tensors representing inputs to the module
        settings: Compilation settings
    Returns:
        TRTInterpreterResult
    """
    output_dtypes = infer_module_output_dtypes(
        module,
        inputs,
        settings.device,
        truncate_double=settings.truncate_double,
    )

    # Use Interpreter
    module_map = {
        "SCALE": (trt.IScaleLayer, [("scale", "SCALE"), ("shift", "SHIFT")]),
        "CONVOLUTION": (
            trt.IConvolutionLayer,
            [("kernel", "KERNEL"), ("bias", "BIAS")],
        ),
        "CONSTANT": (trt.IConstantLayer, [("weights", "CONSTANT")]),
    }
    weight_map = {}
    interpreter = TRTInterpreter(
        module,
        inputs,
        logger_level=(trt.Logger.VERBOSE if settings.debug else trt.Logger.WARNING),
        output_dtypes=output_dtypes,
        compilation_settings=settings,
    )
    interpreter._construct_trt_network_def()
    net = interpreter.ctx.net
    for i in range(net.num_layers):
        layer = net[i]
        layer_type: str = layer.type.name
        if layer_type in module_map:
            layer.__class__ = module_map[layer_type][0]
            for weight_type, weight_name in module_map[layer_type][1]:
                weight_map[f"{layer.name} {weight_name}"] = layer.__getattribute__(
                    weight_type
                ).copy()

        else:
            warnings.warn(f"{layer_type} is not supported yet")

    return weight_map


def refit_single_trt_engine_with_ep(
    exported_program: ExportedProgram,
    inputs: Tuple[Any, ...],
    engine: object,
    *,
    enabled_precisions: (
        Set[torch.dtype | dtype] | Tuple[torch.dtype | dtype]
    ) = _defaults.ENABLED_PRECISIONS,
    debug: bool = _defaults.DEBUG,
    workspace_size: int = _defaults.WORKSPACE_SIZE,
    min_block_size: int = _defaults.MIN_BLOCK_SIZE,
    torch_executed_ops: Optional[Set[str]] = None,
    pass_through_build_failures: bool = _defaults.PASS_THROUGH_BUILD_FAILURES,
    max_aux_streams: Optional[int] = _defaults.MAX_AUX_STREAMS,
    version_compatible: bool = _defaults.VERSION_COMPATIBLE,
    optimization_level: Optional[int] = _defaults.OPTIMIZATION_LEVEL,
    use_python_runtime: Optional[bool] = _defaults.USE_PYTHON_RUNTIME,
    truncate_double: bool = _defaults.TRUNCATE_DOUBLE,
    use_fast_partitioner: bool = _defaults.USE_FAST_PARTITIONER,
    enable_experimental_decompositions: bool = _defaults.ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    device: Device = Device._current_device(),
    require_full_compilation: bool = _defaults.REQUIRE_FULL_COMPILATION,
    disable_tf32: bool = _defaults.DISABLE_TF32,
    sparse_weights: bool = _defaults.SPARSE_WEIGHTS,
    refit: bool = _defaults.REFIT,
    engine_capability: EngineCapability = _defaults.ENGINE_CAPABILITY,
    num_avg_timing_iters: int = _defaults.NUM_AVG_TIMING_ITERS,
    dla_sram_size: int = _defaults.DLA_SRAM_SIZE,
    dla_local_dram_size: int = _defaults.DLA_LOCAL_DRAM_SIZE,
    dla_global_dram_size: int = _defaults.DLA_GLOBAL_DRAM_SIZE,
    **kwargs: Any,
) -> None:

    if "truncate_long_and_double" in kwargs.keys():
        if truncate_double is not _defaults.TRUNCATE_DOUBLE:
            raise ValueError(
                'Provided configuration for "truncate_double" and deprecated API "truncate_long_and_double", please only use "truncate_double"'
            )
        else:
            truncate_double = kwargs["truncate_long_and_double"]
            warnings.warn(
                'Compiler option "truncate_long_and_double" is deprecated in favor of "truncate_double" as int64 is now natively supported, this option will be removed in the next version',
                DeprecationWarning,
                stacklevel=2,
            )

    input_list = list(inputs) if inputs is not None else []
    torch_executed_ops = torch_executed_ops if torch_executed_ops is not None else set()
    # Prepare torch_trt inputs
    input_list = prepare_inputs(input_list)
    device = to_torch_tensorrt_device(device)

    enabled_precisions = {dtype._from(e) for e in enabled_precisions}

    # Try to use the old setting if available
    compilation_options = {
        "enabled_precisions": enabled_precisions,
        "debug": debug,
        "workspace_size": workspace_size,
        "min_block_size": min_block_size,
        "torch_executed_ops": torch_executed_ops,
        "pass_through_build_failures": pass_through_build_failures,
        "max_aux_streams": max_aux_streams,
        "version_compatible": version_compatible,
        "optimization_level": optimization_level,
        "use_python_runtime": use_python_runtime,
        "truncate_double": truncate_double,
        "use_fast_partitioner": use_fast_partitioner,
        "enable_experimental_decompositions": enable_experimental_decompositions,
        "device": device,
        "require_full_compilation": require_full_compilation,
        "disable_tf32": disable_tf32,
        "sparse_weights": sparse_weights,
        "refit": refit,
        "engine_capability": engine_capability,
        "num_avg_timing_iters": num_avg_timing_iters,
        "dla_sram_size": dla_sram_size,
        "dla_local_dram_size": dla_local_dram_size,
        "dla_global_dram_size": dla_global_dram_size,
    }

    # Decompose the exported program
    exported_program = exported_program.run_decompositions(
        get_decompositions(enable_experimental_decompositions)
    )
    gm = exported_program.module()
    logger.debug("Input graph: " + str(gm.graph))

    # Apply lowering on the graph module
    torch_inputs = get_torch_inputs(input_list, device)
    gm = apply_lowering_passes(gm, torch_inputs)
    logger.debug("Lowered Input graph: " + str(gm.graph))

    settings = CompilationSettings(**compilation_options)
    logger.info("Compilation Settings: %s\n", settings)

    # Get the refitting mapping
    mapping = construct_refit_mapping(gm, input_list, settings)

    trt_wt_location = trt.TensorLocation.HOST
    refitter = trt.Refitter(engine, TRT_LOGGER)
    weight_list = refitter.get_all_weights()

    for layer_name in weight_list:
        if layer_name not in mapping:
            print(f"{layer_name} is not found in weight mapping")

        # Use Numpy to create weights
        weight = mapping[layer_name]
        trt_wt_tensor = trt.Weights(trt.DataType.FLOAT, weight.ctypes.data, weight.size)
        refitter.set_named_weights(layer_name, trt_wt_tensor, trt_wt_location)

    if not refitter.refit_cuda_engine():
        print("Error: failed to refit new weights.")
        exit(0)

    print("Refit Successful")
