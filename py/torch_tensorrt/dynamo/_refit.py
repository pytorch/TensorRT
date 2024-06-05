from __future__ import annotations

import collections.abc
import logging
import warnings
from typing import Any, Collection, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
from torch.export import ExportedProgram
from torch.fx.node import Target
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import EngineCapability, dtype
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo import _defaults, partitioning
from torch_tensorrt.dynamo.conversion import CompilationSettings
from torch_tensorrt.dynamo.conversion._conversion import infer_module_output_dtypes
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.conversion._TRTInterpreter import TRTInterpreter
from torch_tensorrt.dynamo.conversion.truncate_double import repair_double_inputs
from torch_tensorrt.dynamo.lowering import apply_lowering_passes, get_decompositions
from torch_tensorrt.dynamo.utils import (
    get_torch_inputs,
    prepare_inputs,
    set_log_level,
    to_torch_device,
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
    MODULE_MAP = {
        "SCALE": (trt.IScaleLayer, [("scale", "SCALE"), ("shift", "SHIFT")]),
        "CONVOLUTION": (
            trt.IConvolutionLayer,
            [("kernel", "KERNEL"), ("bias", "BIAS")],
        ),
        "CONSTANT": (trt.IConstantLayer, [("weights", "CONSTANT")]),
    }

    output_dtypes = infer_module_output_dtypes(
        module,
        inputs,
        settings.device,
        truncate_double=settings.truncate_double,
    )

    # Use Interpreter
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
        if layer_type in MODULE_MAP:
            layer.__class__ = MODULE_MAP[layer_type][0]
            for weight_type, weight_name in MODULE_MAP[layer_type][1]:
                weight_map[f"{layer.name} {weight_name}"] = layer.__getattribute__(
                    weight_type
                ).copy()

        else:
            warnings.warn(f"{layer_type} is not supported yet")

    return weight_map


def refit_single_trt_engine_with_gm(
    new_gm: torch.fx.GraphModule,
    old_engine: trt.ICudaEngine,
    input_list: Tuple[Any, ...],
    settings: CompilationSettings = CompilationSettings(),
) -> None:

    # Get the refitting mapping
    mapping = construct_refit_mapping(new_gm, input_list, settings)

    trt_wt_location = trt.TensorLocation.HOST
    refitter = trt.Refitter(old_engine, TRT_LOGGER)
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


# def refit_module_weights(
#     compiled_module: ExportedProgram,
#     new_weight_module: ExportedProgram
# ) -> torch.fx.GraphModule:
#     pass


def refit_module_weights(
    compiled_module: torch.fx.GraphModule | ExportedProgram,
    new_weight_module: ExportedProgram,
    inputs: Tuple[Any, ...],
    *,
    device: Optional[Union[Device, torch.device, str]] = _defaults.DEVICE,
    disable_tf32: bool = _defaults.DISABLE_TF32,
    sparse_weights: bool = _defaults.SPARSE_WEIGHTS,
    enabled_precisions: (
        Set[torch.dtype | dtype] | Tuple[torch.dtype | dtype]
    ) = _defaults.ENABLED_PRECISIONS,
    engine_capability: EngineCapability = _defaults.ENGINE_CAPABILITY,
    refit: bool = _defaults.REFIT,
    debug: bool = _defaults.DEBUG,
    num_avg_timing_iters: int = _defaults.NUM_AVG_TIMING_ITERS,
    workspace_size: int = _defaults.WORKSPACE_SIZE,
    dla_sram_size: int = _defaults.DLA_SRAM_SIZE,
    dla_local_dram_size: int = _defaults.DLA_LOCAL_DRAM_SIZE,
    dla_global_dram_size: int = _defaults.DLA_GLOBAL_DRAM_SIZE,
    truncate_double: bool = _defaults.TRUNCATE_DOUBLE,
    require_full_compilation: bool = _defaults.REQUIRE_FULL_COMPILATION,
    min_block_size: int = _defaults.MIN_BLOCK_SIZE,
    torch_executed_ops: Optional[Collection[Target]] = None,
    torch_executed_modules: Optional[List[str]] = None,
    pass_through_build_failures: bool = _defaults.PASS_THROUGH_BUILD_FAILURES,
    max_aux_streams: Optional[int] = _defaults.MAX_AUX_STREAMS,
    version_compatible: bool = _defaults.VERSION_COMPATIBLE,
    optimization_level: Optional[int] = _defaults.OPTIMIZATION_LEVEL,
    use_python_runtime: bool = _defaults.USE_PYTHON_RUNTIME,
    use_fast_partitioner: bool = _defaults.USE_FAST_PARTITIONER,
    enable_experimental_decompositions: bool = _defaults.ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    dryrun: bool = _defaults.DRYRUN,
    hardware_compatible: bool = _defaults.HARDWARE_COMPATIBLE,
    **kwargs: Any,
) -> torch.fx.GraphModule:
    """
    Refit a compiled graph module with ExportedProgram
    """

    if debug:
        set_log_level(logger.parent, logging.DEBUG)

    if type(compiled_module) == ExportedProgram:
        compiled_module = compiled_module.module()

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

    engine_capability = EngineCapability._from(engine_capability)

    if torch_executed_modules is not None and torch_executed_modules:
        logger.warning(
            f"Detected torch_executed_modules was non-empty: {torch_executed_modules}"
            "\nThis feature is unimplemented in Torch-TRT Dynamo currently."
        )

    if not isinstance(inputs, collections.abc.Sequence):
        inputs = [inputs]

    # Prepare torch_trt inputs
    inputs = prepare_inputs(inputs)
    device = to_torch_tensorrt_device(device)
    enabled_precisions = {dtype._from(p) for p in enabled_precisions}

    if not isinstance(new_weight_module, ExportedProgram):
        raise AssertionError(
            f"Input graph should be an ExportedProgram but got type {type(new_weight_module)}"
        )
    new_weight_module = new_weight_module.run_decompositions(
        get_decompositions(enable_experimental_decompositions)
    )
    gm = new_weight_module.module()
    logger.debug("Input graph: " + str(gm.graph))
    # Apply lowering on the graph module
    torch_inputs = get_torch_inputs(inputs, device)
    gm = apply_lowering_passes(gm, torch_inputs)

    logger.debug("Lowered Input graph: " + str(gm.graph))

    compilation_options = {
        "enabled_precisions": (
            enabled_precisions if enabled_precisions else _defaults.ENABLED_PRECISIONS
        ),
        "debug": debug,
        "device": device,
        "workspace_size": workspace_size,
        "min_block_size": min_block_size,
        "torch_executed_ops": (
            torch_executed_ops if torch_executed_ops is not None else set()
        ),
        "pass_through_build_failures": pass_through_build_failures,
        "max_aux_streams": max_aux_streams,
        "version_compatible": version_compatible,
        "optimization_level": optimization_level,
        "use_python_runtime": use_python_runtime,
        "truncate_double": truncate_double,
        "use_fast_partitioner": use_fast_partitioner,
        "num_avg_timing_iters": num_avg_timing_iters,
        "enable_experimental_decompositions": enable_experimental_decompositions,
        "require_full_compilation": require_full_compilation,
        "disable_tf32": disable_tf32,
        "sparse_weights": sparse_weights,
        "refit": refit,
        "engine_capability": engine_capability,
        "dla_sram_size": dla_sram_size,
        "dla_local_dram_size": dla_local_dram_size,
        "dla_global_dram_size": dla_global_dram_size,
        "dryrun": dryrun,
        "hardware_compatible": hardware_compatible,
    }

    settings = CompilationSettings(**compilation_options)
    logger.info("Compilation Settings: %s\n", settings)

    # Set torch-executed ops
    CONVERTERS.set_disallowed_targets(settings.torch_executed_ops)

    # If specified, try using the fast partitioner and fall back to the global one on failure
    if settings.use_fast_partitioner:
        try:
            partitioned_module, supported_ops = partitioning.fast_partition(
                gm,
                verbose=settings.debug,
                min_block_size=settings.min_block_size,
                torch_executed_ops=settings.torch_executed_ops,
            )
        except torch.fx.passes.splitter_base.FxNetSplitterInternalError:
            logger.error(
                "Partitioning failed on the subgraph with fast partition. See trace above. "
                + "Retrying with global partition.",
                exc_info=True,
            )

            fast_partitioner_failed = True
            settings.use_fast_partitioner = False

    if not settings.use_fast_partitioner:
        partitioned_module, supported_ops = partitioning.global_partition(
            gm,
            verbose=settings.debug,
            min_block_size=settings.min_block_size,
            torch_executed_ops=settings.torch_executed_ops,
        )

    # Iterate over all components that can be accelerated
    # Generate the corresponding TRT Module for those
    for name, _ in partitioned_module.named_children():
        new_submodule = getattr(partitioned_module, name)
        compiled_submodule = getattr(compiled_module, name)
        engine = compiled_submodule.engine

        # Get the submodule inputs for min, opt, max shapes of the graph inputs
        submodule_inputs = partitioning.construct_submodule_inputs(new_submodule)

        logger.debug(
            "Refitting Submodule name: %s\n",
            str(name),
        )

        assert submodule_inputs is not None
        # Handle long/double inputs if requested by the user
        if settings.truncate_double:
            submodule_inputs = repair_double_inputs(
                partitioned_module,
                new_submodule,
                submodule_inputs,
                to_torch_device(settings.device),
                name,
            )

        # Refit TRT engines from submodule in place
        # TODO: Change it to return a new object
        refit_single_trt_engine_with_gm(
            new_gm=new_submodule,
            old_engine=engine,
            input_list=submodule_inputs,
            settings=settings,
        )
