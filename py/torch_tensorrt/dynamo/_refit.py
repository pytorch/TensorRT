from __future__ import annotations

import collections.abc
import copy
import logging
import pickle
import warnings
from typing import Any, Sequence, Tuple

import numpy as np
import tensorrt as trt
import torch
from torch.export import ExportedProgram
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo.conversion import CompilationSettings
from torch_tensorrt.dynamo.conversion._conversion import infer_module_output_dtypes
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.conversion._TRTInterpreter import TRTInterpreter
from torch_tensorrt.dynamo.conversion.truncate_double import repair_double_inputs
from torch_tensorrt.dynamo.lowering import apply_lowering_passes, get_decompositions
from torch_tensorrt.dynamo.runtime._PythonTorchTensorRTModule import (
    PythonTorchTensorRTModule,
)
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule
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


def _refit_single_trt_engine_with_gm(
    new_gm: torch.fx.GraphModule,
    old_engine: trt.ICudaEngine,
    input_list: Tuple[Any, ...],
    settings: CompilationSettings = CompilationSettings(),
) -> None:
    """
    Refit a TensorRT Engine in place
    """
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


def refit_module_weights(
    compiled_module_file_path: str, new_weight_module: ExportedProgram, inputs: Any
) -> torch.fx.GraphModule:
    """
    Return a copy of compiled_module with refitted weight
    """
    settings_wrapper = {"settings": None}

    compiled_exp_program = torch.export.load(
        compiled_module_file_path, extra_files=settings_wrapper
    )

    decoded_settings = base64.b64decode(settings_wrapper["settings"].encode("utf-8"))
    restored_settings = pickle.loads(decoded_settings)

    new_trt_gm = _refit_module_weights(
        compiled_module=compiled_exp_program,
        new_weight_module=new_weight_module,
        inputs=inputs,
        settings=restored_settings,
    )
    return new_trt_gm


def _refit_module_weights(
    compiled_module: torch.fx.GraphModule | ExportedProgram,
    new_weight_module: ExportedProgram,
    inputs: Tuple[Any, ...],
    settings: Any,
) -> torch.fx.GraphModule:
    """
    Refit a compiled graph module with ExportedProgram
    """

    if settings.debug:
        set_log_level(logger.parent, logging.DEBUG)

    if not isinstance(inputs, collections.abc.Sequence):
        inputs = [inputs]

    if isinstance(compiled_module, ExportedProgram):
        compiled_module = compiled_module.module()

    # Prepare torch_trt inputs
    inputs = prepare_inputs(inputs)
    device = to_torch_tensorrt_device(settings.device)
    runtime = trt.Runtime(TRT_LOGGER)
    if not isinstance(new_weight_module, ExportedProgram):
        raise AssertionError(
            f"Input graph should be an ExportedProgram but got type {type(new_weight_module)}"
        )
    new_weight_module = new_weight_module.run_decompositions(
        get_decompositions(settings.enable_experimental_decompositions)
    )
    gm = new_weight_module.module()
    logger.debug("Input graph: " + str(gm.graph))
    # Apply lowering on the graph module
    torch_inputs = get_torch_inputs(inputs, device)
    gm = apply_lowering_passes(gm, torch_inputs)

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
    # PytorchTensorRTModule does not support deepcopy
    # Create a shallow copy. Replace the TRTModule after
    compiled_module = copy.copy(compiled_module)
    # Iterate over all components that can be accelerated
    # Generate the corresponding TRT Module for those
    for name, _ in partitioned_module.named_children():
        new_submodule = getattr(partitioned_module, name)
        # TODO: Copy the submodule and return a new one
        inline_module = False
        try:
            compiled_submodule = getattr(compiled_module, name)
            if isinstance(compiled_submodule, PythonTorchTensorRTModule):
                engine = copy_cuda_engine(compiled_submodule.engine, runtime)
            elif isinstance(compiled_submodule, TorchTensorRTModule):
                engine_state = compiled_submodule.get_extra_state()
                encoded_engine = engine_state[1][0][3]
                engine = get_engine_from_encoded_engine(encoded_engine, runtime)
            else:
                raise AssertionError("The type of graph module is not supported.")
        except AttributeError:
            inline_module = True
            inline_engine = getattr(compiled_module, f"{name}_engine")
            engine_info = inline_engine.__getstate__()[0]
            engine = get_engine_from_encoded_engine(engine_info[3], runtime)

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

        _refit_single_trt_engine_with_gm(
            new_gm=new_submodule,
            old_engine=engine,
            input_list=submodule_inputs,
            settings=settings,
        )
        serialized_engine = bytes(engine.serialize())
        if inline_module:
            new_engine_info = list(engine_info)
            new_engine_info[3] = serialized_engine
            refitted_inline_engine = torch.classes.tensorrt.Engine(
                tuple(new_engine_info)
            )
            setattr(compiled_module, f"{name}_engine", refitted_inline_engine)
        else:
            # In TorchTensorRTModule, the original module is intact. Create a new module and assign to the fx.Graph
            if isinstance(compiled_submodule, TorchTensorRTModule):
                refitteded_submodule = create_new_TorchTensorRTModule(
                    compiled_submodule,
                    serialized_engine=serialized_engine,
                    settings=settings,
                )
            else:
                refitteded_submodule = create_new_PythonTorchTensorRTModule(
                    compiled_submodule,
                    serialized_engine=serialized_engine,
                    settings=settings,
                )
            setattr(compiled_module, name, refitteded_submodule)
        return compiled_module


# Util functions -----------
import base64


def get_engine_from_encoded_engine(
    encoded_engine: bytes, runtime: trt.Runtime
) -> trt.ICudaEngine:
    serialized_engine = base64.b64decode(encoded_engine)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine


def create_new_TorchTensorRTModule(
    module: TorchTensorRTModule, serialized_engine: trt.ICudaEngine, settings: object
) -> TorchTensorRTModule:
    return TorchTensorRTModule(
        serialized_engine=serialized_engine,
        name=module.name,
        input_binding_names=module.input_binding_names,
        output_binding_names=module.output_binding_names,
        target_device=settings.device,
        hardware_compatible=module.hardware_compatible,
    )


def create_new_PythonTorchTensorRTModule(
    module: PythonTorchTensorRTModule, serialized_engine: bytes, settings: object
) -> PythonTorchTensorRTModule:
    return PythonTorchTensorRTModule(
        engine=serialized_engine,
        input_names=module.input_names,
        output_names=module.output_names,
        target_device=settings.device,
        profiling_enabled=module.profiling_enabled,
    )


def copy_cuda_engine(engine: trt.ICudaEngine, runtime: trt.Runtime) -> trt.ICudaEngine:
    serialized_engine = engine.serialize()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine
