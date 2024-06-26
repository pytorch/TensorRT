from __future__ import annotations

import collections.abc
import copy
import logging
from typing import Any, Sequence, Tuple

import numpy as np
import tensorrt as trt
import torch
from torch.export import ExportedProgram
from torch_tensorrt._enums import dtype
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo._exporter import inline_torch_modules
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
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
    SERIALIZED_METADATA_IDX,
    TorchTensorRTModule,
)
from torch_tensorrt.dynamo.utils import (
    check_output,
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
    """Find out the weight mapping between weight in exported program and TensorRT engine
    Args:
        module: FX GraphModule to interpret
        inputs: Sequence of Tensors representing inputs to the module
        settings: Compilation settings
    Returns:
        Mapping from weight name in TensorRT to actual weight value in np.ndarray
    """
    MODULE_MAP = {
        "SCALE": (trt.IScaleLayer, [("scale", "SCALE"), ("shift", "SHIFT")]),
        "CONVOLUTION": (
            trt.IConvolutionLayer,
            [("kernel", "KERNEL"), ("bias", "BIAS")],
        ),
        "DECONVOLUTION": (
            trt.IDeconvolutionLayer,
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
            # Cast the parent class to child class to access attributes
            # For example: ILayer does not have ILayer.kernal/ILayer.bias
            # So we cast it to IConvolutionLayer and access the attributes
            layer.__class__ = MODULE_MAP[layer_type][0]
            for weight_type, weight_name in MODULE_MAP[layer_type][1]:
                weight = layer.__getattribute__(weight_type).copy()
                weight_dtype = dtype.try_from(weight.dtype).to(trt.DataType)
                weight_map[f"{layer.name} {weight_name}"] = (
                    weight,
                    weight_dtype,
                )

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
    refitted = set()

    trt_wt_location = trt.TensorLocation.HOST
    refitter = trt.Refitter(old_engine, TRT_LOGGER)
    weight_list = refitter.get_all_weights()

    for layer_name in weight_list:
        if layer_name not in mapping:
            raise AssertionError(f"{layer_name} is not found in weight mapping")
        # Use Numpy to create weights
        weight, datatype = mapping[layer_name]
        trt_wt_tensor = trt.Weights(datatype, weight.ctypes.data, weight.size)
        refitter.set_named_weights(layer_name, trt_wt_tensor, trt_wt_location)
        refitted.add(layer_name)

    if len(refitted) != len(weight_list):
        logger.warning("Not all weights have been refitted!!!")

    if not refitter.refit_cuda_engine():
        logger.error("Error: failed to refit new weights.")
        exit(0)


def refit_module_weights(
    compiled_module: torch.fx.GraphModule | ExportedProgram,
    new_weight_module: ExportedProgram,
    inputs: Tuple[Any, ...],
    verify_output: bool = False,
) -> torch.fx.GraphModule:
    """
    Refit a compiled graph module with ExportedProgram. This performs weight updates in compiled_module without recompiling the engine.

    Args:
        compiled_module: compiled TensorRT module that needs to be refitted.
                        This compiled_module should be compmiled by torch_tensorrt.dynamo.compile
                        or load it from disk using trt.load.
        new_weight_module: exported program with the updated weights. This one should have the same model architecture as the compiled module.
        inputs: sample inputs
        verify_output: whether to verify output of refitted module
    Returns:
        A new compiled TensorRT module that has the updated weights.
    """
    inline_module = False
    if isinstance(compiled_module, ExportedProgram):
        inline_module = True
        compiled_module = compiled_module.module()

    compiled_module = copy.deepcopy(compiled_module)

    # Get the settings and check the setting to be uniform
    settings: CompilationSettings = None
    if inline_module:

        # Obtain the settings
        compiled_submodules = [
            (name.replace("_engine", ""), engine)
            for name, engine in compiled_module.__dict__.items()
            if "engine" in name
        ]
        encoded_settings = compiled_submodules[0][1].__getstate__()[0][
            SERIALIZED_METADATA_IDX
        ]
        assert (
            encoded_settings != ""
        ), "Settings are not saved in the engine. Please recompile the engine with refit=True."
        settings = TorchTensorRTModule.decode_metadata(encoded_settings)
        # Handle torch modules
        compiled_submodules_map = dict(compiled_submodules)
        for name, submodule in compiled_module.named_children():
            compiled_submodules_map[name] = submodule

    else:
        for name, submodule in compiled_module.named_children():
            if not isinstance(
                submodule, (PythonTorchTensorRTModule, TorchTensorRTModule)
            ):
                continue
            settings = submodule.settings

    assert (
        settings.refit
    ), "Refitting is not enabled. Please recompile the engine with refit=True."

    if settings.debug:
        set_log_level(logger.parent, logging.DEBUG)

    if not isinstance(inputs, collections.abc.Sequence):
        inputs = [inputs]

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
    new_gm = new_weight_module.module()
    logger.debug("Input graph: " + str(new_gm.graph))
    # Apply lowering on the graph module
    torch_inputs = get_torch_inputs(inputs, device)
    new_gm = apply_lowering_passes(new_gm, torch_inputs)

    logger.info("Compilation Settings: %s\n", settings)

    # Set torch-executed ops
    CONVERTERS.set_disallowed_targets(settings.torch_executed_ops)

    # If specified, try using the fast partitioner and fall back to the global one on failure
    if settings.use_fast_partitioner:
        try:
            new_partitioned_module, supported_ops = partitioning.fast_partition(
                new_gm,
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
        new_partitioned_module, supported_ops = partitioning.global_partition(
            new_gm,
            verbose=settings.debug,
            min_block_size=settings.min_block_size,
            torch_executed_ops=settings.torch_executed_ops,
        )

    if inline_module:
        # Preprocess the partitioned module to be in the same format as the inline module
        inline_torch_modules(new_partitioned_module)
        new_partitioned_module.delete_all_unused_submodules()
        # Check the number of partitions and name
        assert {sm[0] for sm in new_partitioned_module.named_children()} == set(
            compiled_submodules_map.keys()
        ), "New weights module is not compatible with previously compiled Torch-TensorRT module"
    else:
        assert {sm[0] for sm in new_partitioned_module.named_children()} == {
            sm[0] for sm in compiled_module.named_children()
        }, "New weights module is not compatible with previously compiled Torch-TensorRT module"
    # 2. TODO: Check the hash of source fx.Graph and new fx.Graph

    # Iterate over all components that can be accelerated
    # Generate the corresponding TRT Module for those

    for name, new_submodule in new_partitioned_module.named_children():

        # Refit each submodule
        # Extract engine from the submodule
        try:
            if inline_module:
                compiled_submodule = compiled_submodules_map[name]
                # If this is a torch module, load the old state_dict
                if "_run_on_acc" not in name:
                    compiled_submodule.load_state_dict(new_submodule.state_dict())
                    continue
                else:
                    engine_info = compiled_submodule.__getstate__()[0]
                    engine = get_engine_from_encoded_engine(engine_info[3], runtime)
            else:
                compiled_submodule = getattr(compiled_module, name)
                if isinstance(compiled_submodule, PythonTorchTensorRTModule):
                    engine = compiled_submodule.engine
                elif isinstance(compiled_submodule, TorchTensorRTModule):
                    engine_info = compiled_submodule.engine.__getstate__()[0]
                    engine = get_engine_from_encoded_engine(engine_info[3], runtime)
                elif isinstance(compiled_submodule, torch.fx.graph_module.GraphModule):
                    # This is graph break resulted by unsupported ops
                    compiled_submodule.load_state_dict(new_submodule.state_dict())
                    continue
                else:
                    raise AssertionError(
                        "The type of graph module is not supported for refitting."
                    )
        except AttributeError:
            raise AssertionError(
                "The type of graph module is not supported for refitting or two compiled modules do not match."
            )

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
                new_partitioned_module,
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

        if isinstance(compiled_submodule, TorchTensorRTModule):
            serialized_engine = bytes(engine.serialize())
            new_engine_info = list(engine_info)
            new_engine_info[3] = serialized_engine
            refitted_engine = torch.classes.tensorrt.Engine(tuple(new_engine_info))
            compiled_submodule.engine = refitted_engine

        elif inline_module:
            serialized_engine = bytes(engine.serialize())
            new_engine_info = list(engine_info)
            new_engine_info[3] = serialized_engine
            refitted_engine = torch.classes.tensorrt.Engine(tuple(new_engine_info))
            setattr(compiled_module, f"{name}_engine", refitted_engine)

    if verify_output:
        if check_output(
            new_module=new_gm,
            refitted_module=compiled_module,
            inputs=get_torch_inputs(inputs, device),
        ):
            logger.info("Refitting Succeed!")
        else:
            logger.error("Refitting Failed! The outputs do not match.")
    else:
        logger.info("Refitting Completed! Output verification skipped.")

    return compiled_module


# Util functions -----------
import base64


def get_engine_from_encoded_engine(
    encoded_engine: str, runtime: trt.Runtime
) -> trt.ICudaEngine:
    serialized_engine = base64.b64decode(encoded_engine)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine
