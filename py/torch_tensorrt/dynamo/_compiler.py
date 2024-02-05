from __future__ import annotations

import collections.abc
import logging
from typing import Any, Collection, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch_tensorrt
from torch.export import ExportedProgram
from torch.fx.node import Target
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import (  # TODO: Should probabably be the TRT EngineCapability Enum
    EngineCapability,
)
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo._defaults import (
    DEBUG,
    DEVICE,
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
)
from torch_tensorrt.dynamo._exporter import export
from torch_tensorrt.dynamo.conversion import (
    CompilationSettings,
    convert_module,
    repair_long_or_double_inputs,
)
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.lowering import apply_lowering_passes, get_decompositions
from torch_tensorrt.dynamo.utils import (
    get_torch_inputs,
    prepare_inputs,
    set_log_level,
    to_torch_device,
    to_torch_tensorrt_device,
)

logger = logging.getLogger(__name__)


def compile(
    exported_program: ExportedProgram,
    inputs: Tuple[Any, ...],
    *,
    device: Optional[Union[Device, torch.device, str]] = DEVICE,
    disable_tf32: bool = DISABLE_TF32,
    sparse_weights: bool = SPARSE_WEIGHTS,
    enabled_precisions: Set[torch.dtype] | Tuple[torch.dtype] = (torch.float32,),
    engine_capability: EngineCapability = ENGINE_CAPABILITY,
    refit: bool = REFIT,
    debug: bool = DEBUG,
    num_avg_timing_iters: int = NUM_AVG_TIMING_ITERS,
    workspace_size: int = WORKSPACE_SIZE,
    dla_sram_size: int = DLA_SRAM_SIZE,
    dla_local_dram_size: int = DLA_LOCAL_DRAM_SIZE,
    dla_global_dram_size: int = DLA_GLOBAL_DRAM_SIZE,
    truncate_long_and_double: bool = TRUNCATE_LONG_AND_DOUBLE,
    require_full_compilation: bool = REQUIRE_FULL_COMPILATION,
    min_block_size: int = MIN_BLOCK_SIZE,
    torch_executed_ops: Optional[Collection[Target]] = None,
    torch_executed_modules: Optional[List[str]] = None,
    pass_through_build_failures: bool = PASS_THROUGH_BUILD_FAILURES,
    max_aux_streams: Optional[int] = MAX_AUX_STREAMS,
    version_compatible: bool = VERSION_COMPATIBLE,
    optimization_level: Optional[int] = OPTIMIZATION_LEVEL,
    use_python_runtime: bool = USE_PYTHON_RUNTIME,
    use_fast_partitioner: bool = USE_FAST_PARTITIONER,
    enable_experimental_decompositions: bool = ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    output_format: str = OUTPUT_FORMAT,
    **kwargs: Any,
) -> torch.fx.GraphModule:
    """Compile a TorchScript module for NVIDIA GPUs using TensorRT

    Takes a existing TorchScript module and a set of settings to configure the compiler
    and will convert methods to JIT Graphs which call equivalent TensorRT engines

    Converts specifically the forward method of a TorchScript Module

    Arguments:
        exported_program (torch.export.ExportedProgram): Source module, running torch.export on a ``torch.nn.Module``
        inputs (Tuple[Any, ...]): List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or torch_tensorrt datatypes and you can use either torch devices or the torch_tensorrt device type enum
            to select device type. ::

                input=[
                    torch_tensorrt.Input((1, 3, 224, 224)), # Static NCHW input shape for input #1
                    torch_tensorrt.Input(
                        min_shape=(1, 224, 224, 3),
                        opt_shape=(1, 512, 512, 3),
                        max_shape=(1, 1024, 1024, 3),
                        dtype=torch.int32
                        format=torch.channel_last
                    ), # Dynamic input shape for input #2
                    torch.randn((1, 3, 224, 244)) # Use an example tensor and let torch_tensorrt infer settings
                ]

    Keyword Arguments:
        device (Union(torch_tensorrt.Device, torch.device, dict)): Target device for TensorRT engines to run on ::

            device=torch_tensorrt.Device("dla:1", allow_gpu_fallback=True)

        disable_tf32 (bool): Force FP32 layers to use traditional as FP32 format vs the default behavior of rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas
        sparse_weights (bool): Enable sparsity for convolution and fully connected layers.
        enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        refit (bool): Enable refitting
        debug (bool): Enable debuggable engine
        capability (torch_tensorrt.EngineCapability): Restrict kernel selection to safe gpu kernels or safe dla kernels
        num_avg_timing_iters (int): Number of averaging timing iterations used to select kernels
        workspace_size (int): Maximum size of workspace given to TensorRT
        dla_sram_size (int): Fast software managed RAM used by DLA to communicate within a layer.
        dla_local_dram_size (int): Host RAM used by DLA to share intermediate tensor data across operations
        dla_global_dram_size (int): Host RAM used by DLA to store weights and metadata for execution
        truncate_long_and_double (bool): Truncate weights provided in int64 or double (float64) to int32 and float32
        calibrator (Union(torch_tensorrt._C.IInt8Calibrator, tensorrt.IInt8Calibrator)): Calibrator object which will provide data to the PTQ system for INT8 Calibration
        require_full_compilation (bool): Require modules to be compiled end to end or return an error as opposed to returning a hybrid graph where operations that cannot be run in TensorRT are run in PyTorch
        min_block_size (int): The minimum number of contiguous TensorRT convertable operations in order to run a set of operations in TensorRT
        torch_executed_ops (Collection[Target]): Set of aten operators that must be run in PyTorch. An error will be thrown if this set is not empty but ``require_full_compilation`` is True
        torch_executed_modules (List[str]): List of modules that must be run in PyTorch. An error will be thrown if this list is not empty but ``require_full_compilation`` is True
        pass_through_build_failures (bool): Error out if there are issues during compilation (only applicable to torch.compile workflows)
        max_aux_stream (Optional[int]): Maximum streams in the engine
        version_compatible (bool): Build the TensorRT engines compatible with future versions of TensorRT (Restrict to lean runtime operators to provide version forward compatibility for the engines)
        optimization_level: (Optional[int]): Setting a higher optimization level allows TensorRT to spend longer engine building time searching for more optimization options. The resulting engine may have better performance compared to an engine built with a lower optimization level. The default optimization level is 3. Valid values include integers from 0 to the maximum optimization level, which is currently 5. Setting it to be greater than the maximum level results in identical behavior to the maximum level.
        use_python_runtime: (bool): Return a graph using a pure Python runtime, reduces options for serialization
        use_fast_partitioner: (bool): Use the adjacency based partitioning scheme instead of the global partitioner. Adjacency partitioning is faster but may not be optiminal. Use the global paritioner (``False``) if looking for best performance
        enable_experimental_decompositions (bool): Use the full set of operator decompositions. These decompositions may not be tested but serve to make the grap easier to covert to TensorRT, potentially increasing the amount of graphs run in TensorRT.
        output_format (str): Output format of the result of TRT compilation. Options include "exported_program" (or) "ep" | "torchscript" (or) "ts" | "graph_module" (or) "fx". Default is "exported_program"
        **kwargs: Any,
    Returns:
        torch.fx.GraphModule: Compiled FX Module, when run it will execute via TensorRT
    """

    if debug:
        set_log_level(logger.parent, logging.DEBUG)

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

    if not isinstance(exported_program, ExportedProgram):
        raise AssertionError(
            f"Input graph should be an ExportedProgram but got type {type(exported_program)}"
        )
    exported_program = exported_program.run_decompositions(
        get_decompositions(enable_experimental_decompositions)
    )
    gm = exported_program.module()
    logger.debug("Input graph: " + str(gm.graph))

    # Apply lowering on the graph module
    torch_inputs = get_torch_inputs(inputs, device)
    gm = apply_lowering_passes(gm, torch_inputs)
    logger.debug("Lowered Input graph: " + str(gm.graph))

    enabled_precisions = set(enabled_precisions)

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
        "truncate_long_and_double": truncate_long_and_double,
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
        "output_format": output_format,
    }

    settings = CompilationSettings(**compilation_options)
    logger.info("Compilation Settings: %s\n", settings)
    trt_gm = compile_module(gm, inputs, settings)
    trt_result = export(trt_gm, torch_inputs, output_format)
    return trt_result


def compile_module(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[Input],
    settings: CompilationSettings = CompilationSettings(),
) -> torch.fx.GraphModule:
    """Compile a traced FX module

    Includes: Partitioning + Conversion Phases

    Args:
        module: FX GraphModule to convert
        inputs: Inputs to the module
        settings: Compilation settings
    Returns:
        Compiled FX GraphModule
    """

    # Set torch-executed ops
    CONVERTERS.set_disallowed_targets(settings.torch_executed_ops)

    # Check the number of supported operations in the graph
    num_supported_ops, total_ops = partitioning.get_graph_converter_support(
        gm, settings.debug, settings.torch_executed_ops
    )

    # If the number of supported operations is 0 or less than the block size, skip the subgraph
    # TODO: Add condition to second expression below when require_full_compilation is added
    if num_supported_ops == 0 or (num_supported_ops < settings.min_block_size):
        logger.warning(
            f"{num_supported_ops} supported operations detected in subgraph containing {total_ops} computational nodes. "
            f"Skipping this subgraph, since min_block_size was detected to be {settings.min_block_size}"
        )
        return gm
    else:
        logger.debug(
            f"Detected support for {num_supported_ops} operators out of {total_ops} in subgraph."
        )

    # Partition module into components that can be TRT-accelerated
    fast_partitioner_failed = False

    # If specified, try using the fast partitioner and fall back to the global one on failure
    if settings.use_fast_partitioner:
        try:
            partitioned_module = partitioning.fast_partition(
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
        partitioned_module = partitioning.global_partition(
            gm,
            verbose=settings.debug,
            min_block_size=settings.min_block_size,
            torch_executed_ops=settings.torch_executed_ops,
        )

    # Store TRT replicas of Torch subgraphs
    trt_modules = {}
    # Iterate over all components that can be accelerated
    # Generate the corresponding TRT Module for those
    for name, _ in partitioned_module.named_children():
        submodule = getattr(partitioned_module, name)
        # Criteria for a module to be convertible to TRT
        if settings.use_fast_partitioner and "_run_on_acc" not in name:
            continue

        # Get the submodule inputs for min, opt, max shapes of the graph inputs
        submodule_inputs = partitioning.get_submod_inputs(
            partitioned_module,
            submodule,
            sample_inputs,
            to_torch_device(settings.device),
        )

        logger.debug(
            "Submodule name: %s\n Input shapes: %s\n %s",
            str(name),
            [input.shape for input in submodule_inputs],
            str(submodule.graph),
        )

        assert submodule_inputs is not None
        # Handle long/double inputs if requested by the user
        if settings.truncate_long_and_double:
            submodule_inputs = repair_long_or_double_inputs(
                partitioned_module,
                submodule,
                submodule_inputs,
                to_torch_device(settings.device),
                name,
            )

        # Create TRT engines from submodule
        trt_module = convert_module(
            submodule,
            submodule_inputs,
            settings=settings,
            name=name,
        )

        trt_modules[name] = trt_module

    # Replace all FX Modules with TRT Modules
    for name, trt_module in trt_modules.items():
        setattr(partitioned_module, name, trt_module)

    # Reset settings object to user specification after fallback to global partitioning mode
    if fast_partitioner_failed:
        settings.use_fast_partitioner = True

    return partitioned_module
