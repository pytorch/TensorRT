from __future__ import annotations

import collections.abc
import logging
import warnings
from typing import Any, Collection, List, Optional, Sequence, Set, Tuple, Union

import torch
from torch.export import ExportedProgram
from torch.fx.node import Target
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import EngineCapability, dtype
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo import _defaults, partitioning
from torch_tensorrt.dynamo._DryRunTracker import (
    DryRunTracker,
    PerSubgraphData,
    dryrun_stats_display,
    parse_non_trt_nodes,
)
from torch_tensorrt.dynamo.conversion import (
    CompilationSettings,
    UnsupportedOperatorException,
    convert_module,
    interpret_module_to_result,
    repair_double_inputs,
)
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.lowering import apply_lowering_passes, get_decompositions
from torch_tensorrt.dynamo.utils import (
    get_torch_inputs,
    parse_complex_tensor_structs,
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
    device: Optional[Union[Device, torch.device, str]] = _defaults.DEVICE,
    disable_tf32: bool = _defaults.DISABLE_TF32,
    assume_dynamic_shape_support: bool = _defaults.ASSUME_DYNAMIC_SHAPE_SUPPORT,
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
    """Compile an ExportedProgram module for NVIDIA GPUs using TensorRT

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
        assume_dynamic_shape_support (bool): Setting this to true enables the converters work for both dynamic and static shapes. Default: False
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
        truncate_double (bool): Truncate weights provided in double (float64) to float32
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
        dryrun (bool): Toggle for "Dryrun" mode, running everything except conversion to TRT and logging outputs
        hardware_compatible (bool): Build the TensorRT engines compatible with GPU architectures other than that of the GPU on which the engine was built (currently works for NVIDIA Ampere and newer)
        **kwargs: Any,
    Returns:
        torch.fx.GraphModule: Compiled FX Module, when run it will execute via TensorRT
    """

    if debug:
        set_log_level(logger.parent, logging.DEBUG)

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

    compilation_options = {
        "enabled_precisions": (
            enabled_precisions if enabled_precisions else _defaults.ENABLED_PRECISIONS
        ),
        "debug": debug,
        "device": device,
        "assume_dynamic_shape_support": assume_dynamic_shape_support,
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
    trt_gm = compile_module(gm, inputs, settings)
    return trt_gm


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
    dryrun_tracker = DryRunTracker()

    # Assume converters support dynamic shapes and disable validation
    CONVERTERS.set_dynamic_shape_support(settings.assume_dynamic_shape_support)

    # Set torch-executed ops
    CONVERTERS.set_disallowed_targets(settings.torch_executed_ops)

    # Check the number of supported operations in the graph
    num_supported_ops, total_ops = partitioning.get_graph_converter_support(
        gm, settings.debug, settings.torch_executed_ops
    )

    dryrun_tracker.total_ops_in_graph = total_ops
    dryrun_tracker.supported_ops_in_graph = num_supported_ops
    dryrun_tracker.graph_input_shapes = parse_complex_tensor_structs(
        sample_inputs, "shape", lambda x: dict(x) if isinstance(x, dict) else tuple(x)
    )
    dryrun_tracker.graph_input_dtypes = parse_complex_tensor_structs(
        sample_inputs, "dtype", lambda t: t.to(torch.dtype, use_default=True)
    )
    dryrun_tracker.compilation_settings = settings

    if settings.dryrun and settings.min_block_size > 1:
        logger.info(
            "It is recommended to run `dryrun` mode with `min_block_size=1`, "
            "for the most thorough analysis"
        )

    # If the number of supported operations is 0 or less than the block size, skip the subgraph
    # TODO: Add condition to second expression below when require_full_compilation is added
    if num_supported_ops == 0 or (
        num_supported_ops < settings.min_block_size and not settings.dryrun
    ):
        logger.warning(
            f"{num_supported_ops} supported operations detected in subgraph containing {total_ops} computational nodes. "
            f"Skipping this subgraph, since min_block_size was detected to be {settings.min_block_size}"
        )
        return gm
    else:
        logger.debug(
            f"Detected support for {num_supported_ops} operators out of {total_ops} in subgraph."
        )

    def contains_metadata(gm: torch.fx.GraphModule) -> bool:
        for node in gm.graph.nodes:
            if node.op != "output" and (not node.meta) and "val" not in node.meta:
                logger.warning(
                    f"Node {node.name} of op type {node.op} does not have metadata. This could sometimes lead to undefined behavior."
                )
                return False
        return True

    # Check if the module has metadata (shape, dtype).
    if not contains_metadata(gm):
        # TODO: For future, explore when nodes don't have metadata and if fake_tensor_prop can resolve this.
        logger.warning(
            "Some nodes do not have metadata (shape and dtype information). This could lead to problems sometimes if the graph has PyTorch and TensorRT segments."
        )

    # Partition module into components that can be TRT-accelerated
    fast_partitioner_failed = False

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

    dryrun_tracker.unsupported_ops = supported_ops.unsupported_operators

    # The global partitioner leaves non-TRT nodes as-is
    if not settings.use_fast_partitioner:
        dryrun_tracker.to_run_in_torch.extend(parse_non_trt_nodes(partitioned_module))

    # Store TRT replicas of Torch subgraphs
    trt_modules = {}
    # Iterate over all components that can be accelerated
    # Generate the corresponding TRT Module for those
    for name, _ in partitioned_module.named_children():
        submodule = getattr(partitioned_module, name)
        # Criteria for a module to be convertible to TRT
        if settings.use_fast_partitioner and "_run_on_acc" not in name:
            dryrun_tracker.to_run_in_torch.extend(parse_non_trt_nodes(submodule))
            continue

        subgraph_data = PerSubgraphData()
        subgraph_data.subgraph_name = name
        subgraph_data.subgraph_op_count = len(
            [
                node
                for node in submodule.graph.nodes
                if node.op in ("call_function", "call_method", "call_module")
            ]
        )

        # Get the submodule inputs for min, opt, max shapes of the graph inputs
        submodule_inputs = partitioning.construct_submodule_inputs(submodule)

        logger.debug(
            "Submodule name: %s\n Input shapes: %s\n %s",
            str(name),
            [input.shape for input in submodule_inputs],
            str(submodule.graph),
        )

        assert submodule_inputs is not None
        # Handle long/double inputs if requested by the user
        if settings.truncate_double:
            submodule_inputs = repair_double_inputs(
                partitioned_module,
                submodule,
                submodule_inputs,
                to_torch_device(settings.device),
                name,
            )

        subgraph_data.subgraph_input_shapes = parse_complex_tensor_structs(
            submodule_inputs,
            "shape",
            lambda x: dict(x) if isinstance(x, dict) else tuple(x),
        )
        subgraph_data.subgraph_input_dtypes = parse_complex_tensor_structs(
            submodule_inputs, "dtype", lambda t: t.to(torch.dtype)
        )

        submodule_outputs = submodule(
            *get_torch_inputs(submodule_inputs, to_torch_device(settings.device))
        )

        subgraph_data.subgraph_output_shapes = parse_complex_tensor_structs(
            submodule_outputs,
            "shape",
            lambda x: dict(x) if isinstance(x, dict) else tuple(x),
        )
        subgraph_data.subgraph_output_dtypes = parse_complex_tensor_structs(
            submodule_outputs, "dtype"
        )

        dryrun_tracker.tensorrt_graph_count += 1
        dryrun_tracker.per_subgraph_data.append(subgraph_data)

        # Create TRT engines from submodule
        if not settings.dryrun:
            trt_module = convert_module(
                submodule,
                submodule_inputs,
                settings=settings,
                name=name,
            )

            trt_modules[name] = trt_module

    sample_outputs = gm(
        *get_torch_inputs(sample_inputs, to_torch_device(settings.device))
    )

    if not isinstance(sample_outputs, (list, tuple)):
        sample_outputs = [sample_outputs]

    dryrun_tracker.graph_output_shapes = parse_complex_tensor_structs(
        sample_outputs, "shape", lambda x: dict(x) if isinstance(x, dict) else tuple(x)
    )
    dryrun_tracker.graph_output_dtypes = parse_complex_tensor_structs(
        sample_outputs, "dtype"
    )

    # Replace all FX Modules with TRT Modules
    for name, trt_module in trt_modules.items():
        setattr(partitioned_module, name, trt_module)

    # Reset settings object to user specification after fallback to global partitioning mode
    if fast_partitioner_failed:
        settings.use_fast_partitioner = True

    dryrun_stats_display(dryrun_tracker, settings.dryrun)

    return partitioned_module


def convert_module_to_trt_engine(
    exported_program: ExportedProgram,
    inputs: Tuple[Any, ...],
    *,
    enabled_precisions: (
        Set[torch.dtype | dtype] | Tuple[torch.dtype | dtype]
    ) = _defaults.ENABLED_PRECISIONS,
    debug: bool = _defaults.DEBUG,
    assume_dynamic_shape_support: bool = _defaults.ASSUME_DYNAMIC_SHAPE_SUPPORT,
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
    calibrator: object = None,
    allow_shape_tensors: bool = False,
    **kwargs: Any,
) -> bytes:
    """Convert an ExportedProgram to a serialized TensorRT engine

    Converts an ExportedProgram to a serialized TensorRT engine given a dictionary of conversion settings

    Arguments:
        exported_program (torch.export.ExportedProgram): Source module

    Keyword Args:
        inputs (Optional[Sequence[torch_tensorrt.Input | torch.Tensor]]): **Required** List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
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
        enabled_precisions (Optional[Set[torch.dtype | _enums.dtype]]): The set of datatypes that TensorRT can use
        debug (bool): Whether to print out verbose debugging information
        workspace_size (int): Workspace TRT is allowed to use for the module (0 is default)
        min_block_size (int): Minimum number of operators per TRT-Engine Block
        torch_executed_ops (Set[str]): Set of operations to run in Torch, regardless of converter coverage
        pass_through_build_failures (bool): Whether to fail on TRT engine build errors (True) or not (False)
        max_aux_streams (Optional[int]): Maximum number of allowed auxiliary TRT streams for each engine
        version_compatible (bool): Provide version forward-compatibility for engine plan files
        optimization_level (Optional[int]): Builder optimization 0-5, higher levels imply longer build time,
            searching for more optimization options. TRT defaults to 3
        use_python_runtime (Optional[bool]): Whether to strictly use Python runtime or C++ runtime. To auto-select a runtime
            based on C++ dependency presence (preferentially choosing C++ runtime if available), leave the
            argument as None
        truncate_double (bool): Whether to truncate float64 TRT engine inputs or weights to float32
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
        calibrator (Union(torch_tensorrt._C.IInt8Calibrator, tensorrt.IInt8Calibrator)): Calibrator object which will provide data to the PTQ system for INT8 Calibration
        allow_shape_tensors: (Experimental) Allow aten::size to output shape tensors using IShapeLayer in TensorRT

    Returns:
        bytes: Serialized TensorRT engine, can either be saved to a file or deserialized via TensorRT APIs
    """
    if debug:
        set_log_level(logger.parent, logging.DEBUG)

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

    compilation_options = {
        "assume_dynamic_shape_support": assume_dynamic_shape_support,
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

    # Assume converters support dynamic shapes and disable validation
    CONVERTERS.set_dynamic_shape_support(settings.assume_dynamic_shape_support)

    try:
        interpreter_result = interpret_module_to_result(gm, input_list, settings)
    except UnsupportedOperatorException:
        logger.error(
            f"Conversion of module {gm} not currently fully supported or convertible!",
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            f"While interpreting the module got an error: {e}",
            exc_info=True,
        )

    import io

    with io.BytesIO() as engine_bytes:
        engine_bytes.write(interpreter_result.engine)
        engine_bytearray = engine_bytes.getvalue()

    return engine_bytearray
