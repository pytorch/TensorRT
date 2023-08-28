from __future__ import annotations

import collections.abc
import logging
import operator
from typing import Any, List, Optional, Sequence, Set, Tuple, cast

import torch
import torch_tensorrt
from torch._subclasses.fake_tensor import FakeTensor
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import (  # TODO: Should probabably be the TRT EngineCapability Enum
    EngineCapability,
)
from torch_tensorrt.dynamo import CompilationSettings, partitioning
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
from torch_tensorrt.dynamo.conversion import (
    convert_module,
    repair_long_or_double_inputs,
)
from torch_tensorrt.dynamo.utils import constant_fold, prepare_device, prepare_inputs

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
        if logger.parent:
            logger.parent.setLevel(logging.DEBUG)

    enabled_precisions = set(enabled_precisions)

    logger.warning(
        "The Dynamo backend is an experimental feature, for which only the "
        "following arguments are supported: "
        "{enabled_precisions, debug, workspace_size, min_block_size, "
        "torch_executed_ops, pass_through_build_failures, use_fast_partitioner, "
        "enable_experimental_decompositions}"
    )

    logger.debug("Post export graph: " + str(gm.graph))

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
    logger.info("Compilation Settings: %s\n", settings)
    # Run constant folding before TRT compilation
    constant_fold(gm)
    return compile_module(gm, torch_inputs, settings)


def compile_module(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[torch.Tensor],
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

    # Get submodule IO shape map
    submod_inputs_shape_map, submod_outputs_shape_map = partitioning.run_shape_analysis(
        partitioned_module, sample_inputs
    )

    # Iterate over all components that can be accelerated
    # Generate the corresponding TRT Module for those
    for name, _ in partitioned_module.named_children():
        submodule = getattr(partitioned_module, name)
        logger.debug(
            "Submodule name: " + str(name) + " Graph: \n" + str(submodule.graph)
        )

        # Criteria for a module to be convertible to TRT
        if settings.use_fast_partitioner and "_run_on_acc" not in name:
            continue

        # Ensure the submodule has inputs
        submodule_node = [
            node for node in partitioned_module.graph.nodes if node.name == name
        ]
        assert submodule_node
        submodule_node = submodule_node[0]
        assert submodule_node.args

        # Get example submodule inputs
        submodule_inputs = partitioning.get_submod_inputs(
            partitioned_module, submodule, sample_inputs
        )

        assert submodule_inputs is not None
        # Handle long/double inputs if requested by the user
        if settings.truncate_long_and_double:
            submodule_inputs = repair_long_or_double_inputs(
                partitioned_module, submodule, submodule_inputs, name
            )

        # Create TRT Module from submodule
        trt_mod = convert_module(
            submodule,
            submodule_inputs,
            settings=settings,
            name=name,
        )

        num_outputs = len(submod_outputs_shape_map[name])
        # Insert a call_function node to perform inference on TRT engine
        with partitioned_module.graph.inserting_before(submodule_node):
            trt_node = partitioned_module.graph.call_function(
                torch.ops.tensorrt.execute_engine.default,
                (submodule_node.args, trt_mod.engine),
            )
            trt_node.meta["val"] = []
            # Generate meta data for TRT node (a FakeTensor with corresponding output shape)
            for idx in range(num_outputs):
                trt_node.meta["val"].append(
                    cast(
                        FakeTensor,
                        torch.empty_strided(
                            tuple(submod_outputs_shape_map[name][idx]),
                            tuple([1] * len(submod_outputs_shape_map[name][idx])),
                        ),
                    )
                )

        if num_outputs == 1:
            # Insert getitem nodes as outputs (for export serialization to work)
            with partitioned_module.graph.inserting_after(trt_node):
                getitem_output = partitioned_module.graph.call_function(
                    operator.getitem, (trt_node, 0)
                )
            submodule_node.replace_all_uses_with(getitem_output)
        else:
            # Multiple outputs case:
            # Replace uses of submodule with the trt_node.
            # getitem nodes are already added inherently by the partitioner
            submodule_node.replace_all_uses_with(trt_node)

        # Erase the TRT submodule
        partitioned_module.graph.erase_node(submodule_node)

    # Inline pytorch submodules
    partitioning.inline_pytorch_submodules(partitioned_module)

    # Clean the graph
    partitioned_module.graph.eliminate_dead_code()
    partitioned_module.graph.lint()

    # Reset settings object to user specification after fallback to global partitioning mode
    if fast_partitioner_failed:
        settings.use_fast_partitioner = True

    return partitioned_module
