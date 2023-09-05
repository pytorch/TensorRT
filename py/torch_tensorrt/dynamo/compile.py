from __future__ import annotations

import collections.abc
import logging
import operator
from typing import Any, List, Optional, Sequence, Set, Tuple, Union, cast

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
    DEVICE,
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
from torch_tensorrt.dynamo.utils import (
    constant_fold,
    prepare_inputs,
    to_torch_device,
    to_torch_tensorrt_device,
)

logger = logging.getLogger(__name__)


def compile(
    gm: Any,
    inputs: Any,
    *,
    device: Optional[Union[Device, torch.device, str]] = DEVICE,
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

    device = to_torch_tensorrt_device(device)

    _, torch_inputs = prepare_inputs(inputs, to_torch_device(device))

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

        # Create TRT engines from submodule
        trt_module = convert_module(
            submodule,
            submodule_inputs,
            settings=settings,
            name=name,
        )

        # Replace the TRT submodules with TRT nodes
        replace_trt_submodule(partitioned_module, trt_module, sample_inputs)

    # Inline pytorch submodules
    inline_torch_submodules(partitioned_module)

    # Clean the graph
    partitioned_module.graph.eliminate_dead_code()
    partitioned_module.graph.lint()

    # Reset settings object to user specification after fallback to global partitioning mode
    if fast_partitioner_failed:
        settings.use_fast_partitioner = True

    lower_attributes(partitioned_module)
    # import pdb; pdb.set_trace()
    return partitioned_module


def lower_attributes(gm: torch.fx.GraphModule) -> None:
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            # import pdb; pdb.set_trace()
            attribute_name = node.target
            node.target = node.target.lower()
            # node.name = node.name.lower()
            attribute = getattr(gm, attribute_name)
            # delete the existing attribute
            delattr(gm, attribute_name)
            if isinstance(attribute, torch.nn.Parameter):
                gm.register_parameter(attribute_name.lower(), attribute)
            else:
                gm.register_buffer(attribute_name.lower(), attribute)


def replace_trt_submodule(
    gm: torch.fx.GraphModule,
    trt_mod: torch.nn.Module,
    sample_inputs: Sequence[torch.Tensor],
) -> torch.fx.GraphModule:
    """
    Replaces TRT submodules (which are call_module nodes) with TRT nodes(call_function)
    This is necessary for torch.export to work since it cannot handle call_module nodes
    currently.
    """
    # Get submodule IO shape map
    _, submod_outputs_shape_map = partitioning.run_shape_analysis(gm, sample_inputs)

    # Ensure the trt module node in the main graph (gm) has inputs
    submodule_node = [node for node in gm.graph.nodes if node.name == trt_mod.name]
    assert submodule_node
    submodule_node = submodule_node[0]
    assert submodule_node.args

    num_outputs = len(submod_outputs_shape_map[submodule_node.name])
    # Insert a call_function node to perform inference on TRT engine
    with gm.graph.inserting_before(submodule_node):
        trt_node = gm.graph.call_function(
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
                        tuple(submod_outputs_shape_map[trt_mod.name][idx]),
                        tuple([1] * len(submod_outputs_shape_map[trt_mod.name][idx])),
                    ),
                )
            )

    if num_outputs == 1:
        # Insert getitem nodes as outputs (for export serialization to work)
        with gm.graph.inserting_after(trt_node):
            getitem_output = gm.graph.call_function(operator.getitem, (trt_node, 0))
        submodule_node.replace_all_uses_with(getitem_output)
    else:
        # Multiple outputs case:
        # Replace uses of submodule with the trt_node.
        # getitem nodes are already added inherently by the partitioner
        submodule_node.replace_all_uses_with(trt_node)

    # Erase the TRT submodule (call_module) node.
    gm.graph.erase_node(submodule_node)

    return gm


def get_duplicate_nodes(
    gm: torch.fx.GraphModule, submodule: torch.fx.GraphModule
) -> Tuple[Sequence[Any], Sequence[Any]]:
    """
    We check if there are duplicate nodes when we copy submodule graph into gm.
    Handle the case where the subgraph input placeholders are same as
    gm placeholders. This happens when the first submodule in the graph is
    a pytorch submodule
    """
    submodule_placeholder_inputs = [
        node for node in submodule.graph.nodes if node.op == "placeholder"
    ]
    submodule_input_node_names = [node.name for node in submodule_placeholder_inputs]
    gm_node_names = [node.name for node in gm.graph.nodes]
    submodule_duplicate_inputs = [
        node for node in submodule_placeholder_inputs if node.name in gm_node_names
    ]
    gm_duplicate_inputs = [
        node for node in gm.graph.nodes if node.name in submodule_input_node_names
    ]
    return submodule_duplicate_inputs, gm_duplicate_inputs


def inline_torch_submodules(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Inline a submodule within the parent graph (gm). All `call_module` nodes
    should be replaced by their submodule nodes.
    """
    # Clean the graph
    gm.graph.eliminate_dead_code()
    gm.graph.lint()

    for gm_node in gm.graph.nodes:
        if gm_node.op == "call_module" and "_run_on_gpu" in gm_node.name:
            submodule = getattr(gm, gm_node.name)
            with gm.graph.inserting_before(gm_node):
                # Get inputs of submodule node which are most likely outputs of a previous TRT node
                # or a placeholder of the main graph
                submodule_inputs = gm_node.args
                # import pdb; pdb.set_trace()
                submodule_duplicate_inputs, gm_duplicate_inputs = get_duplicate_nodes(
                    gm, submodule
                )
                assert len(submodule_duplicate_inputs) == len(gm_duplicate_inputs)
                # Avoid creating new copies of duplicate inputs by creating a mapping
                val_map = {}
                for i in range(len(submodule_duplicate_inputs)):
                    val_map[submodule_duplicate_inputs[i]] = gm_duplicate_inputs[i]

                # Copy all nodes in the submodule into gm and
                # store the output node of this submodule which is now present in gm

                submodule_output = gm.graph.graph_copy(submodule.graph, val_map)

                # Get their references (since we copied) in the parent graph (gm)
                if len(submodule_duplicate_inputs) == 0:
                    submodule_placeholder_input_names = [
                        node.name
                        for node in submodule.graph.nodes
                        if node.op == "placeholder"
                    ]
                    gm_added_placeholder_inputs = [
                        node
                        for node in gm.graph.nodes
                        if node.name in submodule_placeholder_input_names
                    ]

                    assert len(submodule_inputs) == len(gm_added_placeholder_inputs)

                    # Replace the added placeholder inputs with original inputs to this submodule node
                    for idx in range(len(gm_added_placeholder_inputs)):
                        gm_added_placeholder_inputs[idx].replace_all_uses_with(
                            submodule_inputs[idx]
                        )

                    # Erase the placeholder input nodes in the gm
                    for idx in range(len(gm_added_placeholder_inputs)):
                        gm.graph.erase_node(gm_added_placeholder_inputs[idx])

                # Replace the pytorch submodule node (call_module) with the inlined subgraph output
                gm_node.replace_all_uses_with(submodule_output)

                # copy the attributes of the submodule into gm (graph_copy doesn't do this)
                copy_submodule_attributes(submodule, gm)
            # Erase the pytorch submodule (call_module) node
            gm.graph.erase_node(gm_node)

    return gm


def copy_submodule_attributes(
    submodule: torch.fx.GraphModule, gm: torch.fx.GraphModule
) -> None:
    """
    Copy the getattr attriibutes from submodule to parent module gm.
    The graph_copy call doesn't do this for us unfortunately.
    """
    # Get the submodule attributes mapping
    submodule_attrs = {}
    for node in submodule.graph.nodes:
        if node.op == "get_attr":
            submodule_attr_target = node.target
            submodule_attr = getattr(submodule, submodule_attr_target)
            submodule_attrs[submodule_attr_target] = submodule_attr
    # import pdb; pdb.set_trace()
    # Set the submodule attributes mapping in gm
    for target, attr in submodule_attrs.items():
        if isinstance(attr, torch.nn.Parameter):
            gm.register_parameter(target, attr)
        else:
            gm.register_buffer(target, attr)
