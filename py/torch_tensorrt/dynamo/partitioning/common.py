from typing import Any, Dict, Optional, Sequence, Set, Tuple

import torch
from torch.fx.node import _get_qualified_name
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._defaults import DEBUG
from torch_tensorrt.dynamo.lowering import SUBSTITUTION_REGISTRY

DEFAULT_SINGLE_NODE_PARTITIONS: Set[str] = {
    _get_qualified_name(to_replace.new_operator)
    for to_replace in SUBSTITUTION_REGISTRY.values()
}


def inline_pytorch_submodules(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Inline a submodule within the parent graph (gm). All `call_module` nodes
    should be replaced by their submodule nodes.
    """
    # Clean the graph
    gm.graph.eliminate_dead_code()
    gm.graph.lint()

    for node in gm.graph.nodes:
        if node.op == "call_module":
            submodule = getattr(gm, node.name)
            with gm.graph.inserting_before(node):
                # Copy all nodes in the submodule into gm and
                # store the output node of this submodule which is now present in gm
                submodule_output = gm.graph.graph_copy(submodule.graph, {})

                # Get inputs of submodule node which are most likely outputs of a previous TRT node
                submodule_inputs = node.args

                # Gather the placeholder input names from submodule graph
                submodule_placeholder_input_names = [
                    node.name
                    for node in submodule.graph.nodes
                    if node.op == "placeholder"
                ]

                # Get their references (since we copied) in the parent graph (gm)
                gm_added_placeholder_inputs = [
                    node
                    for node in gm.graph.nodes
                    if node.name in submodule_placeholder_input_names
                ]

                assert len(submodule_inputs) == len(gm_added_placeholder_inputs)

                # Replace the added placeholder inputs with original inputs to this submodule node
                for idx in range(len(submodule_inputs)):
                    gm_added_placeholder_inputs[idx].replace_all_uses_with(
                        submodule_inputs[idx]
                    )

                # Erase the placeholder input nodes in the gm
                for idx in range(len(gm_added_placeholder_inputs)):
                    gm.graph.erase_node(gm_added_placeholder_inputs[idx])

                # Replace the pytorch submodule node (call_module) with the inlined subgraph output
                node.replace_all_uses_with(submodule_output)

            # Erase the pytorch submodule (call_module) node
            gm.graph.erase_node(node)

    return gm


def run_shape_analysis(
    parent_module: torch.fx.GraphModule, inputs: Sequence[Input]
) -> Tuple[Dict[Any, Sequence[Any]], Dict[Any, Sequence[Any]]]:
    submod_inputs_shape_map: Dict[Any, Sequence[Any]] = {}
    submod_outputs_shape_map: Dict[Any, Sequence[Any]] = {}
    sub_inputs: Sequence[torch.Tensor] = []
    sub_outputs: Sequence[torch.Tensor] = []

    # Register a hook to capture IO shapes for submodules
    def get_submodule_io(
        self: Any, inputs: Sequence[torch.Tensor], outputs: Sequence[torch.Tensor]
    ) -> None:
        nonlocal sub_inputs, sub_outputs
        sub_inputs = inputs
        sub_outputs = outputs
        return

    # Iterate through submodules (both Torch and TRT) and store IO shapes
    for name, _ in parent_module.named_children():
        submodule = getattr(parent_module, name)
        handle = submodule.register_forward_hook(get_submodule_io)
        parent_module(*inputs)
        handle.remove()
        submod_inputs_shape_map[name] = (
            [input.shape for input in sub_inputs]
            if isinstance(sub_inputs, (tuple, list))
            else [sub_inputs.shape]
        )
        submod_outputs_shape_map[name] = (
            [output.shape for output in sub_outputs]
            if isinstance(sub_outputs, (tuple, list))
            else [sub_outputs.shape]
        )

    return submod_inputs_shape_map, submod_outputs_shape_map


def get_submod_inputs(
    mod: torch.fx.GraphModule,
    submod: torch.fx.GraphModule,
    inputs: Sequence[torch.Tensor],
) -> Optional[Sequence[torch.Tensor]]:
    """Helper function to get inputs to a Torch submodule

    Args:
        mod: Parent FX GraphModule
        submod: Child FX GraphModule
        inputs: Sample inputs to parent module
    Returns:
        Sequence of Tensors representing inputs to child module
    """
    acc_inputs = None

    def get_input(self: Any, inputs: Sequence[torch.Tensor]) -> None:
        nonlocal acc_inputs
        acc_inputs = inputs
        return

    handle = submod.register_forward_pre_hook(get_input)
    mod(*inputs)
    handle.remove()
    return acc_inputs


def get_graph_converter_support(
    graph_module: torch.fx.GraphModule,
    verbose: bool = DEBUG,
    torch_executed_ops: Optional[Set[str]] = None,
) -> Tuple[int, int]:
    """Helper function to get converter support overview pre-partitioning

    Args:
        graph_module: FX GraphModule to determine support for
        verbose: Bool representing whether to print operator support
        torch_executed_ops: Collection of operations to run in Torch, regardless of converter coverage
    Returns:
        The number of supported call_function nodes in the graph
    """
    from ._global_partitioner import TorchTensorRTOperatorSupport

    # Instantiate operator support object and module dictionary
    op_support = TorchTensorRTOperatorSupport(torch_executed_ops=torch_executed_ops)
    module_dict = dict(graph_module.named_modules())

    number_of_supported_nodes = 0
    total_functional_nodes = 0

    # Iterate over all nodes in the graph, enumerating call_function nodes
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            total_functional_nodes += 1

            if op_support.is_node_supported(module_dict, node):
                number_of_supported_nodes += 1

    # Print node support overview prior to partitioning
    if verbose:
        op_support.print_support_overview(print_node_support=True)

    return number_of_supported_nodes, total_functional_nodes
