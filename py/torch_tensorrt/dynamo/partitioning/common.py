from typing import Any, Optional, Sequence, Set, Tuple

import torch
from torch.fx.node import _get_qualified_name
from torch_tensorrt.dynamo._defaults import DEBUG
from torch_tensorrt.dynamo.lowering import SUBSTITUTION_REGISTRY

DEFAULT_SINGLE_NODE_PARTITIONS: Set[str] = {
    _get_qualified_name(to_replace.new_operator)
    for to_replace in SUBSTITUTION_REGISTRY.values()
}


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
