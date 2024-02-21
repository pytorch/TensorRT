from typing import Any, List

import torch


def clean_up_graph_after_modifications(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Runs dead-code elimination, linting, and recompilation for graph, in-place"""
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


def get_tensor_placeholders(
    gm: torch.fx.GraphModule,
) -> List[torch.fx.Node]:
    """Returns placeholder nodes of GraphModule which are torch.Tensor types"""
    # Tensor placeholders must be subclasses of torch.Tensor
    placeholders = [
        node
        for node in gm.graph.nodes
        if (
            node.op == "placeholder"
            and isinstance(node.type, type)
            and issubclass(node.type, torch.Tensor)
        )
    ]

    return placeholders


def get_metadata(
    gm: torch.fx.GraphModule, target_op: Any
) -> List[torch._ops.OpOverload]:
    """
    Return the list which has the metadata of all the target_op nodes present in the graph.
    """
    return [node.meta for node in gm.graph.nodes if node.target == target_op]


def set_metadata(
    gm: torch.fx.GraphModule, target_op: Any, metadata: List[torch._ops.OpOverload]
) -> None:
    """
    Return the list which has the metadata of all the target_op nodes present in the graph.
    """
    target_nodes = [node for node in gm.graph.nodes if node.target == target_op]
    assert len(target_nodes) == len(metadata)
    for idx, node in enumerate(target_nodes):
        node.meta = metadata[idx]
