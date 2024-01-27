from typing import Any, Dict, List

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


def update_metadata(
    gm: torch.fx.GraphModule, target_op: Any, metadata: Dict[int, torch._ops.OpOverload]
) -> None:
    """
    Given a graph and a node which has target_op in the graph,
    a) If the node has metadata, store it in the map
    b) If the node does not have metadata, retrieve it from the map
       and assign to the node.
    """
    for idx, node in enumerate(gm.graph.nodes):
        if node.target == target_op:
            if idx not in metadata and node.meta:
                metadata[idx] = node.meta
            elif idx in metadata and not node.meta:
                node.meta = metadata[idx]
