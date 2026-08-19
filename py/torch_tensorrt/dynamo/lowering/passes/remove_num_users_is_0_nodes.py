import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def apply_remove_num_users_is_0_nodes(gm: torch.fx.GraphModule) -> bool:
    """Remove unused ops in-place. Returns True if the graph changed."""
    nodes = list(gm.graph.nodes)
    if not nodes:
        return False

    output_node = nodes[-1]
    erased = 0
    for node in nodes[::-1]:
        if (
            node != output_node
            and len(node.users) == 0
            and len(node.all_input_nodes) > 0
        ):
            gm.graph.erase_node(node)
            erased += 1

    if erased:
        logger.debug("Removed %d num_users=0 nodes", erased)
    return erased > 0


def remove_num_users_is_0_nodes(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Remove ops that [num_users=0] in the graph"""
    del settings
    if apply_remove_num_users_is_0_nodes(gm):
        gm = clean_up_graph_after_modifications(gm)
        logger.debug("Graph after remove_num_users_is_0_nodes:\n%s", gm.graph)
    return gm
