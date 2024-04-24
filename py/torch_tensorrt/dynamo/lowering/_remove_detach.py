import logging

import torch
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def remove_detach(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Remove detach ops from the graph
    """
    modified_graph = False

    for node in gm.graph.nodes:
        # If the node is a detach node
        if len(node.users) == 1 and list(node.users)[0].target == "detach":
            modified_graph = True
            detach_node = list(node.users)[0]
            logger.debug(
                f"Removing node {detach_node} from the graph. It is a detach node with a single user."
            )
            detach_node.replace_all_uses_with(node)
            gm.graph.erase_node(detach_node)

    if modified_graph:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Removed detach nodes:\n{gm.graph}")

    return gm
