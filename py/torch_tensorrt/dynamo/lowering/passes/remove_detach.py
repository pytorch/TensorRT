import logging
from typing import Sequence

import torch
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def remove_detach(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]
) -> torch.fx.GraphModule:
    """Remove detach ops in the graph"""

    modified_graph = False
    count = 0
    for node in gm.graph.nodes:
        # If the node is a detach node
        if node.target == torch.ops.aten.detach.default:
            # Detach node has only one input
            node_input = node.all_input_nodes[0]
            node.replace_all_uses_with(node_input)
            gm.graph.erase_node(node)
            count += 1

    if modified_graph:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Removed {count} detach nodes:\n{gm.graph}")

    return gm
