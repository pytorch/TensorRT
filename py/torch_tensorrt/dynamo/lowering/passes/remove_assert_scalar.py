import logging
from typing import Sequence

import torch
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def remove_assert_scalar(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]
) -> torch.fx.GraphModule:
    """Remove assert_scalar ops in the graph"""
    count = 0
    for node in gm.graph.nodes:
        # node.target = "detach" in torch.compile workflow
        if node.target == torch.ops.aten._assert_scalar.default:
            # Detach node has only one input
            gm.graph.erase_node(node)
            count += 1

    if count > 0:
        gm = clean_up_graph_after_modifications(gm)

    logger.debug(f"Removed {count} assert_scalar nodes:\n{gm.graph}")

    return gm
