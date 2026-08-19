import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def apply_remove_assert_nodes(gm: torch.fx.GraphModule) -> bool:
    """Remove assert ops in-place. Returns True if the graph changed."""
    count = 0
    for node in list(gm.graph.nodes):
        if (
            node.target == torch.ops.aten._assert_scalar.default
            or node.target == torch.ops.aten._assert_tensor_metadata.default
        ):
            gm.graph.erase_node(node)
            count += 1

    if count:
        logger.debug("Removed %d assert nodes", count)
    return count > 0


def remove_assert_nodes(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Remove assert_scalar ops in the graph"""
    del settings
    if apply_remove_assert_nodes(gm):
        gm = clean_up_graph_after_modifications(gm)
        logger.debug("Graph after remove_assert_nodes:\n%s", gm.graph)
    return gm
