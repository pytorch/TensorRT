import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings

logger = logging.getLogger(__name__)


def replace_fused_rms_norm(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Replace fused rms norm ops in the graph"""
    count = 0
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten._fused_rms_norm.default:
            # Replace fused rms norm with standard rms norm
            new_node = gm.graph.call_function(
                torch.ops.aten.rms_norm.default,
                args=node.args,
            )
            gm.graph.replace_node_with_new_node(node, new_node)
            gm.graph.erase_node(node)
            count += 1

    logger.debug(f"Replaced {count} fused rms norm nodes:\n{gm.graph}")

    return gm
