import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def force_causal_efficient_attention(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Force efficient-attention calls to causal mode when enabled in settings."""
    if not settings.attn_bias_is_causal:
        return gm

    changed = False
    for node in gm.graph.nodes:
        if (
            node.target
            == torch.ops.aten._scaled_dot_product_efficient_attention.default
        ):
            attn_bias = node.args[3] if len(node.args) > 3 else None
            if attn_bias is None:
                continue
            node.args = (
                node.args[0],
                node.args[1],
                node.args[2],
                None,
                False,
                0.0,
                True,
            )
            changed = True
            logger.debug(
                f"The args of node {node} was changed to causal mode. Now the node's arguments are: {node.args}"
            )

    if changed:
        gm = clean_up_graph_after_modifications(gm)

    logger.debug(f"After forcing causal efficient attention pass:\n{gm.graph}")
    return gm
