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
    """Force efficient-attention calls to causal mode when enabled in settings.

    For square attention (seq_q == seq_k): replaces attn_bias with is_causal=True
    so IAttentionLayer can use its native causal path.

    For decode-phase attention (seq_q != seq_k): skip the transformation.
    Applying is_causal=True is semantically wrong because it creates a lower-triangular
    mask aligned to position 0, so the query attends only to k[0] instead of all past keys.
    The node is left unchanged and passed to IAttentionLayer, which supports non-square Q/K natively.
    """
    if not settings.attn_bias_is_causal:
        return gm

    changed = False
    for node in gm.graph.nodes:
        if (
            node.target
            != torch.ops.aten._scaled_dot_product_efficient_attention.default
        ):
            continue

        attn_bias = node.args[3] if len(node.args) > 3 else None
        if attn_bias is None:
            continue

        query_node, key_node = node.args[0], node.args[1]
        query_meta = query_node.meta.get("val") if hasattr(query_node, "meta") else None
        key_meta = key_node.meta.get("val") if hasattr(key_node, "meta") else None
        if (
            query_meta is not None
            and key_meta is not None
            and query_meta.size(-2) != key_meta.size(-2)
        ):
            logger.debug(
                f"Skipping forcing causal pass for node {node} because seq_q={query_meta.size(-2)} != seq_k={key_meta.size(-2)} (decode-phase, IAttentionLayer handles it)"
            )
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
        logger.debug(f"Node {node} changed to causal mode: {node.args}")

    if changed:
        gm = clean_up_graph_after_modifications(gm)

    logger.debug(f"After forcing causal efficient attention pass:\n{gm.graph}")
    return gm
