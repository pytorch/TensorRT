# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Dict, Optional, Set

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

    For square attention (seq_q == seq_k) whose attn_bias is proven to be a causal
    mask: replaces attn_bias with is_causal=True so IAttentionLayer can use its
    native causal path. Any other attn_bias, such as a block-causal or padding mask,
    is kept as it is.

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

        if not _is_proven_causal(gm, attn_bias, query_meta, key_meta):
            logger.debug(
                f"Keeping attn_bias of node {node} because it is not proven to be a causal mask"
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


def _is_proven_causal(
    gm: torch.fx.GraphModule,
    attn_bias: torch.fx.Node,
    query_meta: Optional[torch.Tensor],
    key_meta: Optional[torch.Tensor],
) -> bool:
    """Whether attn_bias is a causal mask for every input.

    That holds only when the bias depends on no graph input, so its value can be
    computed here once, and that value keeps exactly the lower triangle (0) and
    blocks the rest (-inf, or the dtype's lowest value as Hugging Face uses).
    """
    if query_meta is None or key_meta is None:
        return False
    seq_len = query_meta.size(-2)
    if not isinstance(seq_len, int) or seq_len != key_meta.size(-2):
        return False

    producers: Set[torch.fx.Node] = set()
    to_visit = [attn_bias]
    while to_visit:
        current = to_visit.pop()
        if current in producers:
            continue
        if current.op not in ("call_function", "get_attr") or current.is_impure():
            return False
        producers.add(current)
        to_visit.extend(current.all_input_nodes)

    graph = torch.fx.Graph()
    copies: Dict[torch.fx.Node, torch.fx.Node] = {}
    for current in gm.graph.nodes:
        if current in producers:
            copies[current] = graph.node_copy(current, lambda n: copies[n])
    graph.output(copies[attn_bias])
    try:
        bias = torch.fx.GraphModule(gm, graph)()
    except Exception as e:
        logger.debug(f"Could not compute attn_bias {attn_bias}: {e}")
        return False

    if not isinstance(bias, torch.Tensor) or not bias.is_floating_point():
        return False
    try:
        shape = torch.broadcast_shapes(bias.shape, (seq_len, seq_len))
    except RuntimeError:
        return False
    if shape[-2:] != (seq_len, seq_len):
        return False
    # A broadcast dimension repeats one slice, so checking that slice is enough.
    bias = bias[tuple(slice(0, 1) if st == 0 else slice(None) for st in bias.stride())]
    keep = torch.ones(seq_len, seq_len, dtype=torch.bool, device=bias.device).tril()
    keep = keep.expand(torch.broadcast_shapes(bias.shape, keep.shape))
    bias = bias.expand(keep.shape)
    return torch.equal(bias == 0, keep) and torch.equal(
        bias <= torch.finfo(bias.dtype).min, ~keep
    )
