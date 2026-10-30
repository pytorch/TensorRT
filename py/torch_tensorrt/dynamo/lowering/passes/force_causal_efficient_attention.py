# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Any, Dict, Optional, Set, Tuple

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def force_causal_efficient_attention(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Replace a causal attn_bias of efficient attention with is_causal=True.

    IAttentionLayer then uses its native causal path, which is faster and more
    accurate than a mask. ``settings.attn_bias_is_causal`` picks the biases of square
    attention (seq_q == seq_k) to replace:

    * None: a bias proven causal (see _prove_causal).
    * True: also a bias the pass cannot prove, which the caller asserts is causal. A
      bias the pass checks and finds not causal, such as a block-causal mask, is kept.
    * False: none.

    For decode-phase attention (seq_q != seq_k): skip the transformation.
    Applying is_causal=True is semantically wrong because it creates a lower-triangular
    mask aligned to position 0, so the query attends only to k[0] instead of all past keys.
    The node is left unchanged and passed to IAttentionLayer, which supports non-square Q/K natively.
    """
    if settings.attn_bias_is_causal is False:
        return gm

    proofs: Dict[Tuple[torch.fx.Node, Any], Optional[bool]] = {}
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

        seq_len = (
            query_meta.size(-2)
            if query_meta is not None and key_meta is not None
            else None
        )
        key = (attn_bias, seq_len if isinstance(seq_len, int) else None)
        if key not in proofs:
            proofs[key] = _prove_causal(gm, attn_bias, seq_len)
        proof = proofs[key]
        if not (proof or (proof is None and settings.attn_bias_is_causal)):
            logger.debug(
                f"Keeping attn_bias of node {node} because it is "
                + ("not causal" if proof is False else "not proven causal")
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


def _prove_causal(
    gm: torch.fx.GraphModule,
    attn_bias: torch.fx.Node,
    seq_len: Any,
) -> Optional[bool]:
    """Whether attn_bias is a causal mask for every input, or None if unknown.

    The value of a bias that depends on no graph input can be computed here once.
    It is causal when it keeps exactly the lower triangle (0) and blocks the rest
    (-inf, or the dtype's lowest value as Hugging Face uses). It is not causal when
    its zeros are anywhere else, or when it does not block a position above the
    diagonal.
    """
    if not isinstance(seq_len, int):
        return None

    producers: Set[torch.fx.Node] = set()
    to_visit = [attn_bias]
    while to_visit:
        current = to_visit.pop()
        if current in producers:
            continue
        if current.op not in ("call_function", "get_attr") or current.is_impure():
            return None
        producers.add(current)
        to_visit.extend(current.all_input_nodes)

    graph = torch.fx.Graph()
    copies: Dict[torch.fx.Node, torch.fx.Node] = {}
    for current in gm.graph.nodes:
        if current in producers:
            copies[current] = graph.node_copy(current, lambda n: copies[n])
    graph.output(copies[attn_bias])
    # torch.compile lowers under a fake mode, where the bias would have no values.
    with unset_fake_temporarily():
        try:
            return _check_causal(torch.fx.GraphModule(gm, graph)(), seq_len)
        except Exception as e:
            logger.debug(f"Could not check attn_bias {attn_bias}: {e}")
            return None


def _check_causal(bias: Any, seq_len: int) -> Optional[bool]:
    if (
        not isinstance(bias, torch.Tensor)
        or isinstance(bias, FakeTensor)
        or not bias.is_floating_point()
    ):
        return None
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
    if not torch.equal(bias == 0, keep):
        return False
    blocked = bias[~keep]
    if bool((blocked <= torch.finfo(bias.dtype).min).all()):
        return True
    # exp(-1e4) is 0 even in float32, so softmax gives such a position no weight in
    # practice, though that is not a proof for every score.
    return None if bool((blocked <= -1e4).all()) else False
