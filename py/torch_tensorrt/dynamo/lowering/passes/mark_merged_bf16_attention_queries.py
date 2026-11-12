# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Mark attentions whose bfloat16 query is computed the same way as another attention's.

TensorRT 11.3 and 11.4 crash while building (a segfault, or an internal error) when one bfloat16
query is read by two attention layers. scaled_dot_product_attention_validator decomposes such
attentions when they read the same node. TensorRT also merges identical work, so two attentions
that compute their query separately, with the same pointwise and view ops from the same inputs,
crash it in the same way. This pass marks those attentions, and the validator decomposes them too.
Identical matrix products and reductions are not merged by TensorRT, so two queries that each come
from their own projection or normalization are left as they are.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    _PASS_THROUGH_OPS,
    MERGED_BF16_QUERY,
    _is_bf16,
    _pass_through_source,
)

logger = logging.getLogger(__name__)

_ATTENTION = torch.ops.aten.scaled_dot_product_attention.default


def _mergeable(node: torch.fx.Node) -> bool:
    # TensorRT 11.4 merges identical pointwise and view work. Identical matrix products and
    # reductions (such as a projection and an RMSNorm in front of the attention) stay separate and
    # build, so attentions whose queries differ there keep their fused kernel.
    if node.op != "call_function" or node.is_impure():
        return False
    tags = getattr(node.target, "tags", ())
    if torch.Tag.nondeterministic_seeded in tags:
        return False
    return torch.Tag.pointwise in tags or node.target in _PASS_THROUGH_OPS


def _canonical_nodes(graph: torch.fx.Graph) -> Dict[torch.fx.Node, torch.fx.Node]:
    """Map each node to the first node that computes it with the same pointwise and view ops from the
    same inputs."""
    canonical: Dict[torch.fx.Node, torch.fx.Node] = {}
    first: Dict[Any, torch.fx.Node] = {}

    def key_of(arg: Any) -> Any:
        if isinstance(arg, torch.fx.Node):
            return ("node", canonical[arg])
        if isinstance(arg, (list, tuple)):
            return (type(arg).__name__, tuple(key_of(a) for a in arg))
        if isinstance(arg, dict):
            return ("dict", tuple((k, key_of(v)) for k, v in sorted(arg.items())))
        # 1 and 1.0 hash alike, but can give different dtypes.
        return (type(arg).__name__, arg)

    for node in graph.nodes:
        canonical[node] = node
        if _mergeable(node):
            key = (node.target, key_of(node.args), key_of(node.kwargs))
            try:
                canonical[node] = first.setdefault(key, node)
            except TypeError:  # an argument that cannot be hashed, such as a SymInt
                pass
    return canonical


def mark_merged_bf16_attention_queries(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    attentions = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function"
        and node.target == _ATTENTION
        and _is_bf16(node.args[0])
    ]
    if len(attentions) < 2:
        return gm

    canonical = _canonical_nodes(gm.graph)
    by_query: Dict[torch.fx.Node, List[torch.fx.Node]] = defaultdict(list)
    for node in attentions:
        query = _pass_through_source(node.args[0])
        if isinstance(query, torch.fx.Node):
            by_query[canonical[query]].append(node)

    marked = [node for nodes in by_query.values() if len(nodes) > 1 for node in nodes]
    for node in marked:
        node.meta[MERGED_BF16_QUERY] = True
    if marked:
        logger.debug(
            "%d attentions read a bfloat16 query computed the same way as another attention's: %s",
            len(marked),
            [node.name for node in marked],
        )
    return gm
