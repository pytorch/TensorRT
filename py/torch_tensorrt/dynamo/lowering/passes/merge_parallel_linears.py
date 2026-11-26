# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Dict, List, Tuple

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def merge_parallel_linears(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Run linear ops that read the same input with constant weights as one linear op.

    A decoder layer projects one input to queries, keys, values and more. With few
    rows each projection is a separate pass over a small weight; one pass over the
    stacked weights reads the same bytes with fewer, larger kernels. Each original
    op then reads its columns of the merged output. Only ops the matrix vector kernel
    takes are merged, because TensorRT's own matrix multiply for one row gets slower
    as the weight grows. The stacked weight is built once per set of weights, so a
    layer that runs several times shares it.
    """
    if not settings.matrix_vector_kernel:
        return gm
    from torch_tensorrt.dynamo.conversion import aten_ops_converters

    if (
        aten_ops_converters._grouped_mm_kernel_blocked(settings)
        or aten_ops_converters._grouped_mm_kernel_unavailable()
    ):
        return gm
    order = {node: i for i, node in enumerate(gm.graph.nodes)}
    stacked: Dict[Tuple[str, ...], torch.fx.Node] = {}
    merged = 0
    for node in list(gm.graph.nodes):
        if not hasattr(node.meta.get("val"), "fake_mode"):
            continue
        groups: Dict[Tuple[torch.dtype, int], List[torch.fx.Node]] = {}
        for user in node.users:
            if (
                user.target != torch.ops.aten.linear.default
                or user.args[0] is not node
                or (len(user.args) > 2 and user.args[2] is not None)
                or aten_ops_converters._matrix_vector_kernel_shape(user) is None
            ):
                continue
            weight = user.args[1].meta["val"]
            groups.setdefault((weight.dtype, weight.shape[1]), []).append(user)
        for linears in groups.values():
            if len(linears) < 2:
                continue
            linears.sort(key=order.__getitem__)
            weights = [linear.args[1] for linear in linears]
            key = tuple(weight.target for weight in weights)
            if key not in stacked:
                with gm.graph.inserting_after(max(weights, key=order.__getitem__)):
                    cat = gm.graph.call_function(
                        torch.ops.aten.cat.default, (weights, 0)
                    )
                # The input's fake tensor mode describes the stacked weight without
                # allocating it; the weights' own metadata may be real tensors.
                with node.meta["val"].fake_mode:
                    cat.meta["val"] = torch.empty(
                        (
                            sum(w.meta["val"].shape[0] for w in weights),
                            weights[0].meta["val"].shape[1],
                        ),
                        dtype=weights[0].meta["val"].dtype,
                        device=weights[0].meta["val"].device,
                    )
                stacked[key] = cat
            first = linears[0].meta["val"]
            with gm.graph.inserting_before(linears[0]):
                fused = gm.graph.call_function(
                    torch.ops.aten.linear.default, (node, stacked[key])
                )
                fused.meta["val"] = first.new_empty(
                    (*first.shape[:-1], stacked[key].meta["val"].shape[0])
                )
                start = 0
                for linear in linears:
                    end = start + linear.args[1].meta["val"].shape[0]
                    piece = gm.graph.call_function(
                        torch.ops.aten.slice.Tensor, (fused, -1, start, end)
                    )
                    piece.meta["val"] = linear.meta["val"]
                    linear.replace_all_uses_with(piece)
                    start = end
            for linear in linears:
                gm.graph.erase_node(linear)
            merged += 1
    if merged:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Merged {merged} groups of parallel linear ops")
    return gm
