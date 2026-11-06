# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Fold a slice of a constant_pad_nd into the pad itself.

``F.pad(x, (d, 0))[..., :W]`` shifts ``x`` by ``d`` along its last dim. As two
layers, TensorRT writes the whole padded tensor and then copies the slice out
of it, because a slice output cannot be written straight into the buffer of a
concatenation that consumes it. A model that stacks many shifted copies (a
cost volume, for example) then keeps every padded tensor alive at once.
``constant_pad_nd`` accepts negative amounts, which crop, so the pair is one
pad: ``constant_pad_nd(x, (d, -d))``.
"""

import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def fold_slice_into_constant_pad(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Rewrite slice(constant_pad_nd(x, pad)) into constant_pad_nd(x, pad').

    Only when every amount is a plain int, the slice has step 1 and static
    bounds on a dim that the pad pads, keeps at least one element, and the pad
    has no other user.
    """
    del settings
    folded = 0

    for node in list(gm.graph.nodes):
        if (
            node.op != "call_function"
            or node.target != torch.ops.aten.slice.Tensor
            or node.kwargs
        ):
            continue
        pad_node = node.args[0]
        if (
            not isinstance(pad_node, torch.fx.Node)
            or pad_node.target != torch.ops.aten.constant_pad_nd.default
            or pad_node.kwargs
            or len(pad_node.users) != 1
        ):
            continue
        padded = pad_node.meta.get("val")
        pad = list(pad_node.args[1])
        if not isinstance(padded, torch.Tensor) or not all(
            type(amount) is int for amount in pad
        ):
            continue

        dim, start, end, step = (
            list(node.args[1:]) + [0, None, None, 1][len(node.args) - 1 :]
        )[:4]
        if type(dim) is not int or step != 1:
            continue
        rank = padded.dim()
        dim %= rank
        # constant_pad_nd lists (before, after) pairs starting from the last dim.
        pair = rank - 1 - dim
        if 2 * pair + 1 >= len(pad) or not (pad[2 * pair] or pad[2 * pair + 1]):
            continue
        size = padded.shape[dim]
        if type(size) is not int or not all(
            bound is None or type(bound) is int for bound in (start, end)
        ):
            continue
        start = 0 if start is None else start
        end = size if end is None else end
        start = min(max(start + size if start < 0 else start, 0), size)
        end = min(max(end + size if end < 0 else end, 0), size)
        if end <= start:
            continue

        pad[2 * pair] -= start
        pad[2 * pair + 1] -= size - end
        fill = pad_node.args[2] if len(pad_node.args) > 2 else 0

        with gm.graph.inserting_before(node):
            replacement = gm.graph.call_function(
                torch.ops.aten.constant_pad_nd.default,
                args=(pad_node.args[0], pad, fill),
            )
        replacement.meta.update(node.meta)
        node.replace_all_uses_with(replacement)
        gm.graph.erase_node(node)
        gm.graph.erase_node(pad_node)
        folded += 1

    if folded:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Folded {folded} slice(s) into constant_pad_nd")
    return gm
