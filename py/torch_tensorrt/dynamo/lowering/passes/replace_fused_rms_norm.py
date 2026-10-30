# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import operator
from typing import Any

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def replace_fused_rms_norm(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Replace fused rms norm ops in the graph"""
    count = 0
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten._fused_rms_norm.default:
            x_normalized, rsqrt = process_fused_rms_norm_node(node, gm)
            count += 1

    logger.debug(f"Replaced {count} fused rms norm nodes:\n{gm.graph}")

    # Avoid a full DCE/lint/recompile when nothing changed — on large graphs
    # that cleanup alone is tens of ms.
    if count > 0:
        gm = clean_up_graph_after_modifications(gm)

    return gm


def _call(
    gm: torch.fx.GraphModule, target: torch._ops.OpOverload, args: tuple[Any, ...]
) -> torch.fx.Node:
    node = gm.graph.call_function(target, args=args)
    # Later passes read meta["val"], so compute it from the inputs' fake values.
    vals = [
        arg.meta.get("val") if isinstance(arg, torch.fx.Node) else arg for arg in args
    ]
    if all(val is not None for val in vals):
        node.meta["val"] = target(*vals)
    return node


def process_fused_rms_norm_node(
    node: torch.fx.Node, gm: torch.fx.GraphModule
) -> tuple[torch.fx.Node, torch.fx.Node]:

    x, shape, weight, eps = node.args[0], node.args[1], node.args[2], node.args[3]
    if eps is None:
        eps = 1e-5
    # Calculate dimensions to normalize over (similar to layer_norm)
    # normalized_shape specifies the last N dimensions
    x_dim = len(node.meta["val"][0].shape)
    dims_to_reduce = []
    for i in range(len(shape)):
        dims_to_reduce.append(x_dim - i - 1)

    with gm.graph.inserting_before(node):
        # Replace fused rms norm with standard rms norm
        x_squared = _call(gm, torch.ops.aten.mul.Tensor, (x, x))

        x_squared_sum = _call(
            gm, torch.ops.aten.mean.dim, (x_squared, dims_to_reduce, True)
        )

        x_squared_sum_eps = _call(gm, torch.ops.aten.add.Tensor, (x_squared_sum, eps))

        x_squared_sum_eps_rsqrt = _call(
            gm, torch.ops.aten.rsqrt.default, (x_squared_sum_eps,)
        )

        x_normalized = _call(
            gm, torch.ops.aten.mul.Tensor, (x, x_squared_sum_eps_rsqrt)
        )

        if weight is not None:
            x_normalized = _call(gm, torch.ops.aten.mul.Tensor, (x_normalized, weight))

        for i, user in enumerate(list(node.users)):
            if user.op == "call_function" and user.target == operator.getitem:
                if i == 0:
                    # If the getitem is extracting the first element (the output tensor)
                    user.replace_all_uses_with(x_normalized)
                else:
                    user.replace_all_uses_with(x_squared_sum_eps_rsqrt)

                logger.debug(
                    f"Replaced {i}-th user of fused_rms_norm node [{user}] with lowered rms_norm output [{x_normalized if i == 0 else x_squared_sum_eps_rsqrt}]"
                )
                gm.graph.erase_node(user)

    gm.graph.erase_node(node)

    return x_normalized, x_squared_sum_eps_rsqrt
