import copy
import logging
import operator

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
            new_node = process_fused_rms_norm_node(node, gm)
            count += 1

    logger.debug(f"Replaced {count} fused rms norm nodes:\n{gm.graph}")

    gm = clean_up_graph_after_modifications(gm)

    return gm


def process_fused_rms_norm_node(
    node: torch.fx.Node, gm: torch.fx.GraphModule
) -> torch.fx.Node:

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
        x_squared = gm.graph.call_function(
            torch.ops.aten.mul.Tensor,
            args=(x, x),
        )
        x_squared_sum = gm.graph.call_function(
            torch.ops.aten.mean.dim,
            args=(x_squared, dims_to_reduce, True),
        )
        x_squared_sum_eps = gm.graph.call_function(
            torch.ops.aten.add.Tensor,
            args=(x_squared_sum, eps),
        )
        x_squared_sum_eps_sqrt = gm.graph.call_function(
            torch.ops.aten.sqrt.default,
            args=(x_squared_sum_eps,),
        )
        x_normalized = gm.graph.call_function(
            torch.ops.aten.div.Tensor,
            args=(x, x_squared_sum_eps_sqrt),
        )
        if weight is not None:
            x_normalized = gm.graph.call_function(
                torch.ops.aten.mul.Tensor,
                args=(x_normalized, weight),
            )

        x_normalized.meta = {}

        for user in list(node.users):
            if user.op == "call_function" and user.target == operator.getitem:
                # If the getitem is extracting the first element (the output tensor)
                if not x_normalized.meta:
                    x_normalized.meta = copy.copy(node.meta)
                user.replace_all_uses_with(x_normalized)
                gm.graph.erase_node(user)

    gm.graph.erase_node(node)

    return x_normalized
