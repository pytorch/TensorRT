import logging
from typing import Sequence

import torch
import torch.fx
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def replace_full_like_with_full(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]
) -> torch.fx.GraphModule:
    """Replace full_like nodes with equivalent full nodes"""
    modified_graph = False

    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.full_like.default:
            modified_graph = True

            # Extract arguments from full_like
            input_tensor = node.args[0]
            fill_value = node.args[1]
            shape = list(input_tensor.meta["tensor_meta"].shape)

            # Replace full_like with full, using the shape as a list
            with gm.graph.inserting_after(node):
                full_node = gm.graph.call_function(
                    torch.ops.aten.full.default,
                    args=(shape, fill_value),
                    kwargs=node.kwargs,
                )
                full_node.meta = node.meta

            node.replace_all_uses_with(full_node)
            gm.graph.erase_node(node)

    if modified_graph:
        gm = clean_up_graph_after_modifications(gm)

    return gm
