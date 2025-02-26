import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def remove_sym_size_and_constrain_nodes(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Remove aten.sym_size.int and aten.sym_constrain_range_for_size.default ops in the graph"""
    count = 0
    for node in gm.graph.nodes:
        if node.op == "call_function" and (
            node.target == torch.ops.aten.sym_size.int
            or node.target == torch.ops.aten.sym_constrain_range_for_size.default
        ):
            node_input = node.all_input_nodes[0]
            node.replace_all_uses_with(node_input)
            gm.graph.erase_node(node)
            count += 1

    if count > 0:
        gm = clean_up_graph_after_modifications(gm)

    logger.debug(
        f"Removed {count} aten.sym_size.int or aten.sym_constrain_range_for_size.default nodes:\n{gm.graph}"
    )

    return gm
