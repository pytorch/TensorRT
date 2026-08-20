import logging
from typing import Dict

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def reset_folded_constructors(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Clone folded constructors that escape through a graph boundary.

    A folded constructor is compiler-owned state, unlike a placeholder supplied
    by the caller. If it becomes an output, eager code or a downstream partition
    may mutate it. Cloning at that boundary gives each invocation fresh storage
    while preserving aliases between repeated outputs.

    This pass is intentionally separate from constant folding so it can run
    again after partitioning, when new TensorRT subgraph outputs are known.
    """
    output_node = next(node for node in gm.graph.nodes if node.op == "output")
    clone_cache: Dict[torch.fx.Node, torch.fx.Node] = {}

    def clone_folded_output(node: torch.fx.Node) -> torch.fx.Node:
        if node.op != "get_attr" or not str(node.target).startswith("_frozen_param"):
            return node

        if node not in clone_cache:
            with gm.graph.inserting_before(output_node):
                clone = gm.graph.call_function(
                    torch.ops.aten.clone.default, args=(node,)
                )
                clone.meta.update(node.meta)
            clone_cache[node] = clone

        return clone_cache[node]

    new_output = torch.fx.map_arg(output_node.args[0], clone_folded_output)
    if not clone_cache:
        return gm

    output_node.args = (new_output,)
    gm = clean_up_graph_after_modifications(gm)
    logger.debug("Reset folded constructors at graph outputs:\n%s", gm.graph)
    return gm
