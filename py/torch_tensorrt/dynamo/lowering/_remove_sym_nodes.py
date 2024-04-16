import logging

import torch

logger = logging.getLogger(__name__)


def remove_sym_nodes(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Remove sym_int placeholders which get inserted due to torch.compile's
    dynamic=True behavior
    """
    # Extract SymInt placeholder Tensors
    placeholders = [
        node
        for node in gm.graph.nodes
        if (
            node.op == "placeholder"
            and isinstance(node.type, type)
            and issubclass(node.type, torch.SymInt)
        )
    ]

    for node in placeholders:
        gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    logger.debug(f"Removed SymInt placeholders:\n{gm.graph}")

    return gm
