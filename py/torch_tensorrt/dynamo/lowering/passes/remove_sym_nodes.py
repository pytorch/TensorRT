import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings

logger = logging.getLogger(__name__)


def remove_sym_nodes(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Remove sym_int placeholders which get inserted due to torch.compile's
    dynamic=True behavior
    """
    # Extract SymInt placeholder Tensors
    placeholder_sym_ints = [
        node
        for node in gm.graph.nodes
        if (
            node.op == "placeholder"
            and isinstance(node.type, type)
            and issubclass(node.type, torch.SymInt)
            and not node.users
        )
    ]

    for node in placeholder_sym_ints:
        gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    logger.debug(f"Removed SymInt placeholders:\n{gm.graph}")

    return gm
