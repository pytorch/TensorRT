import logging

import torch
from torch_tensorrt._utils import sanitized_torch_version

from packaging import version

# Modify import location of utilities based on Torch version
if version.parse(sanitized_torch_version()) < version.parse("2.1.1"):
    from torch._inductor.freezing import ConstantFolder, replace_node_with_constant
else:
    from torch._inductor.constant_folding import (
        ConstantFolder,
        replace_node_with_constant,
    )

logger = logging.getLogger(__name__)


@torch.utils._python_dispatch._disable_current_modes()  # type: ignore
def constant_fold(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Adapted from:
    https://github.com/pytorch/pytorch/blob/3a79621c9dce17f77fbddc06aab21f6bc477f313/torch/_inductor/freezing.py#L178-L197

    Folds constants in the graph module, not skipping constructors

    Modifies the graph in-place and replaces node with constants
    """
    cf = ConstantFolder(gm, skip_constructors=False)
    cf.run()

    for node, constant in cf.node_replacements.items():
        replace_node_with_constant(gm, node, constant)

    erased_params = []
    for node in gm.graph.nodes:
        # If get_attr node has no users, mark it for deletion
        if node.op == "get_attr" and len(node.users) == 0:
            # If the node's parameter is not a parameter of any other node, remove it
            if not any(
                other.target == node.target for other in gm.graph.nodes if other != node
            ):
                delattr(gm, node.target)
            erased_params.append(node)

    # Remove unused nodes from the graph
    for node in erased_params:
        gm.graph.erase_node(node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()

    logger.debug(f"Graph after constant folding:\n{gm.graph}")

    return gm
