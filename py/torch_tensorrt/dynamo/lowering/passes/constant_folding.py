import logging
from typing import Any, Sequence

import torch
from torch._inductor.constant_folding import ConstantFolder, replace_node_with_constant
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


@torch.utils._python_dispatch._disable_current_modes()  # type: ignore
def constant_fold(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]
) -> torch.fx.GraphModule:
    """Adapted from:
    https://github.com/pytorch/pytorch/blob/3a79621c9dce17f77fbddc06aab21f6bc477f313/torch/_inductor/freezing.py#L178-L197

    Folds constants in the graph module, not skipping constructors

    Modifies the graph in-place and replaces node with constants
    """
    cf = _TorchTensorRTConstantFolder(gm, skip_constructors=False)
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

    gm = clean_up_graph_after_modifications(gm)

    logger.debug(f"Graph after constant folding:\n{gm.graph}")

    return gm


# TODO: Delete this class when the following code is fixed in nightly:
# https://github.com/pytorch/pytorch/blob/4b881b0da390c1290bb12850ef9daad6f6eb2cb6/torch/_inductor/constant_folding.py#L53-L63
class _TorchTensorRTConstantFolder(ConstantFolder):  # type: ignore[misc]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    # TODO: Update this function when quantization is added
    def is_impure(self, node: torch.fx.node.Node) -> bool:
        return False
