import logging
from typing import Any, Optional, Sequence, Set

import torch
from torch.fx.node import _get_qualified_name
from torch_tensorrt.dynamo.lowering import SUBSTITUTION_REGISTRY

DEFAULT_SINGLE_NODE_PARTITIONS: Set[str] = {
    _get_qualified_name(to_replace.new_operator)
    for to_replace in SUBSTITUTION_REGISTRY.values()
}


logger = logging.getLogger(__name__)


def get_submod_inputs(
    mod: torch.fx.GraphModule,
    submod: torch.fx.GraphModule,
    inputs: Sequence[torch.Tensor],
) -> Optional[Sequence[torch.Tensor]]:
    """Helper function to get inputs to a Torch submodule

    Args:
        mod: Parent FX GraphModule
        submod: Child FX GraphModule
        inputs: Sample inputs to parent module
    Returns:
        Sequence of Tensors representing inputs to child module
    """
    acc_inputs = None

    def get_input(self: Any, inputs: Sequence[torch.Tensor]) -> None:
        nonlocal acc_inputs
        acc_inputs = inputs
        return

    handle = submod.register_forward_pre_hook(get_input)
    mod(*inputs)
    handle.remove()
    return acc_inputs
