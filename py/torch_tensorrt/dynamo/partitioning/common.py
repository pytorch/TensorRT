import torch
import logging
from typing import Sequence, Set
from torch_tensorrt.dynamo.lowering import SUBSTITUTION_REGISTRY
from torch.fx.node import _get_qualified_name

DEFAULT_SINGLE_NODE_PARTITIONS: Set[str] = set(
    _get_qualified_name(to_replace.new_operator)
    for to_replace in SUBSTITUTION_REGISTRY.values()
)


logger = logging.getLogger(__name__)


def get_submod_inputs(
    mod: torch.fx.GraphModule,
    submod: torch.fx.GraphModule,
    inputs: Sequence[torch.Tensor],
) -> Sequence[torch.Tensor]:
    """Helper function to get inputs to a Torch submodule

    Args:
        mod: Parent FX GraphModule
        submod: Child FX GraphModule
        inputs: Sample inputs to parent module
    Returns:
        Sequence of Tensors representing inputs to child module
    """
    acc_inputs = None

    def get_input(self, inputs):
        nonlocal acc_inputs
        acc_inputs = inputs

    handle = submod.register_forward_pre_hook(get_input)
    mod(*inputs)
    handle.remove()
    return acc_inputs
