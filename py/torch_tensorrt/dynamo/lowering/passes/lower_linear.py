import logging
from typing import Callable, Sequence, Tuple

import torch
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def lower_linear(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]
) -> torch.fx.GraphModule:
    """Replace aten.linear with an equivalent implementation which can be easily converted to TRT"""
    orig, replacement = linear_replacement()

    if torch.fx.subgraph_rewriter.replace_pattern(gm, orig, replacement):
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Graph after lowering linear:\n{gm.graph}")

    return gm


def linear_replacement() -> Tuple[
    torch.fx.GraphModule,
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
]:
    """Constructs the original and replacement functions for linear"""

    # Original graph
    def orig(
        input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        W_T = torch.ops.aten.permute.default(weight, [1, 0])
        out = torch.ops.aten.addmm.default(bias, input, W_T)
        return out

    # Replacement graph
    def replacement(
        input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return torch.ops.aten.linear.default(input, weight, bias)

    return orig, replacement
