import logging
from typing import Callable, List, Sequence, Tuple

import torch
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def view_to_reshape(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]
) -> torch.fx.GraphModule:
    """Replace aten.view with an equivalent implementation which avoids Tensor memory issues"""
    orig, replacement = view_replacement()

    if torch.fx.subgraph_rewriter.replace_pattern(gm, orig, replacement):
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Graph after replacing view with reshape:\n{gm.graph}")

    return gm


def view_replacement() -> Tuple[
    torch.fx.GraphModule,
    Callable[[torch.Tensor, List[torch.SymInt]], torch.Tensor],
]:
    """Constructs the original and replacement functions for view"""

    # Original graph
    def orig(input: torch.Tensor, shape: List[torch.SymInt]) -> torch.Tensor:
        return torch.ops.aten.view.default(input, shape)

    # Replacement graph
    def replacement(input: torch.Tensor, shape: List[torch.SymInt]) -> torch.Tensor:
        return torch.ops.aten.reshape.default(input, shape)

    return orig, replacement
