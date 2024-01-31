import logging
import operator
from typing import Callable, Sequence, Tuple

import torch
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def lower_efficient_attention(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]
) -> torch.fx.GraphModule:
    """Replace a specific version of scaled_dot_product_attention with an equivalent
    implementation which can be easily converted to TRT
    """
    orig, replacement = efficient_attention_replacement()

    if torch.fx.subgraph_rewriter.replace_pattern(gm, orig, replacement):
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(
            f"Graph after lowering _scaled_dot_product_efficient_attention:\n{gm.graph}"
        )

    return gm


def efficient_attention_replacement() -> Tuple[
    torch.fx.GraphModule,
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
]:
    """Constructs the original and replacement functions for efficient attention"""

    # Original graph
    def orig(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        outputs = torch.ops.aten._scaled_dot_product_efficient_attention.default(
            q, k, v, None, False
        )
        out = operator.getitem(outputs, 0)
        return out

    # Replacement graph consists of the functional version of scaled_dot_product_attention
    def replacement(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    return orig, replacement
