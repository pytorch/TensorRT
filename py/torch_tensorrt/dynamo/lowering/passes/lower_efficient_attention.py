import logging
import operator
from typing import Callable, Sequence, Tuple

import torch
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
    get_tensor_placeholders,
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


def efficient_attention_replacement() -> (
    Tuple[
        torch.fx.GraphModule,
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ]
):
    """Constructs the original and replacement functions for efficient attention"""

    # Empty boilerplate function taking in three Tensors and returning one
    def boilerplate(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        ...

    # Trace boilerplate function and extract placeholder and output nodes
    orig = torch.fx.symbolic_trace(boilerplate)
    q, k, v = get_tensor_placeholders(orig)
    output = [node for node in orig.graph.nodes if node.op == "output"][0]

    # Graph types to replace are those which use the _scaled_dot_product_efficient_attention
    # function and extract only the first element
    with orig.graph.inserting_before(output):
        att = orig.graph.call_function(
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            args=(q, k, v, None, False),
        )
        out = orig.graph.call_function(
            operator.getitem,
            args=(att, 0),
        )

    # Assign the output of the graph to be the single getitem output
    output.args = (out,)

    orig.graph.lint()
    orig.recompile()

    # Replacement graph consists of the functional version of scaled_dot_product_attention
    def replacement(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    return orig, replacement
