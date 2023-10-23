import logging
from typing import Callable, Sequence, Tuple

import torch
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
    get_tensor_placeholders,
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


def linear_replacement() -> (
    Tuple[
        torch.fx.GraphModule,
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ]
):
    """Constructs the original and replacement functions for linear"""

    # Empty boilerplate function taking in three Tensors and returning one
    def boilerplate(
        input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        ...

    # Trace boilerplate function and extract placeholder and output nodes
    orig = torch.fx.symbolic_trace(boilerplate)
    input, weight, bias = get_tensor_placeholders(orig)
    output = [node for node in orig.graph.nodes if node.op == "output"][0]

    with orig.graph.inserting_before(output):
        W_T = orig.graph.call_function(
            torch.ops.aten.permute.default,
            args=(weight, [1, 0]),
        )
        out = orig.graph.call_function(
            torch.ops.aten.addmm.default,
            args=(bias, input, W_T),
        )

    # Assign the output of the graph to be the single getitem output
    output.args = (out,)

    orig.graph.lint()
    orig.recompile()

    # Replacement graph
    def replacement(
        input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return torch.ops.aten.linear.default(input, weight, bias)

    return orig, replacement
