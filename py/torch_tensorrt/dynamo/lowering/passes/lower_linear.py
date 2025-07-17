import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)
from torch_tensorrt.dynamo.utils import get_metadata, set_metadata

logger = logging.getLogger(__name__)


def lower_linear(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Replace aten.linear with an equivalent implementation which can be easily converted to TRT"""
    orig_op = torch.ops.aten.addmm.default
    replacement_op = torch.ops.aten.linear.default

    # Original graph
    def orig(
        input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        W_T = torch.ops.aten.permute.default(weight, [1, 0])
        out = orig_op(bias, input, W_T)
        return out

    # Replacement graph
    def replacement(
        input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return replacement_op(input, weight, bias)

    metadata = get_metadata(gm, orig_op)
    replaced_nodes = torch.fx.subgraph_rewriter.replace_pattern(gm, orig, replacement)

    if len(replaced_nodes) > 0:
        gm = clean_up_graph_after_modifications(gm)
        set_metadata(gm, replacement_op, metadata)
        logger.debug(f"Graph after lowering linear:\n{gm.graph}")

    return gm
