import logging
from typing import Callable, Tuple

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)
from torch_tensorrt.dynamo.utils import get_output_meta_val, set_output_meta_val

logger = logging.getLogger(__name__)


def lower_linear(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Replace aten.linear with an equivalent implementation which can be easily converted to TRT"""

    outputs = [node for node in gm.graph.nodes if node.op == "output"]
    outputs = outputs[0].args
    outputs_meta_val = get_output_meta_val(outputs)

    orig, replacement = linear_replacement()
    replaced_nodes = torch.fx.subgraph_rewriter.replace_pattern(gm, orig, replacement)

    if len(replaced_nodes) > 0:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Graph after lowering linear:\n{gm.graph}")

        outputs = [node for node in gm.graph.nodes if node.op == "output"]
        outputs = outputs[0].args
        output_num = len(outputs_meta_val)
        assert output_num > 0
        set_output_meta_val(outputs, outputs_meta_val)
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
