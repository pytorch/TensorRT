import logging
from typing import Dict, List, Sequence

import torch
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
    update_metadata,
)

logger = logging.getLogger(__name__)


def view_to_reshape(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]
) -> torch.fx.GraphModule:
    """Replace aten.view with an equivalent implementation which avoids Tensor memory issues"""
    orig_op = torch.ops.aten.view.default
    replacement_op = torch.ops.aten.reshape.default

    # Original graph
    def orig(input: torch.Tensor, shape: List[torch.SymInt]) -> torch.Tensor:
        return orig_op(input, shape)

    # Replacement graph
    def replacement(input: torch.Tensor, shape: List[torch.SymInt]) -> torch.Tensor:
        return replacement_op(input, shape)

    # Store metadata of the orig_op and copy it to the replacement op
    meta_map: Dict[int, torch._ops.OpOverload] = {}
    update_metadata(gm, orig_op, meta_map)

    if torch.fx.subgraph_rewriter.replace_pattern(gm, orig, replacement):
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Graph after replacing view with reshape:\n{gm.graph}")

    update_metadata(gm, replacement_op, meta_map)

    return gm
