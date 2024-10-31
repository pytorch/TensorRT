import logging
from typing import List

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)
from torch_tensorrt.dynamo.utils import get_metadata, set_metadata

logger = logging.getLogger(__name__)


def view_to_reshape(
    gm: torch.fx.GraphModule, settings: CompilationSettings
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

    # Store metadata of the orig_op
    metadata = get_metadata(gm, orig_op)

    if torch.fx.subgraph_rewriter.replace_pattern(gm, orig, replacement):
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Graph after replacing view with reshape:\n{gm.graph}")

    # Copy the orig_op's metadata to the replacement op
    set_metadata(gm, replacement_op, metadata)

    return gm
