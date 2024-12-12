import logging
from typing import List

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)
from torch_tensorrt.dynamo.utils import copy_metadata

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

    match_and_replacements = torch.fx.subgraph_rewriter._replace_pattern(
        gm, orig, replacement
    )
    if match_and_replacements:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Graph after replacing view with reshape:\n{gm.graph}")

    # Copy the orig_op's metadata to the replacement op
    copy_metadata(match_and_replacements)

    return gm
