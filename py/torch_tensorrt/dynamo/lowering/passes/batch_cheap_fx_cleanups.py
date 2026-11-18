"""Batch cheap, non-conflicting FX cleanups into one graph repair cycle.

Several post-lowering passes only delete/rewrite a few node kinds, then each
calls ``clean_up_graph_after_modifications`` (DCE + lint + recompile). On large
graphs that recompile dominates and is paid repeatedly.

Naren's guidance: use one iteration / one cleanup for repairs that do not
conflict. This pass runs those mutations back-to-back and cleans up once.
"""

from __future__ import annotations

import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.eliminate_sym_min_int64_max import (
    apply_eliminate_sym_min_int64_max,
)
from torch_tensorrt.dynamo.lowering.passes.normalize_negative_slice_stop import (
    apply_normalize_negative_slice_stop,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)
from torch_tensorrt.dynamo.lowering.passes.remove_assert_nodes import (
    apply_remove_assert_nodes,
)
from torch_tensorrt.dynamo.lowering.passes.remove_num_users_is_0_nodes import (
    apply_remove_num_users_is_0_nodes,
)

logger = logging.getLogger(__name__)


def batch_cheap_fx_cleanups(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Apply cheap FX cleanups with a single trailing graph cleanup."""
    del settings  # unused; kept for DynamoPassManager signature
    modified = False
    modified |= apply_remove_assert_nodes(gm)
    # Dead-user removal after assert erasure so newly orphaned nodes go away.
    modified |= apply_remove_num_users_is_0_nodes(gm)
    modified |= apply_eliminate_sym_min_int64_max(gm)
    modified |= apply_normalize_negative_slice_stop(gm)

    if modified:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug("Graph after batch_cheap_fx_cleanups:\n%s", gm.graph)

    return gm
