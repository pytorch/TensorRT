# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import operator

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def apply_remove_num_users_is_0_nodes(gm: torch.fx.GraphModule) -> bool:
    """Remove unused ops in-place. Returns True if the graph changed."""
    from torch_tensorrt.dynamo.conversion.plugins._auto_functionalized_converter import (
        _WRAPPER_TARGETS,
        _is_matching_writeback,
    )

    nodes = list(gm.graph.nodes)
    if not nodes:
        return False

    output_node = nodes[-1]
    erased = 0
    for node in nodes[::-1]:
        if (
            node != output_node
            and len(node.users) == 0
            and len(node.all_input_nodes) > 0
            # Functionalized custom ops write back to their inputs even when
            # the model returns None. Other unused writes (e.g. to buffers
            # baked into a standalone engine) retain the existing removal.
            and not _is_matching_writeback(node)
            # The write-back of an in-place update of a user input (e.g. a cache).
            # A functionalized plugin's write-back is decided above: the plugin
            # already wrote its input in place.
            and not (
                node.target == torch.ops.aten.copy_.default
                and node.args[0].op == "placeholder"
                and not (
                    node.args[1].target is operator.getitem
                    and node.args[1].args[0].target in _WRAPPER_TARGETS
                )
            )
        ):
            gm.graph.erase_node(node)
            erased += 1

    if erased:
        logger.debug("Removed %d num_users=0 nodes", erased)
    return erased > 0


def remove_num_users_is_0_nodes(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Remove ops that [num_users=0] in the graph"""
    del settings
    if apply_remove_num_users_is_0_nodes(gm):
        gm = clean_up_graph_after_modifications(gm)
        logger.debug("Graph after remove_num_users_is_0_nodes:\n%s", gm.graph)
    return gm
