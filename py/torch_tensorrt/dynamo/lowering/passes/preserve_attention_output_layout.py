# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch_tensorrt.dynamo._settings import CompilationSettings

logger = logging.getLogger(__name__)


def _preserve_layouts(graph: torch.fx.Graph) -> int:
    count = 0
    for node in list(graph.nodes):
        if node.target != torch.ops.aten.scaled_dot_product_attention.default:
            continue
        val = node.meta.get("val")
        # Under torch.vmap the recorded value is a batched tensor, not a FakeTensor,
        # and those nodes lower without this pass.
        if not isinstance(val, FakeTensor) or val.is_contiguous():
            continue
        dim_order = val.dim_order()
        if dim_order == tuple(range(val.dim())):
            continue

        inverse = [dim_order.index(dim) for dim in range(val.dim())]
        with graph.inserting_before(node):
            attention = graph.call_function(node.target, node.args, node.kwargs)
            permuted = graph.call_function(
                torch.ops.aten.permute.default, (attention, list(dim_order))
            )
            copied = graph.call_function(
                torch.ops.aten.clone.default,
                (permuted,),
                {"memory_format": torch.contiguous_format},
            )
            restored = graph.call_function(
                torch.ops.aten.permute.default, (copied, inverse)
            )
        for new_node in (attention, permuted, copied, restored):
            new_node.meta.update(
                {key: value for key, value in node.meta.items() if key != "tensor_meta"}
            )
        # Recorded as the decomposition returns it, so a second run leaves it alone.
        attention.meta["val"] = val.contiguous()
        permuted.meta["val"] = attention.meta["val"].permute(dim_order)
        copied.meta["val"] = permuted.meta["val"].contiguous()
        restored.meta["val"] = copied.meta["val"].permute(inverse)
        # The original node becomes a copy that keeps the recorded layout, so every
        # user, graph outputs and in-place updates included, keeps reading the same
        # node. A copy rather than a view, so in-place ops such as detach_() work.
        node.target = torch.ops.aten.clone.default
        node.args = (restored,)
        node.kwargs = {}
        count += 1
    return count


def preserve_attention_output_layout(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Keep the exported memory layout of attention outputs that will be decomposed.

    The fused attention kernels return their output in a layout that depends on the
    backend and the query, and export drops a ``.contiguous()`` that is a no-op on it.
    The attention decomposition returns a contiguous tensor instead, so a later
    ``view`` fails when the graph is retraced. Copying each output into the layout
    recorded at export keeps those views valid.
    """
    if not settings.decompose_attention:
        return gm

    count = 0
    # Submodules too: attention in torch.cond branches and in torch.no_grad() or
    # autocast regions lives in subgraphs, nested ones included.
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        changed = _preserve_layouts(module.graph)
        if changed:
            # No dead code elimination: the graph is not functionalized yet, and it
            # would drop nodes with no users such as the vmap nesting calls.
            module.recompile()
            count += changed

    if count:
        logger.debug("Preserved the output layout of %d attention nodes", count)
    return gm
