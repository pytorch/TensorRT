# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import operator

import torch
from torch._functorch.predispatch import (
    _vmap_decrement_nesting,
    _vmap_increment_nesting,
)
from torch._prims_common import is_contiguous_or_false
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch_tensorrt.dynamo._settings import CompilationSettings

logger = logging.getLogger(__name__)

_RESHAPES = {
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
}


def _merges_new_axis(stack: torch.fx.Node, dim: int) -> bool:
    """Whether the only user of ``stack`` merges axis ``dim`` into a neighbor axis."""
    user = next(iter(stack.users))
    if user.target not in _RESHAPES:
        return False
    out, new = stack.meta["val"].shape, user.meta["val"].shape
    if len(new) != len(out) - 1:
        return False
    for a in (dim - 1, dim):
        if 0 <= a < len(out) - 1:
            merged = (*out[:a], out[a] * out[a + 1], *out[a + 2 :])
            if all(statically_known_true(p == q) for p, q in zip(merged, new)):
                return True
    return False


def _rewrite(module: torch.fx.GraphModule, in_vmap: bool = False) -> int:
    """Rewrite the stacks of a graph and of the subgraphs it calls, such as torch.cond
    branches and torch.no_grad() or autocast regions."""
    graph = module.graph
    count = subgraph_count = 0
    # torch.vmap leaves its body in the graph between these calls, and that includes
    # the subgraphs called there. The values there are batched, and their unsqueeze
    # fails outside vmap.
    vmap_depth = int(in_vmap)
    for node in list(graph.nodes):
        if node.target is _vmap_increment_nesting:
            vmap_depth += 1
        elif node.target is _vmap_decrement_nesting:
            vmap_depth -= 1
        elif node.op == "get_attr":
            subgraph = operator.attrgetter(node.target)(module)
            if isinstance(subgraph, torch.fx.GraphModule):
                subgraph_count += _rewrite(subgraph, vmap_depth > 0)
        if (
            vmap_depth
            or node.target != torch.ops.aten.stack.default
            or len(node.users) != 1
        ):
            continue
        val = node.meta["val"]
        tensors = node.args[0]
        # For channels-last or other non-contiguous inputs the concat gives a layout
        # other than the stack's, and a view after it can then fail.
        if not all(is_contiguous_or_false(t.meta["val"]) for t in tensors):
            continue
        dim = (
            node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        ) % val.dim()
        if not _merges_new_axis(node, dim):
            continue

        with graph.inserting_before(node):
            unsqueezed = [
                graph.call_function(torch.ops.aten.unsqueeze.default, (t, dim))
                for t in tensors
            ]
            cat = graph.call_function(torch.ops.aten.cat.default, (unsqueezed, dim))
        meta = {key: value for key, value in node.meta.items() if key != "tensor_meta"}
        for new_node, t in zip(unsqueezed, tensors):
            new_node.meta.update(meta)
            new_node.meta["val"] = t.meta["val"].unsqueeze(dim)
        cat.meta.update(meta)
        node.replace_all_uses_with(cat)
        graph.erase_node(node)
        count += 1
    if count:
        module.recompile()
    return count + subgraph_count


def replace_interleave_stack_with_cat(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Rewrite a stack that feeds an interleaving reshape as a concat of unsqueezed inputs.

    When TensorRT fuses the region around ``torch.stack([a, b], 2).flatten(1, 2)``, it runs
    PyTorch's ``cat`` + ``view`` form of the stack as an extra copy and fuses the unsqueeze
    + ``cat`` form. Where it runs the region as separate layers instead, the unsqueeze form
    adds copies and can be slower, so the pass runs only with
    ``replace_interleave_stack_with_cat``.
    """
    if not settings.replace_interleave_stack_with_cat:
        return gm
    count = _rewrite(gm)
    if count:
        logger.debug("Replaced %d interleaving stack nodes with cat", count)
    return gm
