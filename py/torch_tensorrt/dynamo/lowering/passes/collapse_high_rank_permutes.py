# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import math
from typing import List, Tuple, cast

import tensorrt as trt
import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterRegistry
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)
from torch_tensorrt.dynamo.partitioning.common import node_in_torch_executed_module

logger = logging.getLogger(__name__)

aten = torch.ops.aten

# Reshapes that accept a tensor of any layout, so they stay valid on the new shapes.
_RESHAPES = (aten._reshape_copy.default, aten.reshape.default)

Step = Tuple[List[int], List[int]]


def collapse_high_rank_permutes(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Run a reshape, permute, reshape chain over more than 8 dims on at most 8.

    Window and patch layouts (neighborhood attention, the Qwen2-VL image patches)
    reshape a tensor to more than 8 dimensions, permute it, and reshape it back.
    TensorRT tensors have at most 8 dimensions, so each such chain ran in PyTorch and
    split the graph. A chain with static shapes is rewritten as reshapes and permutes
    of at most 8 dimensions that compute the same values (see ``_steps``). Chains that
    ``torch_executed_ops`` or ``torch_executed_modules`` keep in PyTorch are left as
    they are, since there the rewrite would only add work.
    """
    if not settings.collapse_high_rank_permutes:
        return gm
    excluded = settings.torch_executed_ops
    if any(
        op in excluded or ConverterRegistry.qualified_name_or_str(op) in excluded
        for op in (aten.permute.default, aten.clone.default, *_RESHAPES)
    ):
        return gm
    count = 0
    for node in list(gm.graph.nodes):
        if (
            node.target == aten.permute.default
            and _rank(node) > trt.Dims.MAX_DIMS
            and not node_in_torch_executed_module(node, settings.torch_executed_modules)
        ):
            count += _rewrite(gm, node)
    if count:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Rewrote {count} permute(s) of more than 8 dims")
    return gm


def _rank(node: torch.fx.Node) -> int:
    val = node.meta.get("val")
    return val.ndim if isinstance(val, torch.Tensor) else 0


def _rewrite(gm: torch.fx.GraphModule, permute: torch.fx.Node) -> bool:
    head = permute.all_input_nodes[0]
    if head.target not in _RESHAPES or len(head.users) != 1:
        return False
    val = head.meta.get("val")
    if not isinstance(val, torch.Tensor) or not all(
        isinstance(size, int) for size in val.shape
    ):
        return False
    sizes = list(val.shape)
    clones = [user for user in permute.users if user.target == aten.clone.default]
    for node in [permute, *clones]:
        if any(u not in clones and u.target not in _RESHAPES for u in node.users):
            return False

    dims = [d % len(sizes) for d in cast(List[int], permute.args[1])]
    (shape, dims), *rest = _steps(sizes, dims)
    head.args = (head.args[0], shape)
    permute.args = (head, dims)
    added = [permute]
    for shape, dims in rest:
        with gm.graph.inserting_after(added[-1]):
            reshape = gm.graph.call_function(
                aten._reshape_copy.default, (added[-1], shape)
            )
        with gm.graph.inserting_after(reshape):
            step = gm.graph.call_function(aten.permute.default, (reshape, dims))
        # Keep nn_module_stack, which engine layer names read.
        reshape.meta.update(permute.meta)
        step.meta.update(permute.meta)
        added += [reshape, step]
    if rest:
        permute.replace_all_uses_with(
            added[-1], delete_user_cb=lambda user: user is not added[1]
        )
    # The head's input may get its value only after all passes, so the head starts
    # from its own old value. "tensor_meta" would keep the old shapes.
    for node in [head, *added, *clones]:
        op = cast(torch._ops.OpOverload, node.target)
        source = val if node is head else node.all_input_nodes[0].meta["val"]
        node.meta["val"] = op(source, *node.args[1:], **node.kwargs)
        node.meta.pop("tensor_meta", None)
    return True


def _steps(sizes: List[int], dims: List[int]) -> List[Step]:
    """Shapes and permutes of at most 8 dims that together permute ``sizes`` by ``dims``.

    Each step reshapes the result of the one before to its shape, then permutes it.
    Dropping dims of size 1, and merging dims that stay next to each other and in
    order through the permute, changes no value and is often enough. When it is not,
    dims 0 and 1 first move together, and then dim 1 alone moves to its place, which
    takes 4 dims.
    """
    sizes, dims = _merge(sizes, dims)
    if len(sizes) <= trt.Dims.MAX_DIMS:
        return [(sizes, dims)]
    together = _steps(
        [sizes[0] * sizes[1], *sizes[2:]],
        [d - 1 if d > 1 else d for d in dims if d != 1],
    )
    first, second = dims.index(0), dims.index(1)
    out = [sizes[d] for d in dims]
    if first < second:
        blocks = [
            out[: first + 1],
            [sizes[1]],
            out[first + 1 : second],
            out[second + 1 :],
        ]
    else:
        blocks = [
            out[:second],
            out[second + 1 : first + 1],
            [sizes[1]],
            out[first + 1 :],
        ]
    return together + [_merge([math.prod(b) for b in blocks], [0, 2, 1, 3])]


def _merge(sizes: List[int], dims: List[int]) -> Step:
    """Drop dims of size 1 and merge dims that stay next to each other through ``dims``."""
    out = [d for d in dims if sizes[d] != 1]
    position = {d: i for i, d in enumerate(out)}
    groups: List[List[int]] = []
    for d in sorted(out):
        if groups and position[d] == position[groups[-1][-1]] + 1:
            groups[-1].append(d)
        else:
            groups.append([d])
    index = {group[0]: i for i, group in enumerate(groups)}
    return (
        [math.prod(sizes[d] for d in group) for group in groups],
        [index[d] for d in out if d in index],
    )
