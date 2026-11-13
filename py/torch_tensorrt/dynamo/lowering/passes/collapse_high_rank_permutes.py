# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import math
from typing import List, Optional, Tuple, cast

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
    of at most 8 dimensions that compute the same values (see ``_steps``). With
    ``collapse_high_rank_expands``, an unsqueeze, expand, reshape chain past 8 dims
    whose expand grows only the new dimension, as in newer Qwen2-VL image patches, is
    rewritten on 3 dims (see ``_rewrite_repeat``). All chains of one kind are left as
    they are when
    ``torch_executed_ops`` lists one of their ops (the permute; the unsqueeze or the
    expand; a clone; a reshape). So is a chain in a module that
    ``torch_executed_modules`` keeps in PyTorch.
    """
    permutes = (
        _chains(gm, settings, aten.permute.default)
        if settings.collapse_high_rank_permutes
        else []
    )
    repeats = 0
    if settings.collapse_high_rank_expands:
        # Repeats first, and the last one first: each rewrite can leave a reshape as
        # the only reader of the expand or permute before it.
        repeats = sum(
            _rewrite_repeat(node, permutes)
            for node in reversed(
                _chains(gm, settings, aten.unsqueeze.default, aten.expand.default)
            )
        )
    rewritten = sum(_rewrite(gm, node) for node in permutes)
    if rewritten or repeats:
        gm = clean_up_graph_after_modifications(gm)
    if rewritten:
        logger.debug(f"Rewrote {rewritten} permute(s) of more than 8 dims")
    if repeats:
        logger.debug(f"Rewrote {repeats} expand(s) of more than 8 dims")
    return gm


def _chains(
    gm: torch.fx.GraphModule,
    settings: CompilationSettings,
    *ops: torch._ops.OpOverload,
) -> List[torch.fx.Node]:
    """The ``ops[0]`` nodes over 8 dims whose chains may run in TensorRT.

    ``ops`` are the ops of the chain besides clones and reshapes. None may be in
    ``torch_executed_ops``, even one the rewrite removes, since the partitioner could
    then no longer keep it in PyTorch.
    """
    excluded = settings.torch_executed_ops
    if any(
        op in excluded or ConverterRegistry.qualified_name_or_str(op) in excluded
        for op in (*ops, aten.clone.default, *_RESHAPES)
    ):
        return []
    return [
        node
        for node in gm.graph.nodes
        if node.target == ops[0]
        and _rank(node) > trt.Dims.MAX_DIMS
        and not node_in_torch_executed_module(node, settings.torch_executed_modules)
    ]


def _rank(node: torch.fx.Node) -> int:
    val = node.meta.get("val")
    return val.ndim if isinstance(val, torch.Tensor) else 0


def _static_sizes(node: torch.fx.Node) -> Optional[List[int]]:
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor) and all(isinstance(s, int) for s in val.shape):
        return list(val.shape)
    return None


def _reshaped_clones(node: torch.fx.Node) -> Optional[List[torch.fx.Node]]:
    """The clones of ``node`` if only reshapes read ``node`` and them, else None."""
    clones = [user for user in node.users if user.target == aten.clone.default]
    for n in [node, *clones]:
        if any(u not in clones and u.target not in _RESHAPES for u in n.users):
            return None
    return clones


def _readers(node: torch.fx.Node) -> List[torch.fx.Node]:
    """The readers of ``node`` and its clones, once ``_reshaped_clones`` accepted it."""
    clones = cast(List[torch.fx.Node], _reshaped_clones(node))
    return [u for n in [node, *clones] for u in n.users if u not in clones]


def _set_vals(nodes: List[torch.fx.Node], source: torch.Tensor) -> None:
    """Recompute the values of ``nodes``.

    The first is computed from ``source``, and each other one from its own input.
    """
    for i, node in enumerate(nodes):
        op = cast(torch._ops.OpOverload, node.target)
        value = source if i == 0 else node.all_input_nodes[0].meta["val"]
        node.meta["val"] = op(value, *node.args[1:], **node.kwargs)
        # "tensor_meta" would keep the old shapes.
        node.meta.pop("tensor_meta", None)


def _collapsible(permute: torch.fx.Node) -> bool:
    """Whether ``_rewrite`` can run the chain around ``permute`` on at most 8 dims."""
    head = permute.all_input_nodes[0]
    return (
        head.target in _RESHAPES
        and len(head.users) == 1
        and _static_sizes(head) is not None
        and _reshaped_clones(permute) is not None
    )


def _rewrite(gm: torch.fx.GraphModule, permute: torch.fx.Node) -> bool:
    if not _collapsible(permute):
        return False
    head = permute.all_input_nodes[0]
    sizes, clones = cast(List[int], _static_sizes(head)), _reshaped_clones(permute)
    val = head.meta["val"]

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
    # from its own old value.
    _set_vals([head, *added, *clones], val)
    return True


def _rewrite_repeat(unsqueeze: torch.fx.Node, permutes: List[torch.fx.Node]) -> bool:
    """Run ``unsqueeze(d)``, an expand of dim d alone, and reshapes on 3 dims.

    In row-major order the result repeats each block of the dims from d on, so
    reshaping the input to (A, 1, B) and expanding that to (A, k, B) gives the same
    values, where A is the size of the dims before d and B the size of the rest.
    Each reshape that reads the result must come back to at most 8 dims, or start a
    chain in ``permutes`` that ``_rewrite`` collapses and whose readers come back to at
    most 8 dims. Otherwise the graph stays split and the engine would output the
    expanded tensor instead of the input.
    """
    if len(unsqueeze.users) != 1:
        return False
    expand = next(iter(unsqueeze.users))
    if expand.target != aten.expand.default:
        return False
    # The unsqueeze's own value, since its input may get one only after all passes.
    shape, out = _static_sizes(unsqueeze), _static_sizes(expand)
    clones = _reshaped_clones(expand)
    if shape is None or out is None or clones is None:
        return False
    for reader in _readers(expand):
        permute = next(iter(reader.users), None)
        if _rank(reader) > trt.Dims.MAX_DIMS and not (
            permute in permutes
            and _collapsible(cast(torch.fx.Node, permute))
            and all(_rank(r) <= trt.Dims.MAX_DIMS for r in _readers(permute))
        ):
            return False
    dim = cast(int, unsqueeze.args[1]) % len(shape)
    sizes = shape[:dim] + shape[dim + 1 :]
    if out[:dim] + out[dim + 1 :] != sizes:
        return False

    outer, inner = math.prod(sizes[:dim]), math.prod(sizes[dim:])
    unsqueeze.target = aten._reshape_copy.default
    unsqueeze.args = (unsqueeze.args[0], [outer, 1, inner])
    expand.args = (unsqueeze, [outer, out[dim], inner])
    _set_vals([unsqueeze, expand, *clones], unsqueeze.meta["val"])
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
