# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Iterable

import torch
from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor
from torch.fx.experimental.proxy_tensor import get_proxy_mode, get_proxy_slot

from ._core import register_constant_fold_exclusion_rule

SCATTER_ADD_POSITIONS_RULE_ID = "scatter_add_positions"
_POSITIONS_TAG = "_torch_tensorrt_scatter_add_positions"


def exclude_scatter_add_positions_from_constant_fold(positions: torch.Tensor) -> None:
    """Tag the positions arange of the scatter_add decomposition.

    With a constant index, folding it would store the whole comparison of every
    index with every position as an engine constant. Retraces keep
    ``node.meta["custom"]``, so the rule below finds the tag after the
    decomposed op is gone, for example after the complex decomposition retrace
    or a second ``run_decompositions``.
    """
    proxy_mode = get_proxy_mode()
    if proxy_mode is None:
        return

    tracked = get_proxy_slot(
        mb_unwrap_functional_tensor(positions), proxy_mode.tracer, default=None
    )
    node = getattr(getattr(tracked, "proxy", tracked), "node", None)
    if isinstance(node, torch.fx.Node):
        node.meta.setdefault("custom", {})[_POSITIONS_TAG] = True


def _varies_at_runtime(node: torch.fx.Node) -> bool:
    """Whether constant folding cannot compute this node ahead of time."""
    nodes_to_visit, visited = [node], set()
    while nodes_to_visit:
        current = nodes_to_visit.pop()
        target_tags = getattr(current.target, "tags", ())
        if (
            current.op == "placeholder"
            or torch.Tag.nondeterministic_seeded in target_tags
        ):
            return True
        if current not in visited:
            visited.add(current)
            nodes_to_visit.extend(current.all_input_nodes)
    return False


@register_constant_fold_exclusion_rule(SCATTER_ADD_POSITIONS_RULE_ID)
def _scatter_add_positions_rule(node: torch.fx.Node) -> Iterable[torch.fx.Node]:
    """Select the tagged positions arange when src varies at runtime.

    The decomposition multiplies ``(index == positions.view(-1, 1)).to(dtype)``
    by src. With constant src, a constant index makes the whole sum constant,
    and it folds to a result the size of the output, so the positions stay
    foldable. A runtime index keeps the comparison live either way.
    """
    if not node.meta.get("custom", {}).get(_POSITIONS_TAG):
        return ()
    # Several scatter_adds can share the positions, for example after CSE.
    frontier, comparison, sources = [node], {node}, []
    for _ in range(4):  # view, eq, cast, mul
        frontier = [user for current in frontier for user in current.users]
        for user in frontier:
            if user.target == torch.ops.aten.mul.Tensor:
                sources += [
                    arg for arg in user.all_input_nodes if arg not in comparison
                ]
        comparison.update(frontier)
    if sources and not any(map(_varies_at_runtime, sources)):
        return ()
    return (node,)
