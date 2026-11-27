# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import operator
from typing import Any, Dict, List, Optional

import torch
from torch.fx.node import Argument, Node
from torch_tensorrt.dynamo._meta_keys import NUM_SORT_KEYS

logger = logging.getLogger(__name__)

# Ops whose output holds only values of their input.
_VALUE_PRESERVING_OPS = {
    torch.ops.aten.alias.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.permute.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.select.int,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.dims,
    torch.ops.aten.t.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.view.default,
    torch.ops.aten._reshape_copy.default,
    torch.ops.aten._unsafe_view.default,
}


def key_chain(node: Argument) -> Optional[List[Node]]:
    """The nodes from ``node`` back to the topk, argmax or argmin whose positions it holds.

    Positions along one dim of a tensor come from the indices that topk, argmax and argmin
    return, and they stay in ``[0, size)`` through ops that only move values or cast them to
    int32 or int64. The last node is the topk, argmax or argmin. None if ``node`` holds
    anything else.
    """
    chain = []
    while isinstance(node, Node):
        chain.append(node)
        if node.target in _VALUE_PRESERVING_OPS or (
            node.target == torch.ops.aten._to_copy.default
            and node.kwargs.get("dtype") in (torch.int32, torch.int64)
        ):
            node = node.args[0]
        elif node.target in (
            torch.ops.aten.argmax.default,
            torch.ops.aten.argmin.default,
        ):
            return chain
        elif (
            node.target is operator.getitem
            and node.args[1] == 1
            and isinstance(node.args[0], Node)
            and node.args[0].target == torch.ops.aten.topk.default
        ):
            return chain + [node.args[0]]
        else:
            return None
    return None


def _num_index_values(node: Argument) -> Optional[int]:
    """If ``node`` holds positions along one dim of a tensor, the size of that dim."""
    chain = key_chain(node)
    if chain is None:
        return None
    source = chain[-1]
    if source.target == torch.ops.aten.topk.default:
        dim = source.args[2] if len(source.args) > 2 else source.kwargs.get("dim", -1)
    else:
        dim = source.args[1] if len(source.args) > 1 else source.kwargs.get("dim")

    if not isinstance(source.args[0], Node):
        return None
    data = source.args[0].meta.get("val")
    if not isinstance(data, torch.Tensor):
        return None
    if dim is None:
        size = data.numel()
    elif data.dim() == 0:
        size = 1
    else:
        size = data.shape[dim % data.dim()]
    return size if isinstance(size, int) else None


def _attribute(gm: torch.fx.GraphModule, node: Node) -> Any:
    value: Any = gm
    for atom in str(node.target).split("."):
        value = getattr(value, atom, None)
    return value


def attribute_shapes(gm: torch.fx.GraphModule) -> Optional[Dict[Node, torch.Size]]:
    """The shape of each tensor that a get_attr node reads, or None if there is no stable sort."""
    if not gm.graph.find_nodes(op="call_function", target=torch.ops.aten.sort.stable):
        return None
    shapes = {}
    for node in gm.graph.find_nodes(op="get_attr"):
        value = _attribute(gm, node)
        if isinstance(value, torch.Tensor):
            shapes[node] = value.shape
    return shapes


def _attributes_unchanged(
    gm: torch.fx.GraphModule, shapes_before: Dict[Node, torch.Size]
) -> bool:
    """Whether each tensor that a get_attr node reads has the shape in its meta and, if the node
    was there before the lowering passes, the shape it had then."""
    for node in gm.graph.find_nodes(op="get_attr"):
        value = _attribute(gm, node)
        if not isinstance(value, torch.Tensor):
            continue
        meta = node.meta.get("val")
        if (
            not isinstance(meta, torch.Tensor)
            or meta.shape != value.shape
            or shapes_before.get(node, value.shape) != value.shape
        ):
            return False
    return True


def annotate_sort_num_keys(
    gm: torch.fx.GraphModule,
    shapes_updated: bool,
    shapes_before: Optional[Dict[Node, torch.Size]],
) -> None:
    """Record on each stable sort how many key values its input can hold, when that is known.

    The sort converter counts keys, so it needs that number. It is found here, once the
    lowering passes are done, from the shapes in node meta, so ``shapes_updated`` must say
    whether the refresh after the passes updated those shapes, and ``shapes_before`` must be
    what attribute_shapes returned before the passes. Node meta goes with the node into the
    TensorRT subgraph, so the validator and the converter read it from
    ``node.meta[NUM_SORT_KEYS]``.
    """
    sorts = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.sort.stable
    ]
    if not sorts:
        return
    # The refresh follows edits to nodes only, so a pass that replaces a weight in place leaves
    # the old shapes in the meta of every node that reads it, even if it updates the meta of
    # the get_attr node itself.
    if shapes_updated and (
        shapes_before is None or not _attributes_unchanged(gm, shapes_before)
    ):
        logger.debug("A lowering pass may have changed the shape of a module attribute")
        shapes_updated = False
    for node in sorts:
        num_keys = _num_index_values(node.args[0]) if shapes_updated else None
        if num_keys is None:
            # The converter trusts this key, so one left by an earlier lowering must go.
            node.meta.pop(NUM_SORT_KEYS, None)
            logger.debug(
                f"No range recorded for the keys of {node.name}: "
                + (
                    "they are not positions along a dim of known size"
                    if shapes_updated
                    else "the shapes in node meta may be out of date after lowering"
                )
            )
        else:
            node.meta[NUM_SORT_KEYS] = num_keys
            logger.debug(f"Keys of {node.name} lie in [0, {num_keys})")
