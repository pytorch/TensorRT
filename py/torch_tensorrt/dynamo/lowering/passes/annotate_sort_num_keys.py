# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import operator
from typing import Optional

import torch
from torch.fx.node import Argument, Node
from torch_tensorrt.dynamo._meta_keys import NUM_SORT_KEYS
from torch_tensorrt.dynamo._settings import CompilationSettings

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


def _num_index_values(node: Argument) -> Optional[int]:
    """If ``node`` holds positions along one dim of a tensor, the size of that dim.

    Such positions come from the indices that topk, argmax and argmin return, and they
    stay in ``[0, size)`` through ops that only move values or cast them to int32 or int64.
    """
    while isinstance(node, Node):
        if node.target in _VALUE_PRESERVING_OPS:
            node = node.args[0]
        elif node.target == torch.ops.aten._to_copy.default and node.kwargs.get(
            "dtype"
        ) in (torch.int32, torch.int64):
            node = node.args[0]
        else:
            break
    if not isinstance(node, Node):
        return None

    if node.target is operator.getitem and node.args[1] == 1:
        source = node.args[0]
        if source.target == torch.ops.aten.topk.default:
            dim = (
                source.args[2] if len(source.args) > 2 else source.kwargs.get("dim", -1)
            )
        else:
            return None
    elif node.target in (torch.ops.aten.argmax.default, torch.ops.aten.argmin.default):
        source = node
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    else:
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


def annotate_sort_num_keys(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Record on each stable sort how many key values its input can hold, when that is known.

    The sort converter counts keys, so it needs that number. It is found here, on the whole
    graph, because partitioning can leave the producer of the keys in PyTorch, and the
    TensorRT subgraph then sees only a placeholder. Node meta goes with the node into that
    subgraph, so the validator and the converter read it from ``node.meta[NUM_SORT_KEYS]``.
    """
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.sort.stable:
            continue
        num_keys = _num_index_values(node.args[0])
        if num_keys is None:
            # The converter trusts this key, so one left by an earlier lowering must go.
            node.meta.pop(NUM_SORT_KEYS, None)
        else:
            node.meta[NUM_SORT_KEYS] = num_keys
            logger.debug(f"Keys of {node.name} lie in [0, {num_keys})")
    return gm
