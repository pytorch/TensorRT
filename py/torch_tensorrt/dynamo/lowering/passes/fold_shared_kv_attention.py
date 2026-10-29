# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Run attention against unrepeated keys and values when query heads share them.

With multi-query attention, every query head reads the same key/value head.
Hugging Face models still repeat that head once per query head before the two
attention matmuls (``repeat_kv``: unsqueeze, expand, reshape). TensorRT then
copies the repeated tensors and runs its fused attention kernel, which on some
GPUs is slow at large head sizes.

With ``fold_shared_kv_attention``, this pass folds the query heads into the
rows of one matrix product against the unrepeated keys, and another against
the unrepeated values. The math is the same, but TensorRT then runs plain
matrix products instead of its fused kernel, so float16 results round
differently. ``use_fp32_acc`` accumulates each matrix product in float32.

Only matmuls with static shapes are rewritten. ``scaled_dot_product_attention``
becomes matmuls only with ``decompose_attention``. Under ``torch.compile``,
``use_distributed_mode_trace`` lowers matmuls to ``bmm`` before this pass, and
``torch.compile`` turns it on by itself for multi-process ``torch.distributed``
jobs and for graphs that use vmap.
"""

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)

aten = torch.ops.aten

_RESHAPES = (
    aten._reshape_copy.default,
    aten.reshape.default,
    aten.view.default,
    aten._unsafe_view.default,
)


def _static_shape(node: torch.fx.Node) -> Optional[Tuple[int, ...]]:
    val = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    if not isinstance(val, torch.Tensor) or not all(
        isinstance(size, int) for size in val.shape
    ):
        return None
    return tuple(val.shape)


def _is_call(node: Any, target: torch._ops.OpOverload) -> bool:
    return isinstance(node, torch.fx.Node) and node.target == target


def _shared_kv(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The (batch, 1, seq, dim) source if ``node`` repeats it once per query head, else None."""
    if not isinstance(node, torch.fx.Node) or node.target not in _RESHAPES:
        return None
    expand = node.args[0]
    # repeat_interleave, used by the enable_gqa attention decomposition, adds this clone.
    if _is_call(expand, aten.clone.default):
        expand = expand.args[0]
    if not _is_call(expand, aten.expand.default):
        return None
    unsqueeze = expand.args[0]
    if not _is_call(unsqueeze, aten.unsqueeze.default) or unsqueeze.args[1] != 2:
        return None
    source = unsqueeze.args[0]
    source_shape = _static_shape(source)
    expanded_shape = _static_shape(expand)
    repeated_shape = _static_shape(node)
    if source_shape is None or expanded_shape is None or repeated_shape is None:
        return None
    if len(source_shape) != 4 or source_shape[1] != 1:
        return None
    batch, _, seq, dim = source_shape
    # expand may also broadcast a size-1 seq or dim, and then the copies are not one per head.
    if expanded_shape[3:] != (seq, dim):
        return None
    if repeated_shape[0] != batch or repeated_shape[2:] != (seq, dim):
        return None
    return source


def _add(
    gm: torch.fx.GraphModule,
    target: torch._ops.OpOverload,
    args: Sequence[Any],
    meta: Dict[str, Any],
    kwargs: Optional[Dict[str, Any]] = None,
) -> torch.fx.Node:
    node = gm.graph.call_function(target, args=tuple(args), kwargs=kwargs)
    # Keep nn_module_stack, which torch_executed_modules and engine layer names read.
    node.meta.update({key: val for key, val in meta.items() if key != "tensor_meta"})
    node.meta["val"] = target(
        *(arg.meta["val"] if isinstance(arg, torch.fx.Node) else arg for arg in args),
        **(kwargs or {}),
    )
    return node


def fold_shared_kv_attention(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Rewrite matmuls against a key/value head repeated per query head into matmuls against the head itself."""
    if not settings.fold_shared_kv_attention:
        return gm
    folded = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target != aten.matmul.default:
            continue
        lhs, rhs = node.args
        transpose = _is_call(rhs, aten.permute.default) and list(rhs.args[1]) == [
            0,
            1,
            3,
            2,
        ]
        operand = rhs.args[0] if transpose else rhs
        # use_fp32_acc with decompose_attention casts the repeated keys and values.
        cast = operand if _is_call(operand, aten._to_copy.default) else None
        source = _shared_kv(cast.args[0] if cast is not None else operand)
        lhs_shape = _static_shape(lhs)
        out_shape = _static_shape(node)
        if source is None or lhs_shape is None or out_shape is None:
            continue
        if len(lhs_shape) != 4 or lhs_shape[:2] != out_shape[:2]:
            continue
        batch, heads, rows, inner = lhs_shape
        with gm.graph.inserting_before(node):
            folded_lhs = _add(
                gm,
                aten.reshape.default,
                (lhs, [batch, 1, heads * rows, inner]),
                node.meta,
            )
            kv = source
            if cast is not None:
                kv = _add(gm, aten._to_copy.default, (kv,), node.meta, cast.kwargs)
            if transpose:
                kv = _add(gm, aten.permute.default, (kv, [0, 1, 3, 2]), node.meta)
            product = _add(gm, aten.matmul.default, (folded_lhs, kv), node.meta)
            replacement = _add(
                gm, aten.reshape.default, (product, list(out_shape)), node.meta
            )
        node.replace_all_uses_with(replacement)
        gm.graph.erase_node(node)
        folded += 1

    if folded:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Folded {folded} matmul(s) against a shared key/value head")
    else:
        logger.info(
            "fold_shared_kv_attention is on but found nothing to fold in this graph. "
            "It needs exactly one key/value head, "
            "static shapes, and decompose_attention=True for "
            "scaled_dot_product_attention. It does not apply under torch.compile "
            "with use_distributed_mode_trace, which torch.compile also turns on by "
            "itself for multi-process torch.distributed jobs and for graphs that use "
            "vmap. This is logged once per graph, so a torch.cond branch or a graph "
            "split off by a graph break can log it even when other attention folded."
        )
    return gm
