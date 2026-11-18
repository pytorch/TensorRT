# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Keep batch and heads in separate dims while sequence-first attention transposes them.

``nn.MultiheadAttention`` with ``need_weights=True``, its default, runs attention as
``bmm`` on (batch * heads, seq, head_dim) operands. It makes each operand from a
(seq, batch, embed) tensor with a reshape to (seq, batch * heads, head_dim) and a
transpose of the first two dims. For this form, TensorRT 11.3 and 11.4 and
TensorRT-RTX 1.7 can build a fused attention kernel that gives wrong float16 and
bfloat16 results. Which attention is affected depends on the TensorRT build and on the
shapes. For example, TensorRT 11.4 is wrong for the decoder cross-attention of
LeRobot's ACT policy, where the key has a positional term of its own, and
TensorRT-RTX 1.7 is also wrong for self-attention whose query and key share one
positional term. This pass reshapes each operand to (seq, batch, heads, head_dim)
instead, transposes it to (batch, heads, seq, head_dim), and merges batch and heads
after that. The values are the same, and with more than one head TensorRT builds a
correct engine. When constant folding has turned the query or the value into a
constant, as for the query of ACT's decoder cross-attention, the pass rewrites the
other head splits and leaves the constant as it is.

The pass matches only the form nn.MultiheadAttention writes without a mask. With
attn_mask or key_padding_mask it computes the scores with baddbmm, which the pass leaves
as it is; TensorRT 11.4 built that form correctly in the shapes tested. The same
attention written another way, such as with torch.matmul or with the scores scaled
after the first bmm, is left as it is too, and TensorRT can still build it wrong. The
pass runs only with the TensorRT versions that have the bug, so other versions keep the
original graph.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import tensorrt as trt
import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterRegistry
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)
from torch_tensorrt.dynamo.partitioning.common import node_in_torch_executed_module

logger = logging.getLogger(__name__)

aten = torch.ops.aten

_RESHAPES = (
    aten._reshape_copy.default,
    aten.reshape.default,
    aten.view.default,
    aten._unsafe_view.default,
)


# A head split's source, its sequence and batch * heads sizes, and its heads.
_Split = Tuple[torch.fx.Node, Any, Any, int]


def _is_call(node: Any, target: torch._ops.OpOverload) -> bool:
    return isinstance(node, torch.fx.Node) and node.target == target


def _is_constant(node: Any) -> bool:
    return isinstance(node, torch.fx.Node) and node.op == "get_attr"


def _is_cast(node: Any) -> bool:
    return _is_call(node, aten._to_copy.default) and len(node.users) == 1


def _attention_operands(
    softmax: torch.fx.Node,
) -> Optional[List[Tuple[torch.fx.Node, Any]]]:
    """(user, operand) for the query, key and value of bmm(softmax(bmm(q, k^T)), v),
    also with the casts to float32 and back that torch.autocast adds around the
    softmax."""
    scores = softmax.args[0]
    if _is_cast(scores):
        scores = scores.args[0]
    # TensorRT does not fuse the attention when the attention weights are an output too.
    if (
        softmax.args[1] not in (-1, 2)
        or len(softmax.users) != 1
        or not _is_call(scores, aten.bmm.default)
    ):
        return None
    (weights,) = softmax.users
    if _is_cast(weights):
        (output,) = weights.users
    else:
        weights, output = softmax, weights
    if not _is_call(output, aten.bmm.default) or output.args[0] is not weights:
        return None
    query_user, query = scores, scores.args[0]
    if _is_call(query, aten.mul.Tensor) and type(query.args[1]) in (int, float):
        query_user, query = query, query.args[0]
    key_t = scores.args[1]
    if not _is_call(key_t, aten.permute.default) or list(key_t.args[1]) != [0, 2, 1]:
        return None
    # The rewrite gives the key transpose and the scaled query new inputs. An op outside
    # the attention that also reads one of them, such as as_strided, could then see
    # another memory layout, so such attention is left as it is.
    if len(key_t.users) != 1 or (
        query_user is not scores and len(query_user.users) != 1
    ):
        return None
    return [(query_user, query), (key_t, key_t.args[0]), (output, output.args[1])]


def _head_split(node: Any) -> Optional[_Split]:
    """Match permute(reshape(x, [seq, batch * heads, head_dim]), [1, 0, 2]) of a
    (seq, batch, heads * head_dim) x."""
    if not _is_call(node, aten.permute.default) or list(node.args[1]) != [1, 0, 2]:
        return None
    reshape = node.args[0]
    if not isinstance(reshape, torch.fx.Node) or reshape.target not in _RESHAPES:
        return None
    source = reshape.args[0]
    x = source.meta.get("val") if isinstance(source, torch.fx.Node) else None
    y = reshape.meta.get("val")
    if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
        return None
    if x.dim() != 3 or y.dim() != 3:
        return None
    # The reshape converter reads a 0 in a shape as the size of the input's dim there,
    # which would give the reshapes of the rewrite wrong shapes for an empty sequence.
    # A sequence that a mask selects can be empty at run time.
    if statically_known_true(x.numel() == 0) or not statically_known_true(
        x.shape[0] >= 1
    ):
        return None
    seq, batch, embed = x.shape
    head_dim = y.shape[2]
    if type(embed) is not int or type(head_dim) is not int or embed % head_dim:
        return None
    heads = embed // head_dim
    if not (
        statically_known_true(y.shape[0] == seq)
        and statically_known_true(y.shape[1] == batch * heads)
    ):
        return None
    # The new targets reuse the sizes the graph passes to the reshape, so that
    # dynamic sizes keep working.
    seq_size, batch_heads_size, _ = reshape.args[1]
    if seq_size == -1:
        return None
    return source, seq_size, batch_heads_size, heads


def _needs_split(
    operands: List[Tuple[torch.fx.Node, Any]], splits: List[Optional[_Split]]
) -> bool:
    """Whether to rewrite the attention, see split_sequence_first_attention_heads."""
    key = splits[1]
    if key is None or any(
        split is None and not _is_constant(operand)
        for (_, operand), split in zip(operands, splits)
    ):
        return False
    # TensorRT builds float32 attention correctly.
    if key[0].meta["val"].dtype not in (torch.float16, torch.bfloat16):
        return False
    # When the key is the value, nn.MultiheadAttention selects both from one packed
    # projection. TensorRT builds that attention correctly, and the rewrite only adds
    # copies to it.
    if _is_call(key[0], aten.select.int):
        return False
    # With one head, TensorRT 10 and 11 fail to build the rewritten attention in some
    # cases, such as when the query is a constant, and the rewrite slows down some
    # one-head attention that TensorRT builds correctly.
    return key[3] > 1


def _add(
    gm: torch.fx.GraphModule,
    target: torch._ops.OpOverload,
    args: Tuple[Any, ...],
    meta: Dict[str, Any],
    val: torch.Tensor,
) -> torch.fx.Node:
    node = gm.graph.call_function(target, args)
    # Keep nn_module_stack, which torch_executed_modules and engine layer names read.
    node.meta.update(
        {key: value for key, value in meta.items() if key != "tensor_meta"}
    )
    node.meta["val"] = val
    return node


def _split_heads(
    gm: torch.fx.GraphModule,
    permute: torch.fx.Node,
    source: torch.fx.Node,
    seq_size: Any,
    batch_heads_size: Any,
    heads: int,
) -> torch.fx.Node:
    x = source.meta["val"]
    seq, batch, embed = x.shape
    with gm.graph.inserting_before(permute):
        # Every size is given, because a reshape cannot infer one for a tensor with 0
        # elements.
        batch_size = (
            batch
            if type(batch) is int
            else _add(gm, aten.sym_size.int, (source, 1), permute.meta, batch)
        )
        four = _add(
            gm,
            aten.reshape.default,
            (source, [seq_size, batch_size, heads, embed // heads]),
            permute.meta,
            x.reshape(seq, batch, heads, embed // heads),
        )
        moved = _add(
            gm,
            aten.permute.default,
            (four, [1, 2, 0, 3]),
            permute.meta,
            four.meta["val"].permute(1, 2, 0, 3),
        )
        return _add(
            gm,
            aten.reshape.default,
            (moved, [batch_heads_size, seq_size, embed // heads]),
            permute.meta,
            moved.meta["val"].reshape(batch * heads, seq, embed // heads),
        )


def _has_the_attention_bug() -> bool:
    """Whether the installed TensorRT is 11.3, 11.4 or TensorRT-RTX 1.7.

    These versions can build a fused kernel that gives wrong results for float16 and
    bfloat16 attention whose heads are split as the module docstring describes.
    TensorRT 10.3, 10.15, 10.16, 11.0 and 11.1 build that attention correctly in the
    tested shapes. Other versions were not tested. The engine tests in
    test_split_sequence_first_attention_heads.py compare with PyTorch on every version,
    so they fail with a version that has the bug but is not in this list.
    """
    if trt._package_name == "tensorrt_rtx":
        return bool(trt.__version__.startswith("1.7."))
    return bool(trt.__version__.startswith(("11.3.", "11.4.")))


def split_sequence_first_attention_heads(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Rewrite the head splits of sequence-first attention, see the module docstring.

    Only float16 and bfloat16 attention with more than one head, written as
    bmm(softmax(bmm(q, k^T)), v), with the query optionally multiplied by a number, the
    softmax, k^T and that product used only there and the softmax optionally run in
    float32 as torch.autocast runs it, whose key is split as nn.MultiheadAttention
    splits it and is not selected from a projection packed with the value, and whose
    query and value are split the same way or are constants. Attention whose sequence is
    empty or has a length that depends on the data, such as one that a mask selects, and
    attention that torch_executed_ops or torch_executed_modules keep in PyTorch are left
    as they are. With a TensorRT version other than 11.3, 11.4 and TensorRT-RTX 1.7 the
    pass does nothing.
    """
    if not _has_the_attention_bug():
        return gm
    excluded = settings.torch_executed_ops

    def is_excluded(op: Any) -> bool:
        return op in excluded or ConverterRegistry.qualified_name_or_str(op) in excluded

    def stays_in_pytorch(node: torch.fx.Node) -> bool:
        return is_excluded(node.target) or node_in_torch_executed_module(
            node, settings.torch_executed_modules
        )

    # TensorRT does not fuse attention whose bmm or softmax runs in PyTorch, and the
    # reshapes and permutes that the rewrite adds have to run in TensorRT.
    if any(
        is_excluded(op)
        for op in (
            aten.bmm.default,
            aten._softmax.default,
            aten.permute.default,
            aten.reshape.default,
        )
    ):
        return gm
    count = 0
    rewritten: Dict[torch.fx.Node, torch.fx.Node] = {}
    for softmax in list(gm.graph.nodes):
        if not _is_call(
            softmax, aten._softmax.default
        ) or node_in_torch_executed_module(softmax, settings.torch_executed_modules):
            continue
        operands = _attention_operands(softmax)
        if operands is None:
            continue
        splits = [_head_split(operand) for _, operand in operands]
        # The rewrite would move a head split that stays in PyTorch, by its op or by its
        # module, into TensorRT.
        if not _needs_split(operands, splits) or any(
            split is not None
            and (stays_in_pytorch(operand) or stays_in_pytorch(operand.args[0]))
            for (_, operand), split in zip(operands, splits)
        ):
            continue
        for (user, operand), split in zip(operands, splits):
            if split is None:
                continue
            if operand not in rewritten:
                rewritten[operand] = _split_heads(gm, operand, *split)
            user.replace_input_with(operand, rewritten[operand])
        count += 1

    if count:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Split batch and heads of {count} sequence-first attention(s)")
    return gm
