# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
import torch_tensorrt
from torch.fx.node import Target
from torch_tensorrt._enums import dtype
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    SourceIR,
    cast_trt_tensor,
    get_trt_tensor,
)
from torch_tensorrt.dynamo.types import TRTTensor

logger = logging.getLogger(__name__)


def tril(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    row: TRTTensor,
    col: TRTTensor,
    sliding_window_size: Optional[int] = None,
    offset: Union[int, TRTTensor] = 0,
) -> TRTTensor:
    """
    Create a lower triangular mask tensor for attention mechanisms.

    This function generates a lower triangular mask that can be used in attention
    operations to enforce causal attention (each position can only attend to itself
    and previous positions). It optionally supports sliding window attention by
    limiting the attention span to a specified window size.

    The function creates the mask by:
    1. Generating row and column index tensors
    2. Computing the difference between row and column indices
    3. Creating a mask where row >= col (lower triangular)
    4. Optionally applying sliding window constraints

    Args:
        ctx: TensorRT conversion context for managing the conversion process
        target: Target operation identifier (usually the operation being converted)
        source_ir: Source IR type (e.g., ATEN, TRT) - can be None
        name: Base name for generated TensorRT operations (will be extended with suffixes)
        row: Tensor representing the number of rows (sequence length dimension)
        col: Tensor representing the number of columns (sequence length dimension)
        sliding_window_size: Optional sliding window size for attention span limitation.
                           If None, creates a full lower triangular mask.
                           If specified, creates a sliding window mask where each position
                           can only attend to positions within the window.

    Returns:
        TRTTensor: A boolean mask tensor with shape [batch, heads, seq_len, seq_len]
                  where True values indicate allowed attention positions.

    Example:
        # Create a full lower triangular mask for causal attention
        mask = tril(ctx, target, source_ir, "causal_mask", seq_len, seq_len)

        # Create a sliding window mask with window size 3
        mask = tril(ctx, target, source_ir, "sliding_mask", seq_len, seq_len, 3)

    Mask Examples:
        Without sliding window (sliding_window_size=None):
        For seq_len=5, returns:
        [[ True, False, False, False, False],
         [ True,  True, False, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True]]

        With sliding window (sliding_window_size=3):
        For seq_len=5, returns:
        [[ True, False, False, False, False],
         [ True,  True, False, False, False],
         [ True,  True,  True, False, False],
         [False,  True,  True,  True, False],
         [False, False,  True,  True,  True]]

    Note:
        This function is specifically designed for attention mechanisms in transformer
        models and is used internally by the scaled_dot_product_attention converter.
        The sliding window functionality is particularly useful for models like Gemma3
        that use sliding window attention to reduce computational complexity.
    """
    row_arange_tensor = impl.arange.arange(
        ctx, target, source_ir, name + "_arange_row", start=0, end=row, step=1
    )
    col_arange_tensor = impl.arange.arange(
        ctx, target, source_ir, name + "_arange_col", start=0, end=col, step=1
    )
    row_arange_tensor = impl.unsqueeze.unsqueeze(
        ctx, target, source_ir, name + "_unsqueeze_row", row_arange_tensor, -1
    )
    col_arange_tensor = impl.unsqueeze.unsqueeze(
        ctx, target, source_ir, name + "_unsqueeze_col", col_arange_tensor, 0
    )
    # sub will return the following mask tensor:
    # [[0, -1, -2, -3],
    #  [1,  0, -1, -2],
    #  [2,  1,  0, -1],
    #  [3,  2,  1,  0]]
    mask = impl.elementwise.sub(
        ctx, target, source_ir, name + "_sub", row_arange_tensor, col_arange_tensor
    )
    if not (isinstance(offset, int) and offset == 0):
        # Queries are the last ``row`` of ``col`` positions (KV cache): shift the diagonal.
        mask = impl.elementwise.add(
            ctx, target, source_ir, name + "_offset", mask, offset
        )
    ge_0_mask = impl.elementwise.ge(ctx, target, source_ir, name + "_ge_0", mask, 0.0)
    if sliding_window_size is None:
        # return the following lower triangular mask includes the main diagonal:
        # 0 ■ ⬚ ⬚ ⬚ ⬚     tensor([[[[ True, False, False, False, False],
        # 1 ■ ■ ⬚ ⬚ ⬚               [ True,  True, False, False, False],
        # 2 ■ ■ ■ ⬚ ⬚               [ True,  True,  True, False, False],
        # 3 ■ ■ ■ ■ ⬚               [ True,  True,  True,  True, False],
        # 4 ■ ■ ■ ■ ■               [ True,  True,  True,  True,  True]]]])
        return ge_0_mask

    lt_window_mask = impl.elementwise.lt(
        ctx, target, source_ir, name + "_lt_window_size", mask, sliding_window_size
    )
    mask = impl.elementwise.logical_and(
        ctx, target, source_ir, name + "_logical_and", ge_0_mask, lt_window_mask
    )
    # return the following mask if sliding_window_size is 3:
    # 0 ■ ⬚ ⬚ ⬚ ⬚      tensor([[[[ True, False, False, False, False],
    # 1 ■ ■ ⬚ ⬚ ⬚                [ True,  True, False, False, False],
    # 2 ■ ■ ■ ⬚ ⬚                [ True,  True,  True, False, False],
    # 3 ⬚ ■ ■ ■ ⬚                [False,  True,  True,  True, False],
    # 4 ⬚ ⬚ ■ ■ ■                [False, False,  True,  True,True]]]])
    return mask


def _dim(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    tensor: TRTTensor,
    index: int,
) -> Union[int, TRTTensor]:
    """Size of ``tensor`` along ``index``: a python int when static, else a shape tensor."""
    size = tensor.shape[index]
    if size >= 0:
        return size
    return impl.shape.shape(ctx, target, source_ir, name, tensor, index)


@torch_tensorrt.dynamo.conversion.dynamo_tensorrt_converter(
    torch.nn.functional.scaled_dot_product_attention,
    enabled=True,
    supports_dynamic_shapes=True,
)
def scaled_dot_product_attention(
    ctx: torch_tensorrt.dynamo.conversion.ConversionContext,
    target: Target,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    name: str,
) -> TRTTensor:
    # always create our own attn_mask
    query, key, value, _, dropout_p, is_causal = args

    # TODO: remove this once we have a better way to handle the causal mask
    scale = kwargs.get("scale", None)
    source_ir = SourceIR.ATEN

    assert is_causal == True, "is_causal should be set to True"

    # use_fp32_acc is read from compilation settings rather than node kwargs so
    # that SDPA nodes which fall back to PyTorch execution are never called
    # with an argument that torch.nn.functional.scaled_dot_product_attention
    # does not accept.
    use_fp32_acc = ctx.compilation_settings.use_fp32_acc
    # sliding_window_size is model-specific (Gemma3 only).  The lowering pass
    # stores it in node.meta["trt_sliding_window_size"], but ConversionContext
    # does not expose the FX node directly.  Until the plumbing exists, we
    # read it from kwargs (set to None for all non-Gemma3 models) so that
    # Qwen/Llama compilation is unaffected.
    sliding_window_size = kwargs.get("trt_sliding_window_size", None)

    query_dtype = query.dtype

    # Grouped-query attention: query has ``groups`` times more heads than key/value. Instead of
    # expanding K/V (which would also force the KV cache to be stored expanded), the query heads
    # that share a KV head are folded into the row dimension:
    #   [B, Hq, L, D] -> [B, Hkv, groups * L, D]
    # Row ``g * L + l`` of KV head ``j`` is query head ``j * groups + g`` at position ``l`` (the
    # same head -> KV head mapping as ``repeat_kv``). The causal bias is tiled ``groups`` times.
    num_q_heads, num_kv_heads = query.shape[1], key.shape[1]
    groups: Union[int, TRTTensor] = 1
    if num_q_heads < 0 and num_kv_heads > 0 and value.shape[1] == num_kv_heads:
        # TensorRT lost the (static) query head count, e.g. through a broadcast against a dynamic
        # shaped RoPE table. Recover it as a shape tensor; the group count is then dynamic too.
        num_q_heads = _dim(ctx, target, source_ir, name + "_q_heads", query, 1)
        groups = impl.elementwise.floor_divide(
            ctx, target, source_ir, name + "_groups", num_q_heads, num_kv_heads
        )
    elif num_q_heads != num_kv_heads:
        if (
            num_kv_heads <= 0
            or num_q_heads % num_kv_heads != 0
            or value.shape[1] != num_kv_heads
        ):
            raise ValueError(
                f"Unsupported GQA head counts: query={num_q_heads}, key={key.shape[1]}, "
                f"value={value.shape[1]}"
            )
        groups = num_q_heads // num_kv_heads
    dynamic_groups = not isinstance(groups, int)
    q_batch = _dim(ctx, target, source_ir, name + "_q_batch", query, 0)
    q_len = _dim(ctx, target, source_ir, name + "_q_len", query, 2)
    head_dim = query.shape[-1]
    if dynamic_groups or groups > 1:
        folded_rows = (
            q_len * groups
            if isinstance(q_len, int) and not dynamic_groups
            else impl.elementwise.mul(
                ctx, target, source_ir, name + "_folded_rows", q_len, groups
            )
        )
        query = impl.shuffle.reshape(
            ctx,
            target,
            source_ir,
            name + "_fold_q",
            query,
            [q_batch, num_kv_heads, folded_rows, head_dim],
        )

    if scale is None:
        scale = query.shape[-1]
        if scale < 0:
            # dynamic shape
            scale = impl.shape.shape(ctx, target, source_ir, name + "_shape", query, -1)
            sqrt_scaled = impl.unary.sqrt(ctx, target, source_ir, name + "_sqrt", scale)
        else:
            # static shape
            sqrt_scaled = math.sqrt(scale)
        key = impl.elementwise.div(
            ctx,
            target,
            source_ir,
            name + "_scale",
            key,
            sqrt_scaled,
        )
    else:
        key = impl.elementwise.mul(
            ctx,
            target,
            source_ir,
            name + "_scale",
            key,
            scale,
        )

    if use_fp32_acc and query_dtype == trt.float16:
        query = cast_trt_tensor(
            ctx, query, trt.float32, name + "_query_cast_to_fp32", target, source_ir
        )
        key = cast_trt_tensor(
            ctx, key, trt.float32, name + "_key_cast_to_fp32", target, source_ir
        )

    mm = impl.matmul.matrix_multiply(
        ctx,
        target,
        source_ir,
        name + "_mm",
        query,
        key,
        other_matrix_op=trt.MatrixOperation.TRANSPOSE,
    )

    if use_fp32_acc:
        mm = cast_trt_tensor(
            ctx, mm, query_dtype, name + "_mm_cast_to_fp16", target, source_ir
        )

    L, S = q_len, key.shape[-2]
    if isinstance(L, int) and S >= 0:
        # static shape
        attn_bias = np.zeros((L, S), dtype=dtype._from(query_dtype).to(np.dtype))
        temp_mask = np.logical_not(np.tril(np.ones((L, S), dtype=np.bool_), k=0))
        attn_bias = np.ma.array(attn_bias, mask=temp_mask).filled(float("-inf"))
        if not dynamic_groups:
            attn_bias = np.tile(attn_bias, (groups, 1))
        attn_bias = get_trt_tensor(ctx, attn_bias, name + "_attn_bias")
    else:
        # if any of the L or S is dynamic shape
        if S < 0:
            S = impl.shape.shape(ctx, target, source_ir, name + "_shape_1", key, 2)

        # generate the mask tensor. The L queries are the last L of the S keys (KV cache), so
        # query i sees keys <= i + (S - L): this covers prefill (L == S), chunked prefill and
        # decode (L == 1, everything visible).
        key_offset = (
            S - L
            if isinstance(S, int) and isinstance(L, int)
            else impl.elementwise.sub(ctx, target, source_ir, name + "_s_minus_l", S, L)
        )
        tril_tensor = tril(
            ctx,
            target,
            source_ir,
            name + "_tril",
            L,
            S,
            sliding_window_size,
            key_offset,
        )
        temp_mask = impl.unary.logical_not(
            ctx, target, source_ir, name + "_logical_not", tril_tensor
        )
        temp_mask_casted = cast_trt_tensor(
            ctx, temp_mask, query_dtype, name + "_casted_bool", target, source_ir
        )

        one_minus_temp_mask = impl.elementwise.sub(
            ctx,
            target,
            source_ir,
            name + "_one_minus_temp_mask",
            1.0,
            temp_mask_casted,
        )
        attn_bias = impl.unary.log(
            ctx, target, source_ir, name + "_log", one_minus_temp_mask
        )
        if not dynamic_groups and groups > 1:
            attn_bias = impl.cat.cat(
                ctx,
                target,
                source_ir,
                name + "_tile_bias",
                [attn_bias] * groups,
                dim=0,
            )

    if dynamic_groups:
        # No tiling by a dynamic count: view the rows as [groups, L] so the [L, S] bias broadcasts.
        mm_shape = [
            q_batch,
            num_kv_heads,
            groups,
            q_len,
            _dim(ctx, target, source_ir, name + "_s", mm, 3),
        ]
        mm = impl.shuffle.reshape(ctx, target, source_ir, name + "_mm_5d", mm, mm_shape)
    scaled_add_attn_bias = impl.elementwise.add(
        ctx, target, source_ir, name + "_attn_bias_add", mm, attn_bias
    )
    if dynamic_groups:
        scaled_add_attn_bias = impl.shuffle.reshape(
            ctx,
            target,
            source_ir,
            name + "_mm_4d",
            scaled_add_attn_bias,
            [q_batch, num_kv_heads, folded_rows, mm_shape[-1]],
        )
    softmax = impl.normalization.softmax(
        ctx, target, source_ir, name + "_softmax", scaled_add_attn_bias, -1, False
    )
    if use_fp32_acc:
        softmax = cast_trt_tensor(
            ctx, softmax, trt.float32, name + "_softmax_cast_to_fp32", target, source_ir
        )
        value = cast_trt_tensor(
            ctx, value, trt.float32, name + "_value_cast_to_fp32", target, source_ir
        )
    out = impl.matmul.matrix_multiply(
        ctx,
        target,
        source_ir,
        name + "_out",
        softmax,
        value,
    )
    if use_fp32_acc:
        out = cast_trt_tensor(
            ctx, out, query_dtype, name + "_out_cast_to_fp16", target, source_ir
        )

    if dynamic_groups or groups > 1:
        out = impl.shuffle.reshape(
            ctx,
            target,
            source_ir,
            name + "_unfold_out",
            out,
            [q_batch, num_q_heads, q_len, head_dim],
        )

    return out
