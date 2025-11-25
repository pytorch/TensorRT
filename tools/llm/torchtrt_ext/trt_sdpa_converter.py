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
    prepend_ones,
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
    source_ir = SourceIR.ATEN

    # always create our own attn_mask
    query, key, value, mask, dropout_p, is_causal = args

    # The exported graph of LLM models have -1 in the attention heads dimension for the query tensor. This value is static for key and value tensors though.
    # TODO: We assume that the attention heads dimension is the same for key and value and query tensors. We can implement a lowering pass
    # that reads number of attention heads from model config similar to gemma3. For now, we directly use the key.shape[1] as the attention heads dimension.
    query = impl.shuffle.reshape(
        ctx,
        target,
        source_ir,
        name + "_query_reshape",
        input=query,
        shape=[query.shape[0], key.shape[1], query.shape[2], query.shape[3]],
    )
    # L, S = query.shape[-2], key.shape[-2]
    query_len = impl.shape.shape(ctx, target, source_ir, name + "_query_len", query, -2)
    key_len = impl.shape.shape(ctx, target, source_ir, name + "_key_len", key, -2)
    mask_tensor = tril(
        ctx,
        target,
        source_ir,
        name + "_tril",
        query_len,
        key_len,
    )

    diff = len(query.shape) - len(mask_tensor.shape)

    mask_tensor = prepend_ones(ctx, mask_tensor, name + "_prepend_ones", diff)
    attention_layer = ctx.net.add_attention(
        query, key, value, trt.AttentionNormalizationOp.SOFTMAX, False
    )

    assert attention_layer is not None, "attention layer is None"

    if is_causal:
        attention_layer.mask = mask_tensor

    attention_output = attention_layer.get_output(0)

    return attention_output
