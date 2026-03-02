import logging
from typing import Optional, Tuple, Union

import tensorrt as trt
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_trt_tensor,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)


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


def scaled_dot_product_attention(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    query: TRTTensor,
    key: TRTTensor,
    value: TRTTensor,
    attn_mask: Optional[TRTTensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> TRTTensor:
    """Convert the scaled_dot_product_attention operation to a TensorRT attention layer.

    Args:
        ctx (ConversionContext): Conversion context for managing the conversion process
        target (Target): Target operation identifier (usually the operation being converted)
        source_ir (Optional[SourceIR]): Source IR type (e.g., ATEN)
        name (str): Base name for generated TensorRT operations (will be extended with suffixes)
        query (TRTTensor): Query tensor with shape [batch, heads, seq_len, head_dim]
        key (TRTTensor): Key tensor with shape [batch, heads, seq_len, head_dim]
        value (TRTTensor): Value tensor with shape [batch, heads, seq_len, head_dim]
        attn_mask (Optional[TRTTensor], optional): Attention mask tensor. The shape must be broadcastable to the shape of attention weights. Two types of masks are supported. A boolean mask where a value of True indicates that the element should take part in attention. A float mask of the same type as query, key, value that is added to the attention score. If is_causal is set to True, attn_mask should be None.
        dropout_p (float, optional): Ignored in inference stage
        is_causal (bool, optional): Whether to apply causal masking. If True, the attention mask will be a lower triangular mask. If is_causal is set to True, the attn_mask argument should be None.
        scale (Optional[float], optional): Scaling factor for the attention layer
        enable_gqa (bool, optional): Whether to enable Group Query Attention (GQA). If True, the query tensor will be split into groups and the attention will be computed for each group.

    Returns:
        TRTTensor: Attention output tensor with shape [batch, heads, seq_len, head_dim]
    """
    assert enable_gqa is False, "enable_gqa is not supported yet"
    assert (
        tuple(query.shape) == tuple(key.shape) == tuple(value.shape)
    ), "query, key, and value must have the same shape"

    if scale is None:
        # 1 / math.sqrt(query.size(-1))
        q_dim = impl.shape.shape(ctx, target, source_ir, f"{name}_shape_q", query, -1)
        sqrt_q_dim = impl.unary.sqrt(
            ctx, target, source_ir, f"{name}_sqrt_q_dim", q_dim
        )
        scale_factor = impl.elementwise.div(
            ctx, target, source_ir, f"{name}_div_1_sqrt_q_dim", 1, sqrt_q_dim
        )
    else:
        scale_factor = get_trt_tensor(ctx, scale, f"{name}_scale_factor", query.dtype)

    mask_tensor = None
    if is_causal:
        assert attn_mask is None, "attn_mask should be None when is_causal is True"
        # L = impl.shape.shape(ctx, target, source_ir, name + "_L", query, -2)
        # S = impl.shape.shape(ctx, target, source_ir, name + "_S", key, -2)
        # mask_tensor = tril(
        #     ctx,
        #     target,
        #     source_ir,
        #     name + "_tril",
        #     L,
        #     S,
        # )
        # diff = len(query.shape) - len(mask_tensor.shape)
        # mask_tensor = prepend_ones(ctx, mask_tensor, name + "_prepend_ones", diff)

    if attn_mask is not None:
        if attn_mask.dtype == trt.DataType.BOOL:
            mask_tensor = attn_mask
        else:
            if attn_mask.dtype != query.dtype:
                mask_tensor = cast_trt_tensor(
                    ctx,
                    attn_mask,
                    query.dtype,
                    name + "_cast_attn_mask",
                    target,
                    source_ir,
                )
            else:
                mask_tensor = attn_mask

    scaled_query = impl.elementwise.mul(
        ctx, target, source_ir, f"{name}_scaled_query", query, scale_factor
    )
    if scaled_query.dtype != query.dtype:
        scaled_query = cast_trt_tensor(
            ctx,
            scaled_query,
            query.dtype,
            name + "_cast_scaled_query",
            target,
            source_ir,
        )

    attention_layer = ctx.net.add_attention(
        scaled_query, key, value, trt.AttentionNormalizationOp.SOFTMAX, is_causal
    )
    assert attention_layer is not None, "attention layer is None"

    if mask_tensor is not None:
        attention_layer.mask = mask_tensor
    attention_layer.decomposable = True
    attention_output = attention_layer.get_output(0)
    return attention_output


def scaled_dot_product_flash_attention(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    query: TRTTensor,
    key: TRTTensor,
    value: TRTTensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: Optional[float] = None,
) -> Tuple[
    TRTTensor,
    Optional[TRTTensor],
    Optional[TRTTensor],
    Optional[TRTTensor],
    Optional[TRTTensor],
    Optional[TRTTensor],
    Optional[TRTTensor],
    Optional[TRTTensor],
    Optional[TRTTensor],
]:
    assert return_debug_mask is False, "return_debug_mask is not supported yet"
    assert (
        tuple(query.shape) == tuple(key.shape) == tuple(value.shape)
    ), "query, key, and value must have the same shape"

    if scale is None:
        # 1 / math.sqrt(query.size(-1))
        q_dim = impl.shape.shape(ctx, target, source_ir, f"{name}_shape_q", query, -1)
        sqrt_q_dim = impl.unary.sqrt(
            ctx, target, source_ir, f"{name}_sqrt_q_dim", q_dim
        )
        scale_factor = impl.elementwise.div(
            ctx, target, source_ir, f"{name}_div_1_sqrt_q_dim", 1, sqrt_q_dim
        )
    else:
        scale_factor = get_trt_tensor(ctx, scale, f"{name}_scale_factor", query.dtype)

    scaled_query = impl.elementwise.mul(
        ctx, target, source_ir, f"{name}_scaled_query", query, scale_factor
    )
    if scaled_query.dtype != query.dtype:
        scaled_query = cast_trt_tensor(
            ctx,
            scaled_query,
            query.dtype,
            name + "_cast_scaled_query",
            target,
            source_ir,
        )

    attention_layer = ctx.net.add_attention(
        scaled_query, key, value, trt.AttentionNormalizationOp.SOFTMAX, is_causal
    )
    assert attention_layer is not None, "attention layer is None"

    attention_layer.decomposable = True

    attention_output = attention_layer.get_output(0)
    return attention_output, None, None, None, 0.0, 0.0, None, None, None


def scaled_dot_product_efficient_attention(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    query: TRTTensor,
    key: TRTTensor,
    value: TRTTensor,
    attn_bias: Optional[TRTTensor] = None,
    compute_log_sumexp: bool = False,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tuple[TRTTensor, Optional[TRTTensor], Optional[TRTTensor], Optional[TRTTensor]]:
    assert compute_log_sumexp is False, "compute_log_sumexp is not supported yet"
    assert (
        tuple(query.shape) == tuple(key.shape) == tuple(value.shape)
    ), "query, key, and value must have the same shape"

    if scale is None:
        # 1 / math.sqrt(query.size(-1))
        q_dim = impl.shape.shape(ctx, target, source_ir, f"{name}_shape_q", query, -1)
        sqrt_q_dim = impl.unary.sqrt(
            ctx, target, source_ir, f"{name}_sqrt_q_dim", q_dim
        )
        scale_factor = impl.elementwise.div(
            ctx, target, source_ir, f"{name}_div_1_sqrt_q_dim", 1, sqrt_q_dim
        )
    else:
        scale_factor = get_trt_tensor(ctx, scale, f"{name}_scale_factor", query.dtype)

    scaled_query = impl.elementwise.mul(
        ctx, target, source_ir, f"{name}_scaled_query", query, scale_factor
    )
    if scaled_query.dtype != query.dtype:
        scaled_query = cast_trt_tensor(
            ctx,
            scaled_query,
            query.dtype,
            name + "_cast_scaled_query",
            target,
            source_ir,
        )

    if attn_bias is not None:
        attn_weight = impl.matmul.matrix_multiply(
            ctx,
            target,
            source_ir,
            name + "_mm",
            scaled_query,
            key,
            other_matrix_op=trt.MatrixOperation.TRANSPOSE,
        )
        attn_weight = impl.elementwise.add(
            ctx, target, source_ir, name + "_attn_bias_add", attn_weight, attn_bias
        )
        attn_weight = impl.normalization.softmax(
            ctx, target, source_ir, name + "_softmax", attn_weight, -1, False
        )
        out = impl.matmul.matrix_multiply(
            ctx,
            target,
            source_ir,
            name + "_out",
            attn_weight,
            value,
        )
        return out, None, None, None
    else:
        attention_layer = ctx.net.add_attention(
            scaled_query, key, value, trt.AttentionNormalizationOp.SOFTMAX, is_causal
        )
        assert attention_layer is not None, "attention layer is None"

        attention_layer.decomposable = True

        attention_output = attention_layer.get_output(0)
        return attention_output, None, None, None


def scaled_dot_product_cudnn_attention(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    query: TRTTensor,
    key: TRTTensor,
    value: TRTTensor,
    attn_bias: Optional[TRTTensor] = None,
    compute_log_sumexp: bool = False,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: Optional[float] = None,
) -> Tuple[
    TRTTensor,
    Optional[TRTTensor],
    Optional[TRTTensor],
    Optional[TRTTensor],
    Optional[TRTTensor],
    Optional[TRTTensor],
    Optional[TRTTensor],
    Optional[TRTTensor],
    Optional[TRTTensor],
]:
    assert return_debug_mask is False, "return_debug_mask is not supported yet"

    output, _, _, _ = scaled_dot_product_efficient_attention(
        ctx,
        target,
        source_ir,
        f"{name}_efficient",
        query,
        key,
        value,
        attn_bias,
        compute_log_sumexp,
        dropout_p,
        is_causal,
        scale,
    )
    return output, None, None, None, 0.0, 0.0, None, None, None
