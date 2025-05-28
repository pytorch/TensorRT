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
from torch_tensorrt.fx.types import TRTTensor

logger = logging.getLogger(__name__)


def tril(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    row: TRTTensor,
    col: TRTTensor,
) -> TRTTensor:
    row_arange_tensor = impl.arange.arange(
        ctx, target, source_ir, name + "_arange_row", start=0, end=row, step=1
    )
    row_reshape_tensor = impl.shuffle.reshape(
        ctx, target, source_ir, name + "_reshape_row", row_arange_tensor, [row, 1]
    )

    col_arange_tensor = impl.arange.arange(
        ctx, target, source_ir, name + "_arange_col", start=0, end=col, step=1
    )
    col_reshape_tensor = impl.shuffle.reshape(
        ctx, target, source_ir, name + "_reshape_col", col_arange_tensor, [1, col]
    )

    mask = impl.elementwise.ge(
        ctx, target, source_ir, name + "_ge", row_reshape_tensor, col_reshape_tensor
    )
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
    # TODO: Handle attn_mask and is_causal arguments in the future
    query, key, value, attn_mask, dropout_p, is_causal = args
    logger.info(
        "Ignoring attn_mask and is_causal arguments provided by the original graph. "
        "This converter expects is_causal to be an input to the graph. For prefill phase, is_causal=True "
        "and for generate phase, is_causal=False since we pass only 1 input token at a time"
    )

    # TODO: remove this once we have a better way to handle the causal mask
    scale = kwargs.get("scale", None)
    source_ir = SourceIR.ATEN
    # implementation as described here: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    mm = impl.matmul.matrix_multiply(
        ctx,
        target,
        source_ir,
        name + "_mm",
        query,
        key,
        other_matrix_op=trt.MatrixOperation.TRANSPOSE,
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
        scaled = impl.elementwise.div(
            ctx,
            target,
            source_ir,
            name + "_scale",
            mm,
            sqrt_scaled,
        )
    else:
        scaled = impl.elementwise.mul(
            ctx,
            target,
            source_ir,
            name + "_scale",
            mm,
            scale,
        )

    # If is_causal is True, we need to generate a causal mask
    if is_causal:
        L, S = query.shape[-2], key.shape[-2]
        if L >= 0 and S >= 0:
            # static shape
            attn_bias = np.zeros((L, S), dtype=dtype._from(query.dtype).to(np.dtype))
            temp_mask = np.logical_not(np.tril(np.ones((L, S), dtype=np.bool_), k=0))
            attn_bias = np.ma.array(attn_bias, mask=temp_mask).filled(float("-inf"))
            attn_bias = get_trt_tensor(ctx, attn_bias, name + "_attn_bias")
        else:
            # if any of the L or S is dynamic shape
            if L < 0:
                L = impl.shape.shape(
                    ctx, target, source_ir, name + "_shape_0", query, 2
                )
            if S < 0:
                S = impl.shape.shape(ctx, target, source_ir, name + "_shape_1", key, 2)

            # generate the mask tensor
            tril_tensor = tril(ctx, target, source_ir, name + "_tril", L, S)

            temp_mask = impl.unary.logical_not(
                ctx, target, source_ir, name + "_logical_not", tril_tensor
            )
            temp_mask_casted = cast_trt_tensor(
                ctx, temp_mask, trt.float32, name + "_casted_bool", target, source_ir
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

        scaled_add_attn_bias = impl.elementwise.add(
            ctx, target, source_ir, name + "_attn_bias_add", scaled, attn_bias
        )
    else:
        scaled_add_attn_bias = scaled

    # Create a if condition to check if is_causal is True
    if isinstance(is_causal, TRTTensor):
        if_layer = ctx.net.add_if_conditional()
        condition, true_branch, false_branch = is_causal, scaled_add_attn_bias, scaled
        if_layer.set_condition(condition)
        output_layer = if_layer.add_output(true_branch, false_branch)
        scaled_add_attn_bias = output_layer.get_output(0)

    softmax = impl.normalization.softmax(
        ctx, target, source_ir, name + "_softmax", scaled_add_attn_bias, -1, False
    )
    out = impl.matmul.matrix_multiply(
        ctx,
        target,
        source_ir,
        name + "_out",
        softmax,
        value,
    )

    return out
