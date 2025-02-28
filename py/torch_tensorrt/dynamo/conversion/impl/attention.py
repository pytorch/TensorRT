import math
from typing import Optional, Union

import numpy as np
import tensorrt as trt
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


def tril(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
) -> TRTTensor:
    # the lower triangle of the tensor means the rows greater than and equal to the cols
    row = impl.shape.shape(ctx, target, source_ir, name + "_shape_0", input, 0)
    col = impl.shape.shape(ctx, target, source_ir, name + "_shape_1", input, 1)
    rc = impl.elementwise.mul(ctx, target, source_ir, name + "_mul", row, col)
    arange_tensor = impl.arange.arange(
        ctx, target, source_ir, name + "_arange", start=0, end=rc, step=1
    )
    # get the rows
    row_tensor = impl.elementwise.trunc_div(
        ctx, target, source_ir, name + "_trunc_div_col", arange_tensor, col
    )
    # get the cols
    col_tensor = impl.elementwise.fmod(
        ctx, target, source_ir, name + "_trunc_div_row", arange_tensor, col
    )
    cond = impl.elementwise.ge(
        ctx, target, source_ir, name + "_ge", row_tensor, col_tensor
    )
    return impl.shuffle.reshape(
        ctx, target, source_ir, name + "_reshape", cond, [row, col]
    )


def scaled_dot_product_attention(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    query: TRTTensor,
    key: TRTTensor,
    value: TRTTensor,
    is_causal: bool,
    scale: Optional[float],
) -> TRTTensor:
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
                    ctx, target, source_ir, name + "_shape_0", query, -2
                )
            if S < 0:
                S = impl.shape.shape(ctx, target, source_ir, name + "_shape_1", key, -2)

            LS = impl.elementwise.mul(ctx, target, source_ir, name + "_mul", L, S)

            # this is to generate a tensor which has shape (L, S), type is int32
            arange_tensor = impl.arange.arange(
                ctx, target, source_ir, name=name + "_arange", start=0, end=LS, step=1
            )
            shape_tensor = impl.shuffle.reshape(
                ctx, target, source_ir, name + "_reshape", arange_tensor, [L, S]
            )

            # since we want our attn_bias to be in float32, so cast it to float32
            shape_tensor = cast_trt_tensor(
                ctx, shape_tensor, trt.DataType.FLOAT, name + "_casted", target, source_ir
            )

            # initialize the attn_bias as the zeros tensor
            attn_bias = impl.elementwise.mul(
                ctx, target, source_ir, name + "_mul_zero", shape_tensor, 0.0
            )

            # generate the mask tensor
            tril_tensor = tril(ctx, target, source_ir, name + "_tril", shape_tensor)
            temp_mask = impl.unary.logical_not(
                ctx, target, source_ir, name + "_logical_not", tril_tensor
            )
            inf_tensor = impl.elementwise.mul(
                ctx, target, source_ir, name + "_mul_-inf", shape_tensor, float("-inf")
            )
            cond = impl.elementwise.eq(
                ctx, target, source_ir, name + "_cond_true", temp_mask, bool(True)
            )
            # mask out the certain part of the attn_bias
            attn_bias = impl.condition.select(
                ctx, target, source_ir, name + "_select", inf_tensor, attn_bias, cond
            )

        scaled = impl.elementwise.add(
            ctx, target, source_ir, name + "_attn_bias_add", scaled, attn_bias
        )

    softmax = impl.normalization.softmax(
        ctx, target, source_ir, name + "_softmax", scaled, -1, False
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
