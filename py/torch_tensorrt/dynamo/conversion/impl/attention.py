import math
from typing import Optional, Union

import numpy as np
import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt._enums import dtype
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import SourceIR, get_trt_tensor
from torch_tensorrt.fx.types import TRTTensor


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
    L, S = query.shape[-2], key.shape[-2]

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
        scaled = impl.elementwise.div(
            ctx,
            target,
            source_ir,
            name + "_scale",
            mm,
            math.sqrt(query.shape[-1]),
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
        attn_bias = np.zeros((L, S), dtype=dtype._from(query.dtype).to(np.dtype))
        temp_mask = np.logical_not(np.tril(np.ones((L, S), dtype=np.bool_), k=0))
        attn_bias = np.ma.array(attn_bias, mask=temp_mask).filled(float("-inf"))
        attn_bias = get_trt_tensor(ctx, attn_bias, name + "_attn_bias")

        scaled = impl.elementwise.add(
            ctx, target, source_ir, name + "_attn_bias_add", scaled, attn_bias
        )

    softmax = impl.normalization.softmax(
        ctx, target, source_ir, name + "_softmax", scaled, -1
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
