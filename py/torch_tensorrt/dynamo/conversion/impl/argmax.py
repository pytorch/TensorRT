from typing import Optional

import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    flatten_dims,
    get_axes_for_reduce_op,
)
from torch_tensorrt.fx.converters.converter_utils import (
    get_positive_dim,
    set_layer_name,
)
from torch_tensorrt.fx.types import TRTTensor

from . import squeeze


def argmax(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Optional[int],
    keep_dim: bool = False,
) -> TRTTensor:
    if input.dtype == trt.int32:
        input = cast_trt_tensor(ctx, input, trt.float32, name)

    # Three different cases here:
    # 1. dim == None, flatten input tensor first, keep_dim will be ignore and the output rank == input rank
    # 2. input rank == 1: TopK layer does not support 1 dimensional topk operation. Broadcast input to rank == 2
    # 3. normal cases, no additional handlings
    out = input

    if dim is None:
        shuffle_layer = ctx.net.add_shuffle(input)
        shuffle_layer.reshape_dims = (*flatten_dims(input, 0, -1), 1)
        set_layer_name(shuffle_layer, target, name + "_flatten")
        out = shuffle_layer.get_output(0)
    elif len(input.shape) == 1:
        shuffle_layer = ctx.net.add_shuffle(input)
        shuffle_layer.reshape_dims = (*input.shape, 1)
        set_layer_name(shuffle_layer, target, name + "_broadcast")
        out = shuffle_layer.get_output(0)

    reduce_mask = get_axes_for_reduce_op(0)
    if dim is not None:
        reduce_mask = get_axes_for_reduce_op(get_positive_dim(dim, len(out.shape)))

    topk_layer = ctx.net.add_topk(out, trt.TopKOperation.MAX, 1, reduce_mask)
    set_layer_name(topk_layer, target, name)

    out = topk_layer.get_output(1)

    if dim is None:
        out_shuffle_layer = ctx.net.add_shuffle(out)
        out_shuffle_layer.reshape_dims = (1,) * len(input.shape) if keep_dim else ()
        set_layer_name(out_shuffle_layer, target, name + "_broadcast")
        out = out_shuffle_layer.get_output(0)
    elif len(input.shape) == 1:
        out = squeeze.squeeze(
            ctx,
            target,
            SourceIR.ATEN,
            name + "_squeeze",
            out,
            1 if keep_dim else [0, 1],
        )
    elif not keep_dim:
        out = squeeze.squeeze(ctx, target, SourceIR.ATEN, name + "_squeeze", out, dim)

    return out
