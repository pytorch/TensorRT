from typing import Optional, Tuple, Union

import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    flatten_dims,
    get_axes_for_reduce_op,
    get_positive_dim,
)
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTTensor


def argmax_argmin(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    topk_option: trt.TopKOperation,
    dim: Optional[int],
    keep_dim: bool = False,
) -> TRTTensor:
    if input.dtype == trt.int32:
        input = cast_trt_tensor(ctx, input, trt.float32, name, target, source_ir)

    # Three different cases here:
    # 1. dim == None, flatten input tensor first, keep_dim will be ignore and the output rank == input rank
    # 2. input rank == 1: TopK layer does not support 1 dimensional topk operation. Broadcast input to rank == 2
    # 3. normal cases, no additional handlings
    out = input

    if dim is None:
        new_shape = (*flatten_dims(input, 0, -1), 1)
        out = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_flatten", input, new_shape
        )
    elif len(input.shape) == 1:
        new_shape = (*input.shape, 1)
        out = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_broadcast", input, new_shape
        )

    # Reduce over the flattened input if the dimension is None, otherwise the specified dimension
    reduce_mask = get_axes_for_reduce_op(
        get_positive_dim(dim if dim is not None else 0, len(out.shape))
    )

    topk_layer = ctx.net.add_topk(out, topk_option, 1, reduce_mask)
    set_layer_name(topk_layer, target, name, source_ir)

    out = topk_layer.get_output(1)

    if dim is None:
        new_shape = ((1,) * len(input.shape)) if keep_dim else ()
        out = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_unflatten", out, new_shape
        )
    elif len(input.shape) == 1:
        out = impl.squeeze.squeeze(
            ctx,
            target,
            source_ir,
            f"{name}_squeeze",
            out,
            1 if keep_dim else (0, 1),
        )
    elif not keep_dim:
        out = impl.squeeze.squeeze(ctx, target, source_ir, f"{name}_squeeze", out, dim)

    return out


def argmax(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Optional[int],
    keep_dim: bool = False,
) -> TRTTensor:
    return argmax_argmin(
        ctx, target, source_ir, name, input, trt.TopKOperation.MAX, dim, keep_dim
    )


def argmin(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Optional[int],
    keep_dim: bool = False,
) -> TRTTensor:
    return argmax_argmin(
        ctx, target, source_ir, name, input, trt.TopKOperation.MIN, dim, keep_dim
    )


def sort(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
    descending: bool,
    return_indices: bool = True,
) -> Union[TRTTensor, Tuple[TRTTensor, TRTTensor]]:
    if descending:
        topk_layer = ctx.net.add_topk(
            input,
            trt.TopKOperation.MAX,
            input.shape[dim],
            get_axes_for_reduce_op(get_positive_dim(dim, len(input.shape))),
        )
    else:
        topk_layer = ctx.net.add_topk(
            input,
            trt.TopKOperation.MIN,
            input.shape[dim],
            get_axes_for_reduce_op(get_positive_dim(dim, len(input.shape))),
        )

    set_layer_name(topk_layer, target, name, source_ir)

    if return_indices:
        return topk_layer.get_output(0), topk_layer.get_output(1)
    else:
        return topk_layer.get_output(0)
