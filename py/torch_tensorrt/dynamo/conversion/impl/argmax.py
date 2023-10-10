from typing import Optional

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

    topk_layer = ctx.net.add_topk(out, trt.TopKOperation.MAX, 1, reduce_mask)
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
