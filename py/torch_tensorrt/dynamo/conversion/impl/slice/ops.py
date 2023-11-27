import math
from typing import Optional

from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import get_positive_dim
from torch_tensorrt.dynamo.conversion.impl.slice.base import slice
from torch_tensorrt.fx.converters.converter_utils import (
    has_dynamic_shape,
    prepend_ones,
    set_layer_name,
)
from torch_tensorrt.fx.types import Shape, TRTTensor


def slice_op(  # TODO: This should be slice not whatever is in base
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
    start: Optional[int],
    stop: Optional[int],
    step: int,
) -> TRTTensor:
    # Special case for start being None
    if start is None:
        start = 0

    # Special case for stop being None
    if stop is None:
        stop = input.shape[dim]

    dim = get_positive_dim(dim, len(input.shape))
    start = get_positive_dim(start, input.shape[dim])
    stop = get_positive_dim(stop, input.shape[dim])

    if has_dynamic_shape(input.shape):
        # Check whether slice target dim is dynamic shape dim
        assert input.shape[dim] != -1, "Can't slice on dynamic shape dimension!"

    start_slice = [0] * len(input.shape)
    start_slice[dim] = start
    stride_slice = [1] * len(input.shape)
    stride_slice[dim] = step
    output_shape = list(input.shape)
    output_shape[dim] = math.ceil((stop - start) / step)

    return slice(
        ctx, target, source_ir, name, input, start_slice, output_shape, stride_slice
    )


def expand(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_t: TRTTensor,
    shape: Shape,
) -> TRTTensor:
    shape_rank = len(shape)
    initial_tensor_rank = len(input_t.shape)

    # If the rank of the input tensor is less than the shape's rank, pad with ones
    if initial_tensor_rank < shape_rank:
        input_t = prepend_ones(
            ctx.net,
            input_t,
            name + "_expand_broadcast",
            shape_rank - initial_tensor_rank,
        )
    # If the rank of the input tensor is more than the shape's rank, raise error
    elif initial_tensor_rank > shape_rank:
        raise RuntimeError(
            f"expand called with {shape_rank}-dimensional shape on Tensor with {len(shape)} dimensions. "
            "Cannot expand to shape with rank smaller than original tensor."
        )

    # After the above padding, the shape and tensor rank must be equal
    assert len(input_t.shape) == shape_rank

    # -1 denotes taking the shape from the original input tensor
    shape = tuple(
        [input_t.shape[i] if shape[i] == -1 else shape[i] for i in range(shape_rank)]
    )

    # Establish the desired output shape, strides, and starting indices
    input_tensor_shape = tuple(input_t.shape)
    start = tuple([0] * shape_rank)
    stride = tuple(
        [int(i == o) for i, o in zip(input_tensor_shape, shape)]
    )  # stride == 1 if dimensions match, 0 otherwise
    layer = ctx.net.add_slice(input_t, start=start, shape=shape, stride=stride)
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)
