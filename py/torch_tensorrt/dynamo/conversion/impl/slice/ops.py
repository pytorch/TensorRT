import math
from typing import Optional, Sequence

import numpy as np
import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_positive_dim,
    get_trt_tensor,
)
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


def chunk(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    chunks: int,
    dim: int,
) -> TRTTensor:
    if chunks <= 0:
        raise RuntimeError(
            f"chunk expects `chunks` to be greater than 0, got: {chunks}"
        )

    shape = input.shape
    dim = get_positive_dim(dim, len(shape))

    if dim >= len(shape):
        raise RuntimeError(
            f"chunk expects `dim` to be less than the length of input shape, got: {dim}"
        )

    dynamic_shape = has_dynamic_shape(input.shape)
    if dynamic_shape > 0:
        # Check whether slice target dim is dynamic shape dim
        assert input.shape[dim] != -1, "Can't chunk on dynamic shape dimension!"

    size_dim = shape[dim]
    chunk_size = math.ceil(size_dim / chunks)
    result = []
    start = 0
    end = min(start + chunk_size, size_dim)
    cnt = 0

    while start < end:
        result.append(
            slice_op(
                ctx,
                target,
                source_ir,
                f"{name}_slice_{cnt}",
                input,
                dim,
                start,
                end,
                1,
            )
        )
        start = end
        end = min(start + chunk_size, size_dim)
        cnt += 1

    return result


def cumsum(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
) -> TRTTensor:
    input_shape = input.shape
    dim = get_positive_dim(dim, len(input_shape))
    loop = ctx.net.add_loop()
    axis = np.array(input_shape[dim])
    trip_limit = get_trt_tensor(ctx, axis, f"{name}_trip_limit")
    loop.add_trip_limit(trip_limit, trt.TripLimit.COUNT)
    iterator = loop.add_iterator(input, dim, reverse=False)
    data = iterator.get_output(0)
    new_dims = tuple(data.shape)
    zeros = np.zeros(new_dims)
    zero_trttensor = get_trt_tensor(ctx, zeros, f"{name}_initial_value")

    running_sum = loop.add_recurrence(zero_trttensor)
    set_layer_name(running_sum, target, f"{name}_running_sum", source_ir)
    running_sum_tensor = running_sum.get_output(0)

    current_sum = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_elementwise_add",
        data,
        running_sum_tensor,
    )
    running_sum.set_input(1, current_sum)

    loop_output = loop.add_loop_output(current_sum, trt.LoopOutput.CONCATENATE, dim)
    set_layer_name(loop_output, target, f"{name}_loop_output", source_ir)
    loop_output.set_input(1, trip_limit)
    return loop_output.get_output(0)


def tile(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dims: Sequence[int],
) -> TRTTensor:
    diff = len(dims) - len(input.shape)
    if diff > 0:
        # prepend 1 to input.shape
        new_shape = (1,) * diff + tuple(input.shape)
        input = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_prepend_input_shape", input, new_shape
        )
    elif diff < 0:
        # prepend 1 to dims
        dims = (1,) * -diff + tuple(dims)

    shapes = [i * j for i, j in zip(input.shape, dims)]
    starts = [0] * len(dims)
    strides = [1] * len(dims)
    layer = ctx.net.add_slice(input, tuple(starts), tuple(shapes), tuple(strides))
    layer.mode = trt.SampleMode.WRAP
    set_layer_name(layer, target, name)
    return layer.get_output(0)
