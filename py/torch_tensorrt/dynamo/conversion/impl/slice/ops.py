import math
import sys
import time
from typing import Optional, Sequence

import numpy as np
import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    calculate_strides,
    ceil_divide,
    flatten_dims,
    get_positive_dim,
    get_trt_tensor,
)
from torch_tensorrt.dynamo.conversion.impl.cat import cat
from torch_tensorrt.dynamo.conversion.impl.elementwise import add, floor_divide
from torch_tensorrt.dynamo.conversion.impl.elementwise import min as torch_trt_min
from torch_tensorrt.dynamo.conversion.impl.elementwise import sub
from torch_tensorrt.dynamo.conversion.impl.elementwise.ops import (
    convert_binary_elementwise,
)
from torch_tensorrt.dynamo.conversion.impl.shape import get_shape_with_dynamic_shape
from torch_tensorrt.dynamo.conversion.impl.shape import shape as get_shape
from torch_tensorrt.dynamo.conversion.impl.slice.base import slice
from torch_tensorrt.dynamo.utils import DYNAMIC_DIM
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
    # check if dim is same as dynamic shape dimension
    # this is required when stop is ITensor
    dynamic_input_dim_equal = False
    for i in range(len(input.shape)):
        if input.shape[i] == DYNAMIC_DIM and i == dim:
            dynamic_input_dim_equal = True

    # Special case for start being None
    if start is None:
        start = 0

    # Special case for stop being None
    stop_dynamic_None = False
    if stop is None:
        stop_dynamic_None = True if input.shape[dim] == -1 else False
        stop = 0 if input.shape[dim] == -1 else input.shape[dim]

    dim = get_positive_dim(dim, len(input.shape))

    # Assign the initial start tensor
    start_slice = []
    # the add_slice will take care of dynamic input shape cases here
    if isinstance(start, int):
        start_slice = [0] * len(input.shape)
        start_slice[dim] = start
    else:
        for i in range(len(input.shape)):
            start_slice.append(0) if i != dim else start_slice.append(start)

    # Assign the initial stop tensor
    stop_slice = []
    if isinstance(stop, int) and dynamic_input_dim_equal:
        stop_slice = input.shape
        stop_slice[dim] = stop
    else:
        # required for cases where stop is ITensor and dim != dynamic dim of input
        # not required for cases where stop is negative and dim != dynamic dim of inpu
        for i in range(len(input.shape)):
            if input.shape[i] == DYNAMIC_DIM and i != dim:
                stop_slice.append(
                    get_shape(
                        ctx, target, source_ir, name + f"_shape_dim_stop_{i}", input, i
                    )
                )
            elif i == dim:
                stop_slice.append(stop)
            else:
                stop_slice.append(input.shape[i])

    stride_slice = [1] * len(input.shape)
    stride_slice[dim] = step
    output_shape = list(input.shape)

    if input.shape[dim] != -1 and isinstance(start, int) and isinstance(stop, int):
        start = get_positive_dim(start, input.shape[dim])
        stop = get_positive_dim(stop, input.shape[dim])
        start_slice[dim] = start
    else:
        # the start and stop or None is dynamic along dim or or start or stop is an ITensor
        if (
            not (isinstance(start, int))
            or not (isinstance(stop, int))
            or start < 0
            or stop < 0
            or stop_dynamic_None
            or stop == sys.maxsize
        ):
            # special assignments for dynamic cases
            if isinstance(start, int) and start < 0:
                start_slice = input.shape
                start_slice[dim] = -1 * start
            if (isinstance(stop, int) and stop < 0) or stop_dynamic_None:
                stop_slice = [0] * len(input.shape)
                stop_slice[dim] = -1 * stop
            if stop == sys.maxsize:
                stop_slice = [0] * len(input.shape)
            start_slice_tensor = cat(
                ctx,
                target,
                source_ir,
                name + "_start_slice_concat",
                tuple(start_slice),
                0,
                cast_dtype=trt.int32,
            )
            stop_slice_tensor = cat(
                ctx,
                target,
                source_ir,
                name + "_stop_slice_concat",
                tuple(stop_slice),
                0,
                cast_dtype=trt.int32,
            )
            stride_slice_tensor = cat(
                ctx,
                target,
                source_ir,
                name + "_stride_slice_concat",
                tuple(stride_slice),
                0,
                cast_dtype=trt.int32,
            )

            if isinstance(start, int) and start < 0:
                shape = get_shape_with_dynamic_shape(
                    ctx, target, source_ir, name, output_shape, input
                )
                start_slice_tensor = convert_binary_elementwise(
                    ctx,
                    target,
                    source_ir,
                    name + "_sub_start",
                    trt.ElementWiseOperation.SUB,
                    shape,
                    start_slice_tensor,
                )
            if isinstance(stop, int) and (
                (stop < 0) or stop_dynamic_None or stop == sys.maxsize
            ):
                shape = get_shape_with_dynamic_shape(
                    ctx, target, source_ir, name, output_shape, input
                )
                stop_slice_tensor = convert_binary_elementwise(
                    ctx,
                    target,
                    source_ir,
                    name + "_sub_stop",
                    trt.ElementWiseOperation.SUB,
                    shape,
                    stop_slice_tensor,
                )

            # this is required for the ceil operation
            output_shape_tensor_num = convert_binary_elementwise(
                ctx,
                target,
                source_ir,
                name + "_sub_num",
                trt.ElementWiseOperation.SUB,
                start_slice_tensor,
                stop_slice_tensor,
            )
            output_shape_tensor_neg = floor_divide(
                ctx,
                target,
                source_ir,
                name + "_div",
                output_shape_tensor_num,
                stride_slice_tensor,
            )
            output_shape_tensor = convert_binary_elementwise(
                ctx,
                target,
                source_ir,
                name + "_prod",
                trt.ElementWiseOperation.PROD,
                output_shape_tensor_neg,
                -1,
            )
            layer = ctx.net.add_slice(
                input, start=trt.Dims(), shape=trt.Dims(), stride=trt.Dims()
            )
            layer.set_input(1, start_slice_tensor)
            layer.set_input(2, output_shape_tensor)
            layer.set_input(3, stride_slice_tensor)
            return layer.get_output(0)

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

    # Configure the start, strides and output shape tensors
    start = tuple([0] * shape_rank)

    # stride[dim]=0 implies that dimension is being broadcasted.
    # stride should be 1 for all non-broadcasted dims
    stride = []
    input_tensor_shape = tuple(input_t.shape)
    for i, o in zip(input_tensor_shape, shape):
        # If input dim and target shape dim are static, broadcast if they are not equal
        # If input dim is known and target shape dim is dynamic we treat it as a broadcasted dim
        if (
            isinstance(i, int)
            and i != DYNAMIC_DIM
            and isinstance(o, int)
            and o != DYNAMIC_DIM
        ):
            stride.append(int(i == o))
        elif isinstance(i, int) and i != DYNAMIC_DIM and isinstance(o, TRTTensor):
            stride.append(0)
        else:
            # No broadcasting is happening. The output should have the same size as input at this dimension.
            stride.append(1)

    # Resolve dynamic dimensions in the target shape. These are not broadcasted dims.
    # The value at this dimension should be same as input.
    target_shape = []
    for i in range(shape_rank):
        if shape[i] == DYNAMIC_DIM:
            target_shape.append(
                get_shape(ctx, target, source_ir, name + f"_shape_dim{i}", input_t, i)
            )
        else:
            target_shape.append(shape[i])

    target_shape_t = target_shape
    # Handle dynamic shapes case where shape has dynamic dimension
    if any(isinstance(ele, TRTTensor) for ele in target_shape_t):
        target_shape_t = cat(
            ctx,
            target,
            source_ir,
            name + "_shape_concat",
            target_shape_t,
            0,
            cast_dtype=trt.int32,
        )
        start_tensor = cat(
            ctx,
            target,
            source_ir,
            name + "_start_concat",
            start,
            0,
            cast_dtype=trt.int32,
        )
        stride_tensor = cat(
            ctx,
            target,
            source_ir,
            name + "_stride_concat",
            stride,
            0,
            cast_dtype=trt.int32,
        )
        layer = ctx.net.add_slice(
            input_t, start=trt.Dims(), shape=trt.Dims(), stride=trt.Dims()
        )
        layer.set_input(1, start_tensor)
        layer.set_input(2, target_shape_t)
        layer.set_input(3, stride_tensor)
    else:
        layer = ctx.net.add_slice(
            input_t, start=start, shape=target_shape_t, stride=stride
        )

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

    if shape[dim] == DYNAMIC_DIM:
        # case when chunk dimension is dynamic
        size_dim = get_shape(
            ctx, target, source_ir, name + f"_shape_dim_to_chunk", input, dim
        )
        # implementing ceil operation to give us the chunk size
        chunk_size_num = convert_binary_elementwise(
            ctx,
            target,
            source_ir,
            name + "_sub_num_chunk",
            trt.ElementWiseOperation.SUB,
            size_dim,
            0,
        )
        chunk_size = ceil_divide(
            ctx,
            target,
            source_ir,
            name + "_div_chunk",
            chunk_size_num,
            chunks,
        )
        # implementing the ceil operation to give the chunk cnt
        # note that this can be 1 greater than the chunks
        # this is required to reduce it from 1D to 0D
        chunk_cnt = impl.reduce.sum(
            ctx, target, source_ir, name + "_chunk_cnt", chunks, 0, keepdim=False
        )

        ###################################outer loop, traverse start and end############################
        loop_one = ctx.net.add_loop()
        loop_one.add_trip_limit(chunk_cnt, trt.TripLimit.COUNT)
        start_tensor_i = get_trt_tensor(ctx, 0, f"{name}_initial_start")
        end_tensor_i = torch_trt_min(
            ctx,
            target,
            source_ir,
            name + "_initial_end",
            add(
                ctx,
                target,
                source_ir,
                name + "_chunk_end_initial_tensor",
                start_tensor,
                chunk_size,
            ),
            size_dim,
        )

        running_start_i = loop_one.add_recurrence(start_tensor_i)
        set_layer_name(running_start_i, target, f"{name}_running_start_i", source_ir)
        running_start_tensor_i = running_start_i.get_output(0)

        running_end_i = loop_one.add_recurrence(end_tensor_i)
        set_layer_name(running_end_i, target, f"{name}_running_end_i", source_ir)
        running_end_tensor_i = running_end_i.get_output(0)

        #####################################Method 1#######################################################
        # loop_two_output = slice_op(
        #     ctx,
        #     target,
        #     source_ir,
        #     f"{name}_slice",
        #     input,
        #     dim,
        #     running_start_tensor,
        #     running_end_tensor,
        #     1,
        # )

        ###################################Method 2#######################################################
        ###################################inner loop, traverse j between start and end############################

        loop_two = ctx.net.add_loop()
        loop_two.add_trip_limit(chunk_size_inner_loop, trt.TripLimit.COUNT)
        chunk_size_inner_loop = sub(
            ctx,
            target,
            source_ir,
            name + "_chunk_size_inner_loop",
            running_end_tensor_i,
            running_start_tensor_i,
        )

        running_start_j = loop_one.add_recurrence(running_start_tensor_i)
        set_layer_name(running_start_j, target, f"{name}_running_start_j", source_ir)
        running_start_tensor_i = running_start_j.get_output(0)

        iterator = loop_two.add_iterator(input, dim, reverse=False)
        sliced_input = iterator.get_output(0)

        loop_two_output = loop_two.add_loop_output(
            sliced_input, trt.LoopOutput.CONCATENATE, dim
        )
        set_layer_name(loop_two_output, target, f"{name}_loop_two_output", source_ir)
        loop_two_output.set_input(1, chunk_size_inner_loop)
        ###################################inner loop end##############################

        #####################################Method 3#############################################
        # constant_0 = get_trt_tensor(ctx, 0, f"{name}_constant_tensor_0")
        # size_dim = impl.reduce.sum(
        #      ctx, target, source_ir, name + "_inner_loop_chunk_cnt", size_dim, 0, keepdim=False
        # )
        # loop_two.add_trip_limit(size_dim, trt.TripLimit.COUNT)
        # rec2_j_tensor = loop_two.add_recurrence(constant_0)
        # set_layer_name(rec2_j_tensor, target, f"{name}_rec2_j_tensor", source_ir)
        # j_tensor = rec2_j_tensor.get_output(0)

        # # create a TRT Select layer
        # cond1 = impl.elementwise.ge(
        #     ctx, target, source_ir, f"{name}_ge_{time.time()}", j_tensor, running_start_tensor_i
        # )
        # cond2 = impl.elementwise.lt(
        #     ctx, target, source_ir, f"{name}_lt_{time.time()}", j_tensor, running_end_tensor_i
        # )
        # condition1 = impl.elementwise.logical_and(
        #     ctx, target, source_ir, f"{name}_and_{time.time()}", cond1, cond2
        # )
        # next_j = impl.elementwise.add(
        #     ctx, target, source_ir, f"{name}_j_tensor_add_1_{time.time()}", j_tensor, 1
        # )
        # rec2_j_tensor.set_input(1, next_j)
        # loop_out2 = loop_two.add_loop_output(condition1, trt.LoopOutput.CONCATENATE)
        # loop_out2.set_input(1, size_dim)
        # layer_non_zero = ctx.net.add_non_zero(loop_out2.get_output(0))
        # loop_two_output = ctx.net.add_gather(input, layer_non_zero.get_output(0), dim).get_output(0)
        ##################################inner loop end###########################################

        next_start_tensor_i = end_tensor_i
        next_end_tensor_i = torch_trt_min(
            ctx,
            target,
            source_ir,
            name + "_chunk_end_tensor_loop_assignment",
            add(
                ctx,
                target,
                source_ir,
                name + "_chunk_end_update_tensor",
                current_start_tensor,
                chunk_size,
            ),
            size_dim,
        )
        # current_bool_check = (current_start_tensor != size_dim)
        running_start_i.set_input(1, next_start_tensor_i)
        running_end_i.set_input(1, next_end_tensor_i)

        loop_one_output = loop_one.add_loop_output(
            loop_two_output.get_output(0), trt.LoopOutput.CONCATENATE, 0
        )
        set_layer_name(loop_one_output, target, f"{name}_loop_output", source_ir)
        loop_one_output.set_input(1, chunk_cnt)
        return loop_one_output.get_output(0)
        ##################################outer loop end#################################

    else:
        # case when chunk dimension is not dynamic
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
    if input_shape[dim] < 0:
        trip_limit = impl.shape.shape(
            ctx, target, source_ir, name + "_shape", input, dim
        )
        # the trip_limit has to be a 0D shape tensor, however this impl.shape.shape gives a 1D shape
        # for example if the trip limit is 3, it wants a tensor(3), not a tensor([3])
        # in order to reduce it from 1D to 0D, i have to use this impl.reduce.sum
        trip_limit = impl.reduce.sum(
            ctx, target, source_ir, name, trip_limit, 0, keepdim=False
        )
    else:
        axis = np.array(input_shape[dim])
        trip_limit = get_trt_tensor(ctx, axis, f"{name}_trip_limit")

    loop = ctx.net.add_loop()
    loop.add_trip_limit(trip_limit, trt.TripLimit.COUNT)
    iterator = loop.add_iterator(input, dim, reverse=False)
    data = iterator.get_output(0)
    if has_dynamic_shape(data.shape):
        data_shape = []
        for i in range(len(input_shape)):
            if i != dim:
                if input_shape[i] < 0:
                    data_shape.append(
                        impl.shape.shape(
                            ctx, target, source_ir, name + f"_{i}_shape", input, i
                        )
                    )
                else:
                    data_shape.append(input_shape[i])
        zero_trttensor = impl.full.full(
            ctx, target, source_ir, name + "_full", data_shape, 0.0
        )
    else:
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


def flip(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dims: Sequence[int],
) -> TRTTensor:
    start_slice = []
    output_shape = list(input.shape)
    stride_slice = []

    dynamic_shape = has_dynamic_shape(input.shape)

    shape = input.shape
    rank = len(shape)
    dims = get_positive_dim(dims, rank)

    for i in range(rank):
        if i in dims:
            if shape[i] == DYNAMIC_DIM:
                dim = get_shape(
                    ctx, target, source_ir, f"{name}_shape_dim_{i}", input, i
                )
                last_element_index = impl.elementwise.sub(
                    ctx, target, source_ir, f"{name}_sub_{i}", dim, 1
                )
                start_slice.append(last_element_index)
            else:
                start_slice.append(shape[i] - 1)
            stride_slice.append(-1)
        else:
            start_slice.append(0)
            stride_slice.append(1)

    layer = ctx.net.add_slice(
        input,
        start=[] if dynamic_shape else start_slice,
        shape=[] if dynamic_shape else output_shape,
        stride=stride_slice,
    )
    if dynamic_shape:
        output_shape = get_shape_with_dynamic_shape(
            ctx, target, source_ir, f"{name}_shape", output_shape, input
        )

        start_slice_tensor = cat(
            ctx,
            target,
            source_ir,
            f"{name}_start_slice_concat",
            start_slice,
            0,
        )
        layer.set_input(1, start_slice_tensor)
        layer.set_input(2, output_shape)

    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def diagonal(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    offset: int,
    dim1: int,
    dim2: int,
) -> TRTTensor:
    """
    This implementation is inspired by the reference implementation in PyTorch:
    https://github.com/pytorch/pytorch/blob/082251e76b93b277ff2791d0e2b64934add34644/torch/_refs/__init__.py#L4255
    """
    input_shape = input.shape
    num_dims = len(input_shape)

    # Adjust dimensions to be positive and canonicalize
    dim1 = get_positive_dim(dim1, num_dims)
    dim2 = get_positive_dim(dim2, num_dims)

    # Calculate the size of the diagonal
    if offset >= 0:
        diag_size = max(min(input_shape[dim1], input_shape[dim2] - offset), 0)
    else:
        diag_size = max(min(input_shape[dim1] + offset, input_shape[dim2]), 0)

    if diag_size == 0:
        raise ValueError("The size of the diagonal is non-positive.")

    strides = calculate_strides(input_shape)

    # Compute the storage offset
    storage_offset = 0
    if offset >= 0:
        storage_offset += offset * strides[dim2]
    else:
        storage_offset -= offset * strides[dim1]

    # Calculate new sizes and strides for as_strided
    sizes = [s for i, s in enumerate(input_shape) if i not in (dim1, dim2)]
    sizes.append(diag_size)

    input_strides = [s for i, s in enumerate(strides) if i not in (dim1, dim2)]
    new_stride = strides[dim1] + strides[dim2]
    input_strides.append(new_stride)

    # Use as_strided to get the diagonal elements
    diagonal_output = as_strided(
        ctx,
        target,
        source_ir,
        f"{name}_as_strided",
        input,
        sizes,
        input_strides,
        storage_offset,
    )

    return diagonal_output


def as_strided(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    size: Sequence[int],
    stride: Sequence[int],
    storage_offset: Optional[int],
) -> TRTTensor:
    # Ensure storage_offset is an integer before passing to nested
    if storage_offset is None:
        storage_offset = 0

    flatten_shape = flatten_dims(input, 0, -1)
    flatten_output = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_reshape_flatten_output", input, flatten_shape
    )

    indices = []

    # Recursive function to compute indices for as_strided operation
    def nested(
        rank: int, size: Sequence[int], stride: Sequence[int], current: int, dim: int
    ) -> None:
        if (
            dim == rank
        ):  # If the current dimension equals the rank, append the computed index
            indices.append(current)
            return
        for i in range(size[dim]):  # Recursively compute indices across dimensions
            nested(
                rank, size, stride, current + stride[dim] * i, dim + 1
            )  # Calculate the index for the current dimension and recursively explore further dimensions

    nested(len(size), size, stride, storage_offset, 0)

    indices = np.array(indices, dtype=np.int32)

    indices_tensor = get_trt_tensor(ctx, indices, f"{name}_indices")

    # Use gather to reorder elements based on computed indices
    gather_layer = ctx.net.add_gather(flatten_output, indices_tensor, axis=0)
    gather_output = gather_layer.get_output(0)

    # Reshape the gathered tensor to the desired size
    reshape_output = impl.shuffle.reshape(
        ctx,
        target,
        source_ir,
        f"{name}_reshape_gather_output",
        gather_output,
        tuple(size),
    )

    return reshape_output
