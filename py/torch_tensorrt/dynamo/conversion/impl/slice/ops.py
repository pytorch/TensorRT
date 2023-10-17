import math
from typing import Optional

import numpy as np
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
    start: int,
    stop: int,
    step: int,
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"slice_tensor received input {input} that is not part "
            "of the TensorRT region!"
        )

    ranks = len(input.shape) + (1 if ctx.net.has_implicit_batch_dimension else 0)
    dim = get_positive_dim(dim, ranks)
    dynamic_shape = has_dynamic_shape(input.shape)
    if ctx.net.has_implicit_batch_dimension:
        if dim == 0:
            raise RuntimeError(
                f"We do not support slice_tensor at batch dim when it's implicit, got {dim}!"
            )
        dim = dim - 1
    else:
        if dynamic_shape:
            # Check whether slice target dim is dynamic shape dim
            assert input.shape[dim] != -1, "Can't chunk on dynamic shape dimension!"
    start_int = start
    stop_int = stop
    if stop_int == 2**63 - 1:
        stop_int = input.shape[dim]
    step_int = step
    start_slice = [0] * len(input.shape)
    start_slice[dim] = start_int
    stride_slice = [1] * len(start_slice)
    stride_slice[dim] = step_int
    output_shape = list(input.shape)
    output_shape[dim] = math.ceil((stop_int - start_int) / step_int)

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
    if not isinstance(input_t, TRTTensor):
        raise RuntimeError(
            f"expand received input {input_t} that is not a TensorRT ITensor"
        )

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
    chunk_size = int(np.ceil(size_dim / chunks))
    result = []
    start = 0
    for i in range(chunks):
        end = start + chunk_size
        if end > size_dim:
            end = size_dim
        if start < end:
            result.append(
                slice_op(
                    ctx,
                    target,
                    source_ir,
                    f"{name}_slice{i}",
                    input,
                    dim,
                    start,
                    end,
                    1,
                )
            )
            start = end
        else:
            break

    return result
