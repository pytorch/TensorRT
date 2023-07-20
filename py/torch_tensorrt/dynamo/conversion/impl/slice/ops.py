from typing import Optional, cast
import math

from torch.fx.node import Target

from torch_tensorrt.fx.types import TRTNetwork, TRTTensor, Shape
from torch_tensorrt.dynamo.conversion import SourceIR
from torch_tensorrt.fx.converters.converter_utils import (
    get_positive_dim,
    has_dynamic_shape,
    broadcast,
    get_trt_tensor,
)
from torch_tensorrt.dynamo.conversion.impl.slice.base import slice


def slice_op(
    network: TRTNetwork,
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

    ranks = len(input.shape) + (1 if network.has_implicit_batch_dimension else 0)
    dim = get_positive_dim(cast(int, dim), ranks)
    dynamic_shape = has_dynamic_shape(input.shape)
    if network.has_implicit_batch_dimension:
        if dim == 0:
            raise RuntimeError(
                f"We do not support slice_tensor at batch dim when it's implicit, got {dim}!"
            )
        dim = dim - 1
    else:
        if dynamic_shape:
            # Check whether slice target dim is dynamic shape dim
            assert input.shape[dim] != -1, "Can't chunk on dynamic shape dimension!"
    start_int = cast(int, start)
    if stop_int == 2**63 - 1:
        stop_int = input.shape[dim]
    step_int = cast(int, step)
    start = [0] * len(input.shape)
    start[dim] = start_int
    stride = [1] * len(start)
    stride[dim] = step_int
    output_shape = list(input.shape)
    output_shape[dim] = math.ceil((stop_int - start_int) / step_int)

    return slice(network, target, source_ir, name, input, start, output_shape, stride)


def expand(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    sizes: Shape,
) -> TRTTensor:
    shape = list(sizes)

    input_val = get_trt_tensor(network, input, f"{name}_input")

    if network.has_implicit_batch_dimension:
        shape = shape[1:]

    ranks = len(input_val.shape)
    # TRT does not support different dimension size
    # though this condition is not seen in the case of bmm
    # where input_t and shape dimensions are not equal
    assert len(shape) >= ranks
    if len(shape) != ranks:
        shape_tuple = tuple([0] * len(shape))
        shape_tensor = get_trt_tensor(network, input, f"{name}_shape")
        input_val, shape_tensor = broadcast(
            network, input_val, shape_tensor, f"{name}_input_val", f"{name}_shape_val"
        )
        ranks = len(shape)

    inshape = tuple(input_val.shape)
    shape = tuple(shape)
    start = tuple([0] * ranks)
    stride = tuple(
        [int(i == o) for i, o in zip(inshape, shape)]
    )  # stride == 1 if dimensions match, 0 otherwise
    return slice(network, target, source_ir, name, input_val, start, shape, stride)
