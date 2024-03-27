from typing import Optional, Sequence, Union

import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    flatten_dims,
    get_positive_dim,
)
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTTensor


def permute(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    permutation: Sequence[int],
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"permute received input {input} that is not a TensorRT ITensor"
        )

    permutation = get_positive_dim(permutation, len(input.shape))

    layer = ctx.net.add_shuffle(input)
    layer.second_transpose = tuple(permutation)
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def roll(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    shifts: Union[int, Sequence[int]],
    dims: Union[int, Sequence[int]],
) -> TRTTensor:
    shape = input.shape
    if isinstance(shifts, int):
        shifts = [shifts]
    if isinstance(dims, int):
        dims = [dims]

    if dims != []:
        rank = len(shape)
        start = [0] * rank
        stride = [1] * rank
        for i in range(len(dims)):
            d = dims[i]
            s = shifts[i]
            start[d] += get_positive_dim(
                -s, shape[d]
            )  # in case that dims has multiple same dim

        layer = ctx.net.add_slice(
            input,
            start=start,
            shape=shape,
            stride=stride,
        )
        layer.mode = trt.SampleMode.WRAP
        set_layer_name(layer, target, f"{name}_slice_wrap", source_ir)
        return layer.get_output(0)

    else:
        flatten_shape = flatten_dims(input, 0, -1)
        output = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_reshape", input, flatten_shape
        )
        start = [get_positive_dim(-shifts[0], output.shape[0])]
        stride = [1]
        layer = ctx.net.add_slice(
            output,
            start=start,
            shape=flatten_shape,
            stride=stride,
        )
        layer.mode = trt.SampleMode.WRAP
        set_layer_name(layer, target, f"{name}_slice_wrap", source_ir)
        output = layer.get_output(0)
        output = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_reshape_back", output, shape
        )
        return output
