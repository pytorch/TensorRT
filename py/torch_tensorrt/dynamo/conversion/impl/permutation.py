from typing import Dict, Optional, Sequence, Union

import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_positive_dim,
    get_trt_tensor,
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl.shape import get_shape_with_dynamic_shape
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


def calc_start_by_static_shape(
    input: TRTTensor,
    shift_dict: Dict[int, int],
) -> Sequence[int]:
    start = [0] * len(input.shape)
    for d, s in shift_dict.items():
        start[d] = get_positive_dim(-s, input.shape[d])
    return start


def calc_start_by_dynamic_shape(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    shift_dict: Dict[int, int],
) -> TRTTensor:
    start = []
    default_tensor = get_trt_tensor(ctx, 0, name + "_get_0")
    for i in range(len(input.shape)):
        start.append(default_tensor)
    for d, s in shift_dict.items():
        dim_length = impl.shape.shape(ctx, target, source_ir, name + "_shape", input, d)
        start[d] = impl.elementwise.sub(
            ctx, target, source_ir, name + "_sub", dim_length, s
        )
    concat_layer = ctx.net.add_concatenation(start)
    concat_layer.axis = 0
    set_layer_name(concat_layer, target, f"{name}_gather", source_ir)
    return concat_layer.get_output(0)


def roll(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    shifts: Union[int, Sequence[int]],
    dims: Union[int, Sequence[int]],
) -> TRTTensor:
    if isinstance(shifts, int):
        shifts = [shifts]
    if isinstance(dims, int):
        dims = [dims]

    is_dynamic_shape = has_dynamic_shape(input.shape)
    if all(isinstance(shift, TRTTensor) for shift in shifts):
        is_dynamic_shift = False
    else:
        is_dynamic_shift = True

    shift_dict = {}
    if dims == []:
        shift_dict[1] = shifts[0]
    else:
        # preprocess dims, in case that dims has multiple same dim
        # for example shifts:[1, 2, 1], dims: [1, 0, 1]
        # can be simplified to shifts: [2, 2], dims: [1, 0]
        for shift, dim in zip(shifts, dims):
            if dim in shift_dict:
                shift_dict[dim] += shift
            else:
                shift_dict[dim] = shift

    # handle static shape for the input tensor and shifts:
    if not is_dynamic_shape and not is_dynamic_shift:
        orignal_shape = input.shape
        if dims == []:
            # flatten input tensor
            input = impl.shuffle.reshape(
                ctx, target, source_ir, name + "_reshape", input, (1, -1)
            )
        start = calc_start_by_static_shape(input, shift_dict)
        stride = [1] * len(input.shape)
        slice_layer = ctx.net.add_slice(
            input,
            start=start,
            shape=input.shape,
            stride=stride,
        )
        slice_layer.mode = trt.SampleMode.WRAP
        set_layer_name(slice_layer, target, f"{name}_slice_wrap", source_ir)
        output = slice_layer.get_output(0)
        if dims == []:
            # reshape back
            output = impl.shuffle.reshape(
                ctx, target, source_ir, name + "_reshape_back", output, orignal_shape
            )
    else:
        # handle dynammic shape for the input tensor and shifts
        orignal_input = input
        if dims == []:
            # flatten the input tensor
            input = impl.shuffle.reshape(
                ctx, target, source_ir, f"{name}_reshape", input, (1, -1)
            )
        start = calc_start_by_dynamic_shape(
            ctx, target, source_ir, name + "_calc", input, shift_dict
        )
        stride = [1] * len(input.shape)
        slice_layer = ctx.net.add_slice(
            input,
            start=[],
            shape=[],
            stride=stride,
        )
        slice_layer.set_input(1, start)
        slice_layer.set_input(
            2,
            get_shape_with_dynamic_shape(
                ctx, target, source_ir, name + "_dynamic_shape", input.shape, input
            ),
        )
        slice_layer.mode = trt.SampleMode.WRAP
        set_layer_name(slice_layer, target, f"{name}_slice_wrap", source_ir)
        output = slice_layer.get_output(0)
        if dims == []:
            # reshape back to the original shape
            shape_back = get_shape_with_dynamic_shape(
                ctx,
                target,
                source_ir,
                name + "_shape_back",
                orignal_input.shape,
                orignal_input,
            )
            shape_layer = ctx.net.add_shuffle(output)
            shape_layer.set_input(1, shape_back)
            set_layer_name(shape_layer, target, name + "_reshape_back", source_ir)
            output = shape_layer.get_output(0)

    return output
