from typing import Optional, Sequence, Union

import numpy as np
import torch_tensorrt.dynamo.conversion.impl as impl
from torch.fx.node import Target
from torch_tensorrt import _enums
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    SourceIR,
    cast_trt_tensor,
    flatten_dims,
    get_trt_tensor,
    set_layer_name,
)
from torch_tensorrt.fx.types import TRTTensor

import tensorrt as trt


def reshape(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    shape: Sequence[int],
) -> TRTTensor:
    layer = ctx.net.add_shuffle(input)
    if all(isinstance(s, int) for s in shape):
        layer.reshape_dims = tuple(shape)
    else:
        # Convert all the dimensions to trt Tensors.
        trt_shape = []

        for i, s in enumerate(shape):
            if isinstance(s, TRTTensor):
                dim_int32 = cast_trt_tensor(
                    ctx,
                    s,
                    _enums.dtype.int32,
                    name + f"_int32_casted_{i}",
                )
                trt_shape.append(dim_int32)
            else:
                a = get_trt_tensor(ctx, s, f"{name}_{i}")
                trt_shape.append(a)
        shape_layer = ctx.net.add_concatenation(inputs=trt_shape)
        shape_layer.axis = 0
        shape_layer.name = f"{name}_output_shape"
        layer.set_input(1, shape_layer.get_output(0))

    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def pixel_shuffle(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    upscale_factor: int,
) -> TRTTensor:
    shape = input.shape
    in_channels, in_height, in_width = shape[-3:]
    out_channels = in_channels // (upscale_factor**2)
    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor
    new_shape = shape[:-3] + (
        out_channels,
        upscale_factor,
        upscale_factor,
        in_height,
        in_width,
    )
    reshaped_tensor = reshape(
        ctx, target, source_ir, f"{name}_reshape1", input, new_shape
    )
    rank = len(shape)
    permute_shape = list(range(rank))
    permute_shape.insert(-2, rank)
    permute_shape.insert(-1, rank + 1)
    permuted_tensor = impl.permutation.permute(
        ctx, target, source_ir, f"{name}_permute", reshaped_tensor, permute_shape
    )
    return reshape(
        ctx,
        target,
        source_ir,
        f"{name}_reshape2",
        permuted_tensor,
        shape[:-3] + (out_channels, out_height, out_width),
    )


def pixel_unshuffle(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    downscale_factor: int,
) -> TRTTensor:
    shape = input.shape
    in_channels, in_height, in_width = shape[-3:]
    out_channels = in_channels * (downscale_factor**2)
    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor
    new_shape = shape[:-3] + (
        in_channels,
        out_height,
        downscale_factor,
        out_width,
        downscale_factor,
    )
    reshaped_tensor = reshape(
        ctx, target, source_ir, f"{name}_reshape1", input, new_shape
    )
    rank = len(new_shape)
    permute_shape = tuple(range(rank - 5)) + (
        rank - 5,  # in_channels
        rank - 3,  # downscale_factor
        rank - 1,  # downscale_factor
        rank - 4,  # out_height
        rank - 2,  # out_width
    )
    permuted_tensor = impl.permutation.permute(
        ctx, target, source_ir, f"{name}_permute", reshaped_tensor, permute_shape
    )
    return reshape(
        ctx,
        target,
        source_ir,
        f"{name}_reshape2",
        permuted_tensor,
        shape[:-3] + (out_channels, out_height, out_width),
    )


def resize(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    sizes: Sequence[int],
) -> TRTTensor:
    input_np_dtype = _enums.dtype._from(input.dtype).to(np.dtype)
    input_val = get_trt_tensor(ctx, input, f"{name}_input")

    # Calculate the total number of elements for new and current shape
    new_num_elements = np.prod(sizes)
    current_num_elements = np.prod(input_val.shape)

    if new_num_elements > current_num_elements:
        # Create a padding tensor with the required size and initialize new elements with zeros
        padding_size = new_num_elements - current_num_elements
        padding_tensor = ctx.net.add_constant(
            (padding_size,), trt.Weights(np.zeros(padding_size, dtype=input_np_dtype))
        ).get_output(0)

        # Flatten input tensor to 1D for concatenation
        flatten_shape = flatten_dims(input_val, 0, -1)
        flattened_input = reshape(
            ctx, target, source_ir, f"{name}_flatten_input", input_val, flatten_shape
        )

        # Concatenate the flattened input tensor and padding tensor
        reshaped_tensor = impl.cat.cat(
            ctx,
            target,
            source_ir,
            f"{name}_cat",
            [flattened_input, padding_tensor],
            dim=0,
        )
    elif new_num_elements < current_num_elements:
        # Flatten input tensor to 1D for slicing
        flatten_shape = flatten_dims(input_val, 0, -1)
        flattened_input = reshape(
            ctx, target, source_ir, f"{name}_flatten_input", input_val, flatten_shape
        )

        # Slice the flattened input tensor to the desired number of elements
        slice_layer = ctx.net.add_slice(flattened_input, [0], [new_num_elements], [1])
        reshaped_tensor = slice_layer.get_output(0)
    else:
        reshaped_tensor = input_val

    # Reshape the final output tensor to the target sizes
    resized_output = reshape(
        ctx, target, source_ir, f"{name}_final_reshape", reshaped_tensor, sizes
    )

    return resized_output
