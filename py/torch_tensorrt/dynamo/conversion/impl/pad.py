from typing import Optional, Sequence, Union

import numpy as np
import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor
from torch_tensorrt.fx.converters.converter_utils import (
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.fx.types import TRTTensor

"""
Note: IPaddingLayer is deprecated in TensorRT 8.2 and will be removed in TensorRT 10.0.
Use ISliceLayer to pad the tensor, which supports new non-constant, reflects padding
mode and clamp, and supports padding output with dynamic shape.
"""


def constant_padNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    pad: Sequence[int],
    value: Union[int, float] = 0,
) -> TRTTensor:
    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for padding."

    rank = len(input.shape)

    if len(pad) // 2 > rank:
        raise RuntimeError(
            f"Trying to pad last {len(pad) // 2} dimension but the input only has {rank} dimension."
        )

    start_list = [0] * rank
    new_shape = list(input.shape)

    for i in range(0, len(pad) // 2):
        start_list[-i - 1] = -pad[i * 2]
        new_shape[-i - 1] += pad[i * 2] + pad[i * 2 + 1]

    stride_list = [1] * rank
    layer = ctx.net.add_slice(
        input,
        start=tuple(start_list),
        shape=tuple(new_shape),
        stride=tuple(stride_list),
    )
    value_const = get_trt_tensor(ctx, value, f"{name}_value", input.dtype)
    layer.set_input(4, value_const)
    layer.mode = trt.SampleMode.FILL

    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def reflection_padNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    padding: Sequence[int],
) -> TRTTensor:
    rank = len(input.shape)

    if len(padding) // 2 > rank:
        raise RuntimeError(
            f"Trying to pad last {len(padding) // 2} dimensions but the input only has {rank} dimensions."
        )

    input_shape_tensor = ctx.net.add_shape(input).get_output(0)
    new_shape_tensor = input_shape_tensor

    start_list = [0] * rank
    stride_list = [1] * rank

    for i in range(len(padding) // 2):
        dim_index = rank - (i + 1)
        pad_before = padding[i * 2]
        pad_after = padding[i * 2 + 1]

        pad_sum = get_trt_tensor(
            ctx, pad_before + pad_after, f"{name}_pad_sum_{i}", dtype=np.int64
        )
        dim_shape = ctx.net.add_slice(
            input_shape_tensor,
            start=(dim_index,),
            shape=(1,),
            stride=(1,),
        ).get_output(0)

        new_dim_shape = impl.elementwise.add(
            ctx, target, source_ir, f"{name}_shape_dim_{i}", dim_shape, pad_sum
        )
        start_list[dim_index] = -pad_before

        slices = []
        for j in range(rank):
            if j == dim_index:
                slices.append(new_dim_shape)
            else:
                slices.append(
                    ctx.net.add_slice(
                        new_shape_tensor,
                        start=(j,),
                        shape=(1,),
                        stride=(1,),
                    ).get_output(0)
                )
        new_shape_tensor = ctx.net.add_concatenation(slices).get_output(0)

    start_tensor = get_trt_tensor(
        ctx,
        np.array(start_list, dtype=np.int64),
        f"{name}_start_tensor",
        dtype=np.int64,
    )

    stride_tensor = get_trt_tensor(
        ctx,
        np.array(stride_list, dtype=np.int64),
        f"{name}_stride_tensor",
        dtype=np.int64,
    )

    layer = ctx.net.add_slice(
        input, start=trt.Dims(), shape=trt.Dims(), stride=trt.Dims()
    )
    layer.set_input(1, start_tensor)
    layer.set_input(2, new_shape_tensor)
    layer.set_input(3, stride_tensor)
    layer.mode = trt.SampleMode.REFLECT

    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def replication_padNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    padding: Sequence[int],
) -> TRTTensor:
    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for padding."

    rank = len(input.shape)

    if len(padding) // 2 > rank:
        raise RuntimeError(
            f"Trying to pad last {len(padding) // 2} dimension but the input only has {rank} dimension."
        )

    start_list = [0] * rank
    new_shape = list(input.shape)

    for i in range(0, len(padding) // 2):
        start_list[-i - 1] = -padding[i * 2]
        new_shape[-i - 1] += padding[i * 2] + padding[i * 2 + 1]

    stride_list = [1] * rank
    layer = ctx.net.add_slice(
        input,
        start=tuple(start_list),
        shape=tuple(new_shape),
        stride=tuple(stride_list),
    )
    layer.mode = trt.SampleMode.CLAMP

    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def circular_padNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    pad: Sequence[int],
) -> TRTTensor:
    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for padding."

    rank = len(input.shape)

    if len(pad) // 2 > rank:
        raise RuntimeError(
            f"Trying to pad last {len(pad) // 2} dimension but the input only has {rank} dimension."
        )

    start_list = [0] * rank
    new_shape = list(input.shape)

    for i in range(0, len(pad) // 2):
        start_list[-i - 1] = -pad[i * 2]
        new_shape[-i - 1] += pad[i * 2] + pad[i * 2 + 1]

    stride_list = [1] * rank
    layer = ctx.net.add_slice(
        input,
        start=tuple(start_list),
        shape=tuple(new_shape),
        stride=tuple(stride_list),
    )
    layer.mode = trt.SampleMode.WRAP

    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def pad(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    pad: Sequence[int],
    mode: str = "constant",
    value: Optional[float] = None,
) -> TRTTensor:
    if mode == "constant":
        return constant_padNd(
            ctx,
            target,
            source_ir,
            f"{name}_{mode}",
            input,
            pad,
            value if value is not None else 0,
        )
    elif mode == "reflect":
        return reflection_padNd(ctx, target, source_ir, f"{name}_{mode}", input, pad)
    elif mode == "replicate":
        return replication_padNd(ctx, target, source_ir, f"{name}_{mode}", input, pad)
    elif mode == "circular":
        return circular_padNd(ctx, target, source_ir, f"{name}_{mode}", input, pad)
    else:
        raise RuntimeError(
            f'We currently only support for `mode` in ["constant", "reflect", "replicate", "circular"], but got {mode}'
        )
