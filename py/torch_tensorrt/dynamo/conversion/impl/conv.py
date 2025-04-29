from typing import Optional, Sequence, Union

import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    SourceIR,
    cast_trt_tensor,
    extend_attr_to_tuple,
    get_trt_tensor,
    to_torch,
)
from torch_tensorrt.fx.converters.converter_utils import (
    get_dyn_range,
    has_dynamic_shape,
    mark_as_int8_layer,
    set_layer_name,
)
from torch_tensorrt.fx.types import TRTTensor


def convNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    is_conv1d: bool,
    input: TRTTensor,
    weight: Union[TRTTensor, torch.Tensor, np.ndarray],
    bias: Optional[Union[TRTTensor, torch.Tensor, np.ndarray]],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]],
    dilation: Union[int, Sequence[int]],
    groups: Optional[int],
    output_padding: Union[int, Sequence[int]] = 0,
    scale: Optional[Union[torch.Tensor, float]] = None,
    zero_point: Optional[Union[torch.Tensor, float]] = None,
) -> TRTTensor:
    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for convolution."

    num_dims = len(input.shape) - 2
    if is_conv1d:
        # Apply an unsqueeze operation to transform the conv1d problem into conv2d
        input = impl.unsqueeze.unsqueeze(
            ctx, target, source_ir, name + "_unsqueeze_conv1d", input, -1
        )

    # Process bias terms
    if isinstance(bias, (torch.Tensor, np.ndarray)):
        bias = to_torch(bias, dtype=input.dtype)
        bias = get_trt_tensor(ctx, bias, f"{name}_bias")

    elif isinstance(bias, TRTTensor):
        bias = get_trt_tensor(ctx, bias, f"{name}_bias")

    elif bias is not None:
        raise RuntimeError(
            f"Convolution {name} has bias of type {type(bias)}, Expected Torch Tensor or TRT Tensor"
        )

    # Process weight terms
    if isinstance(weight, TRTTensor):
        weight = get_trt_tensor(ctx, weight, f"{name}_weight")
        # Append new dimension (unsqueeze) if the convolution is 1d
        if is_conv1d:
            weight = impl.unsqueeze.unsqueeze(
                ctx, target, source_ir, weight.name + "_unsqueeze_conv1d", weight, -1
            )
    elif isinstance(weight, (torch.Tensor, np.ndarray)):
        weight = to_torch(weight, dtype=input.dtype)
        # Append new dimension (unsqueeze) if the convolution is 1d
        if is_conv1d:
            weight = torch.unsqueeze(weight, -1)
        weight = get_trt_tensor(ctx, weight, f"{name}_weight")

    else:
        raise RuntimeError(
            f"Convolution {name} has weight of type {type(weight)}, Expect Optional[Tensor]"
        )

    # add conv layer
    conv_layer = ctx.net.add_convolution_nd(
        input=input,
        num_output_maps=weight.shape[0],
        kernel_shape=weight.shape[2:],
        kernel=trt.Weights() if isinstance(weight, TRTTensor) else weight,
        bias=trt.Weights() if isinstance(bias, TRTTensor) else bias,
    )
    # If the weight is a TRTTensor, set it as an input of the layer
    if isinstance(weight, TRTTensor):
        weight = cast_trt_tensor(ctx, weight, input.dtype, name)
        conv_layer.set_input(1, weight)

    # If the bias is a TRTTensor, set it as an input of the layer
    if isinstance(bias, TRTTensor):
        bias = cast_trt_tensor(ctx, bias, input.dtype, name)
        conv_layer.set_input(2, bias)

    # Cast certain fields to tuples, in accordance with TRT requirements
    if isinstance(padding, int):
        padding = (padding,) * num_dims
    elif isinstance(padding, (list, tuple)):
        padding = tuple(padding)
        if len(padding) == 1:
            padding = (padding[0],) * num_dims

    if isinstance(stride, int):
        stride = (stride,) * num_dims
    elif isinstance(stride, (list, tuple)):
        stride = tuple(stride)
        if len(stride) == 1:
            stride = (stride[0],) * num_dims

    if isinstance(dilation, int):
        dilation = (dilation,) * num_dims
    elif isinstance(dilation, (list, tuple)):
        dilation = tuple(dilation)
        if len(dilation) == 1:
            dilation = (dilation[0],) * num_dims

    # Expand parameters manually for Conv1D computations
    if is_conv1d:
        padding = (tuple(padding) + (0,)) if padding is not None else padding
        stride = extend_attr_to_tuple(stride, 2) if stride is not None else stride
        dilation = (
            extend_attr_to_tuple(dilation, 2) if dilation is not None else dilation
        )

    set_layer_name(conv_layer, target, name, source_ir)

    # Set relevant attributes of convolution layer
    if padding is not None:
        conv_layer.padding_nd = padding
    if stride is not None:
        conv_layer.stride_nd = stride
    if dilation is not None:
        conv_layer.dilation_nd = dilation
    if groups is not None:
        conv_layer.num_groups = groups

    # Handle quantization cases
    if scale is not None and zero_point is not None:
        # Assume the dtype of activation is torch.quint8
        mark_as_int8_layer(conv_layer, get_dyn_range(scale, zero_point, torch.quint8))

    result = conv_layer.get_output(0)

    if is_conv1d:
        # Apply a squeeze operation to transform the conv2d problem back into conv1d
        result = impl.squeeze.squeeze(
            ctx, target, source_ir, name + "_squeeze_conv1d", result, -1
        )

    return result
