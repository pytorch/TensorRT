# @manual=//deeplearning/trt/python:py_tensorrt
import numpy as np
import tensorrt as trt
import torch

from ..converter_registry import tensorrt_converter

from .converter_utils import (
    extend_mod_attr_to_tuple,
    get_dyn_range,
    mark_as_int8_layer,
    to_numpy,
)


def common_conv(network, mod, dimension, input_val, layer_name, is_quantized):
    if mod.padding_mode != "zeros":
        raise RuntimeError(f"Only support padding mode: zeros, got {mod.padding_mode}.")

    kernel_size = extend_mod_attr_to_tuple(mod, "kernel_size", dimension)
    stride = extend_mod_attr_to_tuple(mod, "stride", dimension)
    padding = extend_mod_attr_to_tuple(mod, "padding", dimension)
    dilation = extend_mod_attr_to_tuple(mod, "dilation", dimension)

    kernel = to_numpy(mod.weight() if is_quantized else mod.weight)
    bias = to_numpy(mod.bias() if is_quantized else mod.bias)

    if dimension == 1:
        # Append unsqueeze before conv2d to calculate conv1d
        unsqueeze_layer = network.add_shuffle(input=input_val)
        unsqueeze_layer.reshape_dims = (*input_val.shape, 1)
        unsqueeze_layer.name = f"{layer_name}_unsqueeze"
        input_val = unsqueeze_layer.get_output(0)

        kernel = np.expand_dims(kernel, -1)
        kernel_size = kernel.shape[2:]
        if bias is not None:
            bias = bias[None]
        stride = (stride[0], 1)
        padding = (padding[0], 0)
        dilation = (dilation[0], 1)
    layer = network.add_convolution_nd(
        input=input_val,
        num_output_maps=mod.out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias,
    )
    layer.name = layer_name
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.dilation_nd = dilation
    layer.num_groups = mod.groups

    if is_quantized:
        # Assume the dtype of activation is torch.quint8
        mark_as_int8_layer(
            layer, get_dyn_range(mod.scale, mod.zero_point, torch.quint8)
        )

    result = layer.get_output(0)
    if dimension == 1:
        # Append squeeze after conv2d to calculate conv1d
        squeeze_layer = network.add_shuffle(input=result)
        squeeze_layer.reshape_dims = tuple(result.shape[:-1])
        squeeze_layer.name = f"{layer_name}_squeeze"
        result = squeeze_layer.get_output(0)

    return result


def common_conv_relu(network, mod, dimension, input_val, layer_name, is_quantized):
    conv_output = common_conv(
        network,
        mod,
        dimension=2,
        input_val=input_val,
        layer_name=f"{layer_name}_conv",
        is_quantized=is_quantized,
    )

    layer = network.add_activation(input=conv_output, type=trt.ActivationType.RELU)
    layer.name = f"{layer_name}_relu"

    if is_quantized:
        mark_as_int8_layer(layer, conv_output.dynamic_range)

    return layer.get_output(0)


@tensorrt_converter(torch.nn.modules.conv.Conv1d)
def conv1d(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"Conv1d received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if layer_name is None:
        raise RuntimeError("layer name is none")
    return common_conv(
        network,
        submod,
        dimension=1,
        input_val=input_val,
        layer_name=layer_name,
        is_quantized=False,
    )


@tensorrt_converter(torch.nn.modules.conv.Conv2d)
def conv2d(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"Conv2d received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    return common_conv(
        network,
        submod,
        dimension=2,
        input_val=input_val,
        layer_name=layer_name,
        is_quantized=False,
    )


@tensorrt_converter(torch.nn.modules.conv.Conv3d)
def conv3d(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"Conv3d received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    return common_conv(
        network,
        submod,
        dimension=3,
        input_val=input_val,
        layer_name=layer_name,
        is_quantized=False,
    )


@tensorrt_converter(torch.nn.quantized.modules.conv.Conv2d)
def quantized_conv2d(network, submod, args, kwargs, layer_name):
    input_val = args[0]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"Quantized Conv2d received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    return common_conv(
        network,
        submod,
        dimension=2,
        input_val=input_val,
        layer_name=layer_name,
        is_quantized=True,
    )


@tensorrt_converter(torch.nn.intrinsic.quantized.modules.ConvReLU2d)
def quantized_conv_relu2d(network, submod, args, kwargs, layer_name):
    input_val = args[0]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"Quantized ConvReLU2d received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    return common_conv_relu(
        network,
        submod,
        dimension=2,
        input_val=input_val,
        layer_name=f"{layer_name}_conv",
        is_quantized=True,
    )
