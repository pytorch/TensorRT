# flake8: noqa
import logging
import math
import operator
import warnings
from typing import cast, Dict, Optional, Sequence, Tuple, Union

import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
from torch_tensorrt.fx.converters import acc_ops_converters

from ..converter_registry import tensorrt_converter

from ..types import *  # noqa: F403
from torch.fx.immutable_collections import immutable_list
from torch.fx.node import Argument, Target

from ..utils import get_dynamic_dims, torch_dtype_from_trt, torch_dtype_to_trt

from .converter_utils import *  # noqa: F403
import torch_tensorrt.fx.tracer.acc_tracer.acc_utils as acc_utils
from fx2trt_ops_converter_utils import *

_LOGGER: logging.Logger = logging.getLogger(__name__)

@tensorrt_converter(torch.ops.aten.add.Tensor)
def convert_add(network, target, args, kwargs, name):
    input_a = args[0]
    input_b = args[1]
    input_a_trt, input_b_trt = add_missing_trt_tensors(network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(network, [input_a_trt, input_b_trt], len(output.shape))
    layer = network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.SUM)
    output = layer.get_output(0) 
    return output

@tensorrt_converter(torch.ops.aten.leaky_relu)
def convert_leaky_relu(network, target, args, kwargs, name):
    input = get_arg(args, kwargs, 'input', pos=0, default=None)
    negative_slope = get_arg(args, kwargs, 'negative_slope', pos=1, default=0.01)
    input_trt = add_missing_trt_tensors(network, [input])[0]
    layer = network.add_activation(input_trt, trt.ActivationType.LEAKY_RELU)
    layer.alpha = negative_slope
    output = layer.get_output(0)
    return output

@tensorrt_converter(torch.ops.aten._adaptive_avg_pool2d.default)
@tensorrt_converter(torch.ops.aten.adaptive_avg_pool2d)
def convert_adaptive_avg_pool2d(network, target, args, kwargs, name):
    method_args = (network, torch.nn.AdaptiveAvgPool2d(args[1]), args[0])
    output = convert_AdaptiveAvgPool2d(method_args)

@tensorrt_converter(torch.ops.aten._adaptive_avg_pool3d.default)
def convert_adaptive_avg_pool3d(network, target, args, kwargs, name):
    method_args = (network, torch.nn.AdaptiveAvgPool3d(args[1]), args[0])
    output = convert_AdaptiveAvgPool3d(method_args)

#FIXME: check if this is required
@tensorrt_converter(torch.ops.aten.mean.dim)
def convert_avg_pool(network, target, args, kwargs, name):
    # parse args
    input = get_arg(args, kwargs, 'input', pos=0, default=None)
    kernel_size = get_arg(args, kwargs, 'kernel_size', pos=1, default=None)
    stride = get_arg(args, kwargs, 'stride', pos=2, default=None)
    padding = get_arg(args, kwargs, 'padding', pos=3, default=0)
    ceil_mode = get_arg(args, kwargs,'ceil_mode', pos=4, default=False)
    count_include_pad = get_arg(args, kwargs, 'count_include_pad', pos=5, default=True)
    
    # get input trt tensor (or create constant if it doesn't exist)
    input_trt = add_missing_trt_tensors(network, [input])[0]
    input_dim = input.dim() - 2

    # get kernel size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * input_dim

    # get stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * input_dim

    # get padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * input_dim

    layer = network.add_pooling_nd(
        input=input_trt, type=trt.PoolingType.AVERAGE, window_size=kernel_size)
    
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.average_count_excludes_padding = not count_include_pad
    
    if ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    output = layer.get_output(0)
    return output

@tensorrt_converter(torch.ops.aten.batch_norm)
def convert_batch_norm(network, target, args, kwargs, name):
    input = get_arg(args, kwargs, 'input', pos=0, default=None) 
    running_mean = get_arg(args, kwargs, 'running_mean', pos=1, default=None) 
    running_var = get_arg(args, kwargs, 'running_var', pos=2, default=None) 

    weight = get_arg(args, kwargs, 'weight', pos=3, default=None) 
    bias = get_arg(args, kwargs, 'bias', pos=4, default=None) 
    eps = get_arg(args, kwargs, 'eps', pos=7, default=10e-6) 

    input_trt = add_missing_trt_tensors(network, [input])[0]
    
    
    scale = weight.detach().cpu().numpy() / np.sqrt(running_var.detach().cpu().numpy() + eps)
    bias = bias.detach().cpu().numpy() - running_mean.detach().cpu().numpy() * scale
    power = np.ones_like(scale)

    layer = network.add_scale_nd(input_trt, trt.ScaleMode.CHANNEL, bias, scale, power, 1)
    output = layer.get_output(0)
    return output

@tensorrt_converter(torch.ops.aten.cat.default)
def convert_cat(network, target, args, kwargs, name):
    inputs = get_arg(args, kwargs, 'input', pos=0, default=None)
    dim = get_arg(args, kwargs, 'dim', pos=1, default=0)

    # Reverse negative dims.
    if dim < 0:
        dim = len(inputs[0].shape) - abs(dim)

    trt_inputs = add_missing_trt_tensors(ctx.network, inputs)
    trt_inputs = broadcast_trt_tensors(ctx.network, trt_inputs, len(output.shape))

    layer = network.add_concatenation(inputs=trt_inputs)
    layer.axis = dim
    output = layer.get_output(0)
    return output

@tensorrt_converter(torch.ops.aten.convolution.default)
def convert_cat(network, target, args, kwargs, name):
    module = args[0]
    input = args[1]
    input_trt = add_missing_trt_tensors(network, [input])[0]

    input_dim = input.dim() - 2

    kernel_size = module.kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * input_dim

    stride = module.stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * input_dim

    padding = module.padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * input_dim

    dilation = module.dilation
    if not isinstance(dilation, tuple):
        dilation = (dilation, ) * input_dim

    kernel = module.weight.detach().cpu().numpy()
    
    bias = None #trt.Weights(torch_dtype_to_trt(module.weight.dtype))
    if module.bias is not None:
        bias = module.bias.detach().cpu().numpy()

    layer = network.add_convolution_nd(
        input=input_trt,
        num_output_maps=module.out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias)
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.dilation_nd = dilation

    if module.groups is not None:
        layer.num_groups = module.groups

    output = layer.get_output(0)
    return output
