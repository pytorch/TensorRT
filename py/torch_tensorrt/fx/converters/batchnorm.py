import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch

from ..converter_registry import tensorrt_converter
from .converter_utils import to_numpy


def common_batchnorm(network, mod, input_val, layer_name, is_quantized):
    scale = to_numpy(mod.weight) / np.sqrt(to_numpy(mod.running_var) + mod.eps)
    bias = to_numpy(mod.bias) - to_numpy(mod.running_mean) * scale
    power = np.ones_like(scale)

    layer = network.add_scale(input_val, trt.ScaleMode.CHANNEL, bias, scale, power)
    layer.name = layer_name

    return layer.get_output(0)


@tensorrt_converter(torch.nn.modules.batchnorm.BatchNorm2d)
def batchnorm2d(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"BatchNorm2d received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    return common_batchnorm(network, submod, input_val, layer_name, is_quantized=False)


@tensorrt_converter(torch.nn.quantized.modules.batchnorm.BatchNorm2d)
def quantized_batchnorm2d(network, submod, args, kwargs, layer_name):
    input_val = args[0]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"Quantized BatchNorm2d received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    return common_batchnorm(network, submod, input_val, layer_name, is_quantized=True)
