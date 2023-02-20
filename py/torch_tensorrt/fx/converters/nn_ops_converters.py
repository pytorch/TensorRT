import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch

from ..converter_registry import tensorrt_converter

from .converter_utils import mark_as_int8_layer
import activation

@tensorrt_converter(torch.nn.functional.relu)
@tensorrt_converter(torch.nn.modules.activation.ReLU)
def relu(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    return activation.add_relu(network,"tensorrt", kwargs, layer_name)

@tensorrt_converter(torch.nn.modules.activation.Sigmoid)
def sigmoid(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    return activation.add_sigmoid(network,"tensorrt", kwargs, layer_name)