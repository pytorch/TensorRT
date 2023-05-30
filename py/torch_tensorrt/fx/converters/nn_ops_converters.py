import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch

from torch_tensorrt.fx.converter_registry import tensorrt_converter
from torch_tensorrt.fx.converters.impl import activation
from torch_tensorrt.fx.converters.converter_utils import SourceIR


@tensorrt_converter(torch.nn.functional.relu)
@tensorrt_converter(torch.nn.modules.activation.ReLU)
def relu(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0

    return activation.relu(
        network=network,
        target="torch.nn.functional.relu",
        source_ir=SourceIR.NN,
        name=layer_name,
        input_val=kwargs["input"],
    )


@tensorrt_converter(torch.nn.modules.activation.Sigmoid)
def sigmoid(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0

    activation.sigmoid(
        network=network,
        target="torch.nn.modules.activation.Sigmoid",
        source_ir=SourceIR.NN,
        name=layer_name,
        input_val=kwargs["input"],
    )


@tensorrt_converter(torch.nn.functional.hardtanh)
@tensorrt_converter(torch.nn.modules.activation.Hardtanh)
def hardtanh(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0

    return activation.hardtanh(
        network=network,
        target="torch.nn.modules.activation.Hardtanh",
        source_ir=SourceIR.NN,
        name=layer_name,
        input_val=kwargs["input"],
    )


@tensorrt_converter(torch.nn.functional.tanh)
@tensorrt_converter(torch.nn.modules.activation.Tanh)
def tanh(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0

    return activation.tanh(
        network=network,
        target="torch.nn.modules.activation.Tanh",
        source_ir=SourceIR.NN,
        name=layer_name,
        input_val=kwargs["input"],
    )


@tensorrt_converter(torch.nn.functional.leaky_relu)
@tensorrt_converter(torch.nn.modules.activation.LeakyReLU)
def leaky_relu(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0

    return activation.leaky_relu(
        network=network,
        target="torch.nn.functional.leaky_relu",
        source_ir=SourceIR.NN,
        name=layer_name,
        input_val=kwargs["input"],
        alpha=kwargs["negative_slope"],
    )
