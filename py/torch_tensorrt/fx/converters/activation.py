import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch

from ..converter_registry import tensorrt_converter

from .converter_utils import mark_as_int8_layer


@tensorrt_converter(torch.nn.modules.activation.Sigmoid)
def sigmoid(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"Sigmoid received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    def activation_dyn_range_fn(dyn_range):
        def sigmoid_fn(x):
            return 1 / (1 + np.exp(-x))

        return sigmoid_fn(dyn_range[0]), sigmoid_fn(dyn_range[1])

    return common_activation(
        network,
        submod,
        input_val,
        trt.ActivationType.SIGMOID,
        activation_dyn_range_fn,
        layer_name,
    )
