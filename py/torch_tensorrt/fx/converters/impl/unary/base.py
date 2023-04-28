import operator
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from enum import Enum, auto

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
from torch.fx.node import Target

from torch_tensorrt.fx.types import (
    TRTNetwork,
    TRTTensor,
)

from torch_tensorrt.fx.converters.converter_utils import SourceIR, set_layer_name

from torch_tensorrt.fx.converters.impl.elementwise.base import (
    convert_binary_elementwise,
)


def convert_unary(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    operation_type: trt.UnaryOperation,
    input_val: TRTTensor,
) -> TRTTensor:
    """
    Add a TensorRT Unary layer to `network`.

    Args:
        network (TRTNetwork): TensorRT network object.
        input_val (TRTTensor): Input to the unary op. Must be a TensorRT tensor.
        op_type (trt.ElementWiseOperation): Type of the TensorRT unary operation.
        target (Target): Target of fx node.
        name (str): The name we want to assign to the created TensorRT layer.

    Returns:
        The output of TensorRT Unary layer.
    """
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"{operation_type} received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    layer = network.add_unary(input_val, operation_type)
    set_layer_name(layer, target, name, source_ir)
    output = layer.get_output(0)
    output.name = output.name + "_" + target.__name__
    return layer.get_output(0)
