import operator
import warnings
from typing import Union, Callable, Any, Optional

import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
from torch.fx.node import Target

from torch_tensorrt.fx.types import TRTNetwork, TRTTensor, TRTElementWiseOp
from torch_tensorrt.fx.utils import torch_dtype_from_trt
from torch_tensorrt.fx.converters.converter_utils import (
    SourceIR,
    set_layer_name,
    to_numpy,
)

from torch_tensorrt.fx.converters.impl.elementwise.base import (
    convert_binary_elementwise,
)


def get_shape_with_dynamic_shape(
    network: TRTNetwork,
    target: Target,
    source_ir: SourceIR,
    name: str,
    shape: Union[list, tuple, torch.Tensor],
    input_val: TRTTensor,
) -> TRTTensor:
    """
    Prepare the real output tensor shape for dynamic shape mode tensor input.
    How this functions works:
    Assuming the input_val has actual shape [2048, 256, 512], expected reduce operation
    output shape is [-1, 128, 256], this function should return [2048, 128, 256] as the actual
    reduce operation output shape. Steps of calculations are:
        1. get the actual tensor shape of input_val via add_shape layer;
        2. create a all 0 tensor [0, 0, 0];
        3. run elementwise comparision the [0, 0, 0] and [-1, 128, 256] tensor, get a condition tensor [True, False, False];
        4. use the condition tensor [True, False, False] to do selection between [2048, 256, 512] and [-1, 128, 256], replace
           all -1 dynamic shape dimensions with actual batch_size value;
        5. output shape with actual batch_size as [2048, 128, 256]

    Args:
        network (TRTNetwork): TensorRT network object.
        shape: calculated shape of the expected output tensor
        input_val (TRTTensor): A TensorRT ITensor.
        target (Target): Target of fx node.
        name (str): The name we want to assign to the created TensorRT layer.
    Returns:
        TensorRT ITensors that represents the actual shape of the input_val
    """
    # Ger real shape info for input_val
    input_shape = network.add_shape(input_val).get_output(0)

    scale_layer = network.add_constant(
        input_shape.shape, np.ascontiguousarray(shape, dtype=np.int32)
    )
    set_layer_name(scale_layer, target, f"{name}_scale")
    scale_res = scale_layer.get_output(0)

    length = input_shape.shape[0]
    zero_layer = network.add_constant(
        input_shape.shape, to_numpy(torch.zeros((length), dtype=torch.int32))
    )
    set_layer_name(zero_layer, target, f"{name}_zeros")

    condition_val = convert_binary_elementwise(
        network,
        target,
        source_ir,
        f"{name}_shape",
        trt.ElementWiseOperation.LESS,
        scale_res,
        zero_layer.get_output(0),
    )
    select_layer = network.add_select(condition_val, input_shape, scale_res)
    set_layer_name(select_layer, target, f"{name}_select")
    return select_layer.get_output(0)
