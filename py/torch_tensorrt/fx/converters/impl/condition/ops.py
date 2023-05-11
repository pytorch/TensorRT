import operator
import warnings
from typing import Optional, cast

import numpy as np

import tensorrt as trt
import torch
from torch.fx.node import Target

from torch_tensorrt.fx.types import TRTNetwork, TRTTensor, Shape
from torch_tensorrt.fx.converters.converter_utils import (
    SourceIR,
    broadcast,
    broadcastable,
    get_trt_tensor,
    set_layer_name,
)
from torch_tensorrt.fx.converters.impl.slice import expand


def where(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    other: TRTTensor,
    condition: TRTTensor,
) -> TRTTensor:
    input_dim = len(tuple(input.shape))
    other_dim = len(tuple(other.shape))
    condition_dim = len(tuple(condition.shape))

    if type(input) != TRTTensor:
        assert type(input) is torch.Tensor, f"value {input} is not torch.Tensor!"

    if type(other) != TRTTensor:
        assert type(other) is torch.Tensor, f"value {other} is not torch.Tensor!"

    if not (broadcastable(input, other)):
        assert f"The two torch tensors should be broadcastable"

    # get output shape
    # purpose of this is to bring input and other rank same as
    # output_shape to input it to the add_expand operation
    # condition will have dimension of either input or other
    input, other = broadcast(network, input, other, f"{name}_x", f"{name}_y")
    if len(tuple(condition.shape)) != len(tuple(input.shape)):
        condition, input = broadcast(
            network, condition, input, f"{name}_condition", f"{name}_x"
        )

    x_shape = list(input.shape)
    y_shape = list(other.shape)
    condition_shape = list(condition.shape)
    output_shape = list(torch.broadcast_shapes(condition_shape, x_shape, y_shape))

    # expand shape
    if type(condition) != TRTTensor:
        assert condition.dtype == torch.bool, "condition dtype is not bool"
        if condition_shape != output_shape:
            condition.expand(output_shape)
        condition = condition.to(torch.int32)
        condition_const = get_trt_tensor(network, condition, f"{name}_condition")
        condition_layer = network.add_identity(condition_const)
        condition_layer.set_output_type(0, trt.bool)
        set_layer_name(condition_layer, target, f"{name}_condition")
        condition_val = condition_layer.get_output(0)
    else:
        assert condition.dtype == trt.bool, "mask dtype is not bool!"
        if condition_shape != condition_dim:
            condition_val = expand(
                network, target, source_ir, f"{name}_expand", condition, output_shape
            )
        else:
            condition_val = condition

    if type(input) != TRTTensor:
        if x_shape != input_dim:
            # special case where 1 element in input
            if len(input.shape) == 0:
                input = input.unsqueeze(0)
            input = input.expand(output_shape)
        x_val = get_trt_tensor(network, input, f"{name}_x")
    else:
        x_val = input
        if x_shape != output_shape:
            x_val = expand(
                network, target, source_ir, f"{name}_x_expand", input, output_shape
            )

    if type(other) != TRTTensor:
        if y_shape != output_shape:
            # special case where 1 element in other
            if len(other.shape) == 0:
                other = other.unsqueeze(0)
            other = other.expand(output_shape)
        y_val = get_trt_tensor(network, other, f"{name}_y")
    else:
        y_val = other
        if y_shape != other_dim:
            y_val = expand(
                network, target, source_ir, f"{name}_y_expand", y_val, output_shape
            )

    select_layer = network.add_select(condition_val, x_val, y_val)

    set_layer_name(select_layer, target, f"{name}_select")

    return select_layer.get_output(0)
