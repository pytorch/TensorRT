from typing import Optional, Union

import numpy as np
import tensorrt as trt
import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    broadcastable,
    get_trt_tensor,
)
from torch_tensorrt.dynamo.conversion.impl.slice import expand
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTTensor


def where(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: Union[TRTTensor, np.ndarray, torch.Tensor],
    other: Union[TRTTensor, np.ndarray, torch.Tensor],
    condition: Union[TRTTensor, np.ndarray, torch.Tensor],
) -> TRTTensor:
    if not (broadcastable(input, other)):
        assert "The two torch tensors should be broadcastable"

    x_shape = list(input.shape)
    y_shape = list(other.shape)
    condition_shape = list(condition.shape)

    output_shape = list(torch.broadcast_shapes(condition_shape, x_shape, y_shape))

    # expand shape
    if not isinstance(condition, TRTTensor):
        assert condition.dtype in (torch.bool, np.bool_), "condition dtype is not bool"
        if condition_shape != output_shape:
            condition = (
                condition.expand(output_shape)
                if isinstance(condition, torch.Tensor)
                else np.broadcast_to(condition, output_shape)
            )
        condition_val = get_trt_tensor(ctx, condition, f"{name}_condition")
    else:
        assert condition.dtype == trt.bool, "mask dtype is not bool!"
        if condition_shape != output_shape:
            condition_val = expand(
                ctx, target, source_ir, f"{name}_expand", condition, output_shape
            )
        else:
            condition_val = condition

    if not isinstance(input, TRTTensor):
        if x_shape != output_shape:
            # special case where 1 element in input
            if len(input.shape) == 0:
                input = (
                    input.unsqueeze(0)
                    if isinstance(input, torch.Tensor)
                    else np.expand_dims(input, axis=0)
                )
            input = (
                input.expand(output_shape)
                if isinstance(input, torch.Tensor)
                else np.broadcast_to(input, output_shape)
            )
        x_val = get_trt_tensor(ctx, input, f"{name}_x")
    else:
        x_val = input
        if x_shape != output_shape:
            x_val = expand(
                ctx, target, source_ir, f"{name}_x_expand", input, output_shape
            )

    if not isinstance(other, TRTTensor):
        if y_shape != output_shape:
            # special case where 1 element in other
            if len(other.shape) == 0:
                other = (
                    other.unsqueeze(0)
                    if isinstance(other, torch.Tensor)
                    else np.expand_dims(other, axis=0)
                )
            other = (
                other.expand(output_shape)
                if isinstance(other, torch.Tensor)
                else np.broadcast_to(other, output_shape)
            )
        y_val = get_trt_tensor(ctx, other, f"{name}_y")
    else:
        y_val = other
        if y_shape != output_shape:
            y_val = expand(
                ctx, target, source_ir, f"{name}_y_expand", y_val, output_shape
            )

    select_layer = ctx.net.add_select(condition_val, x_val, y_val)

    set_layer_name(select_layer, target, f"{name}_select")

    return select_layer.get_output(0)
