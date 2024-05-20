from typing import Optional, Union

import numpy as np
import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    broadcastable,
    get_trt_tensor,
)
from torch_tensorrt.fx.converters.converter_utils import prepend_ones, set_layer_name
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
    max_shape_len = max(len(x_shape), len(y_shape), len(condition_shape))

    if not isinstance(condition, TRTTensor):
        assert condition.dtype in (torch.bool, np.bool_), "condition dtype is not bool"
        condition = get_trt_tensor(ctx, condition, f"{name}_condition")
    diff = max_shape_len - len(condition_shape)
    if diff > 0:
        condition = prepend_ones(
            ctx.net, condition, f"{name}_condition_broadcast", diff
        )

    if not isinstance(input, TRTTensor):
        input = get_trt_tensor(ctx, input, f"{name}_x")
    diff = max_shape_len - len(x_shape)
    if diff > 0:
        input = prepend_ones(ctx.net, input, f"{name}_input_broadcast", diff)

    if not isinstance(other, TRTTensor):
        other = get_trt_tensor(ctx, other, f"{name}_y")
    diff = max_shape_len - len(y_shape)
    if diff > 0:
        other = prepend_ones(ctx.net, other, f"{name}_other_broadcast", diff)

    return select(ctx, target, source_ir, name, input, other, condition)


def select(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    other: TRTTensor,
    condition: TRTTensor,
) -> TRTTensor:
    select_layer = ctx.net.add_select(condition, input, other)
    set_layer_name(select_layer, target, name + "_select", source_ir)
    return select_layer.get_output(0)
