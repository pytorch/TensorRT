import operator
import warnings
from typing import cast, Union, Callable, Any, Optional, Sequence
import logging

import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
from torch.fx.node import Target

from torch_tensorrt.fx.types import TRTNetwork, TRTTensor
from torch_tensorrt.fx.utils import get_dynamic_dims

from torch_tensorrt.fx.converters.converter_utils import (
    SourceIR,
    set_layer_name,
    has_dynamic_shape,
    to_numpy,
)

from torch_tensorrt.fx.converters.impl.unary.base import (
    convert_unary,
)

from torch_tensorrt.fx.converters.impl.elementwise.base import (
    convert_binary_elementwise,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)


def batch_norm(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    training: torch.Tensor,
    momentum: torch.Tensor,
    eps: list,
) -> Union[TRTTensor, Sequence[TRTTensor]]:

    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"BatchNorm2d received input {input} that is not part "
            "of the TensorRT region!"
        )

    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for batch norm."

    scale = cast(torch.Tensor, to_numpy(cast(torch.Tensor, weight))) / np.sqrt(
        cast(torch.Tensor, to_numpy(cast(torch.Tensor, running_var))) + cast(float, eps)
    )

    bias = (
        to_numpy(cast(torch.Tensor, bias))
        - to_numpy(cast(torch.Tensor, running_mean)) * scale
    )
    power = np.ones_like(scale)

    # For BatchNorm1d, reshape 1d to 2d
    output_shape = input.shape
    if not network.has_implicit_batch_dimension and len(input.shape) < 4:
        assert (
            len(get_dynamic_dims(input.shape)) <= 1
        ), "BatchNorm1D with more than one dynamic dims is not currently supported."
        reshape_layer = network.add_shuffle(input)
        if len(input.shape) == 2:
            reshape_layer.reshape_dims = (input.shape[0], input.shape[1], 1, 1)
        else:  # len(input_val.shape) == 3
            reshape_layer.reshape_dims = (
                input.shape[0],
                input.shape[1],
                input.shape[2],
                1,
            )
        set_layer_name(reshape_layer, target, f"{name}_reshape_2d")
        input = reshape_layer.get_output(0)
    layer = network.add_scale(input, trt.ScaleMode.CHANNEL, bias, scale, power)
    set_layer_name(layer, target, name)

    # For BatchNorm1d, reshape output back to 1d
    if not network.has_implicit_batch_dimension and len(output_shape) < 4:
        reshape_output_layer = network.add_shuffle(layer.get_output(0))
        reshape_output_layer.reshape_dims = tuple(output_shape)
        set_layer_name(reshape_output_layer, target, f"{name}_reshape_1d")
        layer = reshape_output_layer
    return layer.get_output(0)
