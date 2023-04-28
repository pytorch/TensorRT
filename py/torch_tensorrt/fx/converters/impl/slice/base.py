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
    has_dynamic_shape,
    set_layer_name,
)

from torch_tensorrt.fx.converters.impl.shape import get_shape_with_dynamic_shape


def slice(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    start: Shape,
    shape: Shape,
    stride: Shape,
) -> TRTTensor:
    dynamic_shape = has_dynamic_shape(input.shape)
    if dynamic_shape:
        shape = get_shape_with_dynamic_shape(
            network, target, source_ir, name, shape, input
        )
    layer = network.add_slice(
        input,
        start=start,
        shape=[] if dynamic_shape else shape,
        stride=stride,
    )
    if dynamic_shape:
        layer.set_input(2, shape)
    set_layer_name(layer, target, name)
    return layer.get_output(0)
