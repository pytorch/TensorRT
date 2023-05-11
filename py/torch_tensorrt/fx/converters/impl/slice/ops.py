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
    get_trt_tensor,
)
from torch_tensorrt.fx.converters.impl.slice.base import slice


def expand(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    sizes: Shape,
) -> TRTTensor:
    shape = list(sizes)

    input_val = get_trt_tensor(network, input, f"{name}_input")

    if network.has_implicit_batch_dimension:
        shape = shape[1:]

    ranks = len(input_val.shape)
    # TRT does not support different dimension size
    # though this condition is not seen in the case of bmm
    # where input_t and shape dimensions are not equal
    assert len(shape) >= ranks
    if len(shape) != ranks:
        shape_tuple = tuple([0] * len(shape))
        shape_tensor = get_trt_tensor(network, input, f"{name}_shape")
        input_val, shape_tensor = broadcast(
            network, input_val, shape_tensor, f"{name}_input_val", f"{name}_shape_val"
        )
        ranks = len(shape)

    inshape = tuple(input_val.shape)
    shape = tuple(shape)
    start = tuple([0] * ranks)
    stride = tuple(
        [int(i == o) for i, o in zip(inshape, shape)]
    )  # stride == 1 if dimensions match, 0 otherwise
    return slice(network, target, source_ir, name, input_val, start, shape, stride)
