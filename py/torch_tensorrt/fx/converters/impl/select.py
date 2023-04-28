import operator
import warnings
from typing import Union, Callable, Any, Optional, cast

import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
from torch.fx.node import Target

from torch_tensorrt.fx.types import TRTNetwork, TRTTensor, Shape
from torch_tensorrt.fx.converters.converter_utils import (
    SourceIR,
    get_positive_dim,
    has_dynamic_shape,
    to_numpy,
)
from torch_tensorrt.fx.converters.impl.shape import get_shape_with_dynamic_shape


def select(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Shape,
    index: Shape,
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"slice_tensor received input {input} that is not part "
            "of the TensorRT region!"
        )

    ranks = len(input.shape) + (1 if network.has_implicit_batch_dimension else 0)
    dim = get_positive_dim(cast(int, dim), ranks)
    dynamic_shape = has_dynamic_shape(input.shape)
    if network.has_implicit_batch_dimension:
        if dim == 0:
            raise RuntimeError(
                f"We do not support slice_tensor at batch dim when it's implicit, got {dim}!"
            )
        dim = dim - 1
    else:
        if dynamic_shape:
            # Check whether slice target dim is dynamic shape dim
            assert input.shape[dim] != -1, "Can't select on negative shape dimension!"
    index = index

    if index >= input.shape[dim]:
        raise RuntimeError(
            f"cannot have index greater than the dimension length! {input.shape[dim]}"
        )
    output_shape = list(input.shape)
    output_shape[dim] = 1
    if dynamic_shape > 0:
        output_shape = get_shape_with_dynamic_shape(
            network, target, source_ir, name, output_shape, input
        )
    index_value = torch.tensor(index, dtype=torch.int32)
    indices_tensor = network.add_constant(
        index_value.shape, to_numpy(index_value)
    ).get_output(0)
    layer = network.add_gather(input, indices_tensor, dim)
    out = layer.get_output(0)
    if len(out.shape) != 1:
        layer = network.add_shuffle(out)
    return layer.get_output(0)
