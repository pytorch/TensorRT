import operator
import warnings
from typing import Optional, cast, Any

import numpy as np

import tensorrt as trt
import torch
from torch.fx.node import Target

from torch_tensorrt.fx.types import TRTNetwork, TRTTensor, Shape
from torch_tensorrt.fx.converters.converter_utils import (
    SourceIR,
    get_positive_dim,
)

from torch_tensorrt.fx.converters.converter_utils import (
    SourceIR,
    get_positive_dim,
    set_layer_name,
)

from torch_tensorrt.fx.utils import get_dynamic_dims


def squeeze(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Optional[Any] = None,
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"squeeze received input {input} that is not part "
            "of the TensorRT region!"
        )
    dims = []
    if dim is not None:
        if isinstance(dim, int):
            dims.append(cast(Optional[int], dim))
        else:
            for dim in dim:
                dims.append(cast(Optional[int], dim))

    # dim = cast(Optional[int], kwargs["dim"] if "dim" in kwargs else None)
    # Squeeze with dim=None would only work in explicit batch dim mode without any dynamic
    # dim, which is a very rare case. For now we just claim not supporting dim=None.
    assert not (len(dims) == 0), "We don't support dim=None right now for squeeze."

    for dim in dims:
        dim = get_positive_dim(
            dim,
            len(input.shape) + (1 if network.has_implicit_batch_dimension else 0),
        )
        if network.has_implicit_batch_dimension:
            assert dim != 0, "We don't support squeeze batch dim when it's implicit."
            dim -= 1

        assert input.shape[dim] != -1, "We don't support squeeze dynamic dim."
        assert (
            len(get_dynamic_dims(input.shape)) <= 1
        ), "Currently more than one dynamic dim for input to squeeze is not supported."

    output_shape = []
    for i, s in enumerate(input.shape):
        if (i in dims) and s == 1:
            continue
        output_shape.append(s)
    layer = network.add_shuffle(input)
    layer.reshape_dims = tuple(output_shape)
    set_layer_name(layer, target, name)
    return layer.get_output(0)
