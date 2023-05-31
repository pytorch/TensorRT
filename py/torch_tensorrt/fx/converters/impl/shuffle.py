import numpy as np
import operator
import warnings
from typing import cast, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
from torch.fx.node import Argument, Target

from ..converter_utils import *  # noqa: F403
from ...utils import get_dynamic_dims, torch_dtype_from_trt, torch_dtype_to_trt

from torch_tensorrt.fx.types import (
    TRTNetwork,
    TRTTensor,
)


def convert_permute(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    index: Sequence[TRTTensor],
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    ranks = len(input_val.shape) + (1 if network.has_implicit_batch_dimension else 0)  # type: ignore[union-attr]
    if len(index) == 1:
        index = index[0]
    permutation = [get_positive_dim(i, ranks) for i in cast(Sequence[int], index)]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"permute received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if network.has_implicit_batch_dimension:
        assert permutation[0] == 0, "Can't permute batch dimension when it's implicit."
        permutation = [i - 1 for i in permutation[1:]]

    layer = network.add_shuffle(input_val)
    layer.second_transpose = tuple(permutation)
    set_layer_name(layer, target, name)
    return layer.get_output(0)


def convert_squeeze(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val,
    dim: int,
) -> Union[TRTTensor, Sequence[TRTTensor]]:

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"squeeze received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    # Squeeze with dim=None would only work in explicit batch dim mode without any dynamic
    # dim, which is a very rare case. For now we just claim not supporting dim=None.
    assert dim is not None, "We don't support dim=None right now for squeeze."

    dim = get_positive_dim(
        dim, len(input_val.shape) + (1 if network.has_implicit_batch_dimension else 0)
    )
    if network.has_implicit_batch_dimension:
        assert dim != 0, "We don't support squeeze batch dim when it's implicit."
        dim -= 1

    assert input_val.shape[dim] != -1, "We don't support squeeze dynamic dim."
    assert (
        len(get_dynamic_dims(input_val.shape)) <= 1
    ), "Currently more than one dynamic dim for input to squeeze is not supported."

    output_shape = []
    for i, s in enumerate(input_val.shape):
        if i == dim and s == 1:
            continue
        output_shape.append(s)
    layer = network.add_shuffle(input_val)
    layer.reshape_dims = tuple(output_shape)
    set_layer_name(layer, target, name)
    return layer.get_output(0)


def convert_unsqueeze(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_t,
    dim,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = get_trt_tensor(network, input_t, f"{name}_input_t")
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"unsqueeze received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    dim = cast(int, dim)
    input_shape = input_val.shape
    input_shape_size = (
        len(input_val.shape) + 1
        if network.has_implicit_batch_dimension
        else len(input_val.shape)
    )
    dim = get_positive_dim(dim, input_shape_size + 1)

    if network.has_implicit_batch_dimension:
        assert dim != 0
        dim -= 1

    assert (
        len(get_dynamic_dims(input_val.shape)) <= 1
    ), "Currently we don't support unsqueeze with more than one dynamic dims."
    layer = network.add_shuffle(input_val)
    layer.reshape_dims = (
        tuple(input_val.shape)[:dim] + (1,) + tuple(input_val.shape)[dim:]
    )
    set_layer_name(layer, target, name)
    return layer.get_output(0)
