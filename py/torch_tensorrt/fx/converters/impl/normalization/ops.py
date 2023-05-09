import operator
import warnings
from typing import cast, Union, Callable, Any, Optional, Sequence

import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
from torch.fx.node import Target

from torch_tensorrt.fx.types import TRTNetwork, TRTTensor

from torch_tensorrt.fx.converters.converter_utils import (
    SourceIR,
    set_layer_name,
    get_positive_dim,
)


def softmax(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Optional[Any] = None,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_ranks = len(input.shape) + (1 if network.has_implicit_batch_dimension else 0)  # type: ignore[union-attr]

    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"softmax received input {input} that is not part "
            "of the TensorRT region!"
        )

    # Used to get dim when dim is None. Copied from PyTorch softmax implementation.
    def get_softmax_dim(ndim: int) -> int:
        if ndim == 0 or ndim == 1 or ndim == 3:
            ret = 0
        else:
            ret = 1
        return ret

    if dim is None:
        dim = get_softmax_dim(input_ranks)
    else:
        dim = cast(int, dim)

    dim = get_positive_dim(dim, input_ranks)
    if network.has_implicit_batch_dimension:
        assert dim != 0, "Can't apply softmax on batch dimension when it's implicit."
        dim -= 1

    layer = network.add_softmax(input)
    layer.axes = 1 << dim
    set_layer_name(layer, target, name)
    return layer.get_output(0)
