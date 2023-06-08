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


def permute(
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
