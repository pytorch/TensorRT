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


def convert_scatter(
        network: TRTNetwork,
        target: Target,
        source_ir: Optional[SourceIR],
        name: str,
        data,
        indices,
        updates,
        axis,
        reduction="add",
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    scatter_layer = network.add_scatter(data=data, indices=indices, updates=updates, mode=ScatterMode.ELEMENT)
    scatter_layer.setAxis(axis)
    return scatter_layer.get_output(0)
