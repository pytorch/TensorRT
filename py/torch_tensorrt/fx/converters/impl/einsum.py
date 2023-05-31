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


def convert_einsum(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val,
    equation,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    assert type(equation) is str, "equation type is not str"
    const_flag = False
    for i, input_source in enumerate(input_val):
        if type(input_source) == torch.Tensor:
            # const change to TRTensor always output with dtype FLOAT even though stored memory is other type
            # so we cast to float first. And we need other inputs to be the same float type
            input_source = input_source.to(torch.float)
            const_flag = True
        input_val[i] = get_trt_tensor(network, input_source, name + f"_input_source{i}")

    if const_flag:
        for i, input_source in enumerate(input_val):
            if input_source.dtype != trt.float32:
                input_val[i] = type_cast(
                    network, target, f"{name}_input_cast{i}", input_source, trt.float32
                )
    einsum_layer = network.add_einsum(inputs=input_val, equation=equation)
    return einsum_layer.get_output(0)
