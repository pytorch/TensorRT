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

def add_clamp(network, input, val, op, name):
    if not len(input.shape):
        # clamping scalar
        acc_ops_clamp_trt = get_trt_tensor(
            network,
            squeeze_left(torch.tensor([val], dtype=torch_dtype_from_trt(input.dtype))),
            f"{name}_clamp_{val}",
        )
    else:
        acc_ops_clamp_shape = (1,) * len(input.shape)  # broadcast all dimensions
        acc_ops_clamp_tensor = (
            (
                val
                * torch.ones(
                    acc_ops_clamp_shape, dtype=torch_dtype_from_trt(input.dtype)
                )
            )
            .cpu()
            .numpy()
        )
        acc_ops_clamp_trt = network.add_constant(
            acc_ops_clamp_shape, acc_ops_clamp_tensor
        ).get_output(0)
    layer = network.add_elementwise(input, acc_ops_clamp_trt, op)
    return layer

def convert_clamp(
        network: TRTNetwork,
        target: Target,
        source_ir: Optional[SourceIR],
        name: str,
        input_val,
        min_val = None,
        max_val = None,
) -> TRTTensor:
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"Clamp received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if min_val is not None:
        clamp_min_layer = add_clamp(
            network, input_val, min_val, trt.ElementWiseOperation.MAX, name
        )
        set_layer_name(clamp_min_layer, target, f"{name}_clamp_min")
        input_val = clamp_min_layer.get_output(0)
    if max_val is not None:
        clamp_max_layer = add_clamp(
            network, input_val, max_val, trt.ElementWiseOperation.MIN, name
        )
        set_layer_name(clamp_max_layer, target, f"{name}_clamp_max")
        input_val = clamp_max_layer.get_output(0)

    return input_val
