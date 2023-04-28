import operator
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from enum import Enum, auto

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
from torch.fx.node import Target

from torch_tensorrt.fx.types import (
    TRTNetwork,
    TRTTensor,
)

from torch_tensorrt.fx.converters.converter_utils import (
    SourceIR,
)

from torch_tensorrt.fx.converters.impl.elementwise.base import (
    convert_binary_elementwise,
)
from torch_tensorrt.fx.converters.impl.unary.base import convert_unary


def sign(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    """
    Sign is calculated as below:
       x = input
       sign = (exp(x) // exp(abs(x))) * 2 - 1
       For positive number and 0, (exp(x) // exp(abs(x))) yield 1; for negative number, (exp(x) // exp(abs(x))) yield 0.
       With multiply 2, the value become 2(for pos and 0) and 0(for neg).
       Finally minus 1, the value become 1(for pos and 0) and -1(for neg).

    Args:
        network (TRTNetwork): TensorRT network object.
        target (Target): fx node target.
        source_ir (SourceIR): Source IR calling the function
        name (str): Name of the fx node with optional suffix.
        input_val (TRTTensor): The input tensor.

    Returns:
        A TensorRT tensor represent the result of sign operator.
    """
    input_exp_output = convert_unary(
        network,
        target,
        source_ir,
        f"{name}_prod_exp",
        trt.UnaryOperation.EXP,
        input_val,
    )
    input_abs_output = convert_unary(
        network,
        target,
        source_ir,
        f"{name}_prod_abs",
        trt.UnaryOperation.ABS,
        input_val,
    )
    input_abs_exp_output = convert_unary(
        network,
        target,
        source_ir,
        f"{name}_prod_abs_exp",
        trt.UnaryOperation.EXP,
        input_abs_output,
    )

    floor_div_output = convert_binary_elementwise(
        network,
        target,
        source_ir,
        f"{name}_exp_floor_div",
        trt.ElementWiseOperation.FLOOR_DIV,
        input_exp_output,
        input_abs_exp_output,
    )

    double_floor_div_output = convert_binary_elementwise(
        network,
        target,
        source_ir,
        f"{name}_floor_div*2",
        trt.ElementWiseOperation.PROD,
        floor_div_output,
        2,
    )

    return convert_binary_elementwise(
        network,
        target,
        source_ir,
        f"{name}_sign",
        trt.ElementWiseOperation.SUB,
        double_floor_div_output,
        1,
    )
