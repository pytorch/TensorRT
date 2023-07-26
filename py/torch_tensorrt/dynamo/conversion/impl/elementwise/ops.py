from typing import Any, Optional

import tensorrt as trt
from torch.fx.node import Target

from torch_tensorrt.fx.types import TRTNetwork, TRTTensor
from torch_tensorrt.fx.utils import (
    unified_dtype_converter,
    Frameworks,
)
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.fx.converters.converter_utils import get_trt_tensor

from torch_tensorrt.dynamo.conversion.impl.elementwise.base import (
    convert_binary_elementwise,
)
from torch_tensorrt.dynamo.conversion.impl.unary.base import convert_unary
from torch_tensorrt.dynamo.conversion.impl.unary import sign


def trunc_div(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    other: TRTTensor,
) -> TRTTensor:
    """
    Perform trunc divide on Tensor, result of divide will be round toward zero.
    This means for positive number, it will be floor round; for negative number,
    it will be ceil round. Example: [2.1, 0.8, -3.2] -> [2, 0, -3].

    Args:
        network: INetworkDefinition.
        target: node target
        source_ir (SourceIR): Source IR calling the function.
        name: namespace for the op
        input: divisor.
        other: dividend.

    Returns:
        A TensorRT tensor represent the result of trunc divide.
    """
    prod_output = convert_binary_elementwise(
        network,
        target,
        source_ir,
        f"{name}_prod",
        trt.ElementWiseOperation.PROD,
        input,
        other,
    )

    sign_output = sign(
        network,
        target,
        source_ir,
        name,
        prod_output,
    )

    # Convert constant input into ITensor for UnaryOperation
    if not isinstance(input, trt.tensorrt.ITensor):
        input = get_trt_tensor(network, input, f"{name}_input")
    if not isinstance(other, trt.tensorrt.ITensor):
        other = get_trt_tensor(
            network,
            other,
            f"{name}_other",
            dtype=unified_dtype_converter(input.dtype, Frameworks.TORCH),
        )

    abs_input_output = convert_unary(
        network,
        target,
        source_ir,
        f"{name}_abs_input",
        trt.UnaryOperation.ABS,
        input,
    )
    abs_other_output = convert_unary(
        network,
        target,
        source_ir,
        f"{name}_abs_other",
        trt.UnaryOperation.ABS,
        other,
    )
    abs_floor_output = convert_binary_elementwise(
        network,
        target,
        source_ir,
        f"{name}_floor_div",
        trt.ElementWiseOperation.FLOOR_DIV,
        abs_input_output,
        abs_other_output,
    )
    output = convert_binary_elementwise(
        network,
        target,
        source_ir,
        f"{name}_output",
        trt.ElementWiseOperation.PROD,
        abs_floor_output,
        sign_output,
    )

    return output


def rsqrt(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
) -> TRTTensor:

    sqrt_trt_output = convert_unary(
        network,
        target,
        source_ir,
        f"{name}_sqrt",
        trt.UnaryOperation.SQRT,
        input,
    )

    output = convert_binary_elementwise(
        network,
        target,
        source_ir,
        f"{name}_output",
        trt.ElementWiseOperation.DIV,
        1,
        sqrt_trt_output,
    )

    return output


def fmod(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    other: TRTTensor,
) -> TRTTensor:
    # NOTE: TRT doesnt currently implement fmod so we need multiple operations to perform it
    trunc_div_value = trunc_div(
        network,
        target,
        source_ir,
        name + "_trunc_div",
        input,
        other,
    )
    prod_value = convert_binary_elementwise(
        network,
        target,
        source_ir,
        name + "_prod",
        trt.ElementWiseOperation.PROD,
        trunc_div_value,
        other,
    )
    sub_value = convert_binary_elementwise(
        network,
        target,
        SourceIR.ACC,
        name + "_sub",
        trt.ElementWiseOperation.SUB,
        input,
        prod_value,
    )
    return sub_value
