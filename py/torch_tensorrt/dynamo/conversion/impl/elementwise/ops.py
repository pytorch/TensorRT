from typing import Optional, Union

import numpy as np
import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_int_int_div_trt_tensor,
    cast_int_or_float_to_bool,
)
from torch_tensorrt.dynamo.conversion.impl.elementwise.base import (
    convert_binary_elementwise,
)
from torch_tensorrt.dynamo.conversion.impl.unary import sign
from torch_tensorrt.dynamo.conversion.impl.unary.base import convert_unary
from torch_tensorrt.fx.converters.converter_utils import (
    get_trt_tensor,
    set_layer_name,
    squeeze_left,
)
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor
from torch_tensorrt.fx.utils import Frameworks, unified_dtype_converter


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


def clamp(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> TRTTensor:
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"Clamp received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    def _add_layer(
        network: TRTNetwork,
        input: TRTTensor,
        val: float,
        op: trt.ElementWiseOperation,
        name: str,
    ) -> (
        trt.ILayer
    ):  # TODO: Simplify and merge implementations, should just be max and min stacked
        if not len(input.shape):
            # clamping scalar
            acc_ops_clamp_trt = get_trt_tensor(
                network,
                squeeze_left(
                    np.array(
                        [val],
                        dtype=unified_dtype_converter(input.dtype, Frameworks.NUMPY),
                    )
                ),
                f"{name}_clamp_{val}",
            )
        else:
            acc_ops_clamp_shape = (1,) * len(input.shape)  # broadcast all dimensions
            acc_ops_clamp_tensor = np.full(
                acc_ops_clamp_shape,
                val,
                dtype=unified_dtype_converter(input.dtype, Frameworks.NUMPY),
            )
            acc_ops_clamp_trt = network.add_constant(
                acc_ops_clamp_shape, acc_ops_clamp_tensor
            ).get_output(0)
        layer = network.add_elementwise(input, acc_ops_clamp_trt, op)
        return layer

    if min_val is not None:
        clamp_min_layer = _add_layer(
            network, input_val, min_val, trt.ElementWiseOperation.MAX, name
        )
        set_layer_name(clamp_min_layer, target, f"{name}_clamp_min")
        input_val = clamp_min_layer.get_output(0)
    if max_val is not None:
        clamp_max_layer = _add_layer(
            network, input_val, max_val, trt.ElementWiseOperation.MIN, name
        )
        set_layer_name(clamp_max_layer, target, f"{name}_clamp_max")
        input_val = clamp_max_layer.get_output(0)

    return input_val


def add(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        network, target, source_ir, name, trt.ElementWiseOperation.SUM, lhs_val, rhs_val
    )


def mul(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        network,
        target,
        source_ir,
        name,
        trt.ElementWiseOperation.PROD,
        lhs_val,
        rhs_val,
    )


def max(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        network, target, source_ir, name, trt.ElementWiseOperation.MAX, lhs_val, rhs_val
    )


def min(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        network, target, source_ir, name, trt.ElementWiseOperation.MIN, lhs_val, rhs_val
    )


def sub(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        network, target, source_ir, name, trt.ElementWiseOperation.SUB, lhs_val, rhs_val
    )


def div(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    if isinstance(lhs_val, TRTTensor) and isinstance(rhs_val, TRTTensor):
        lhs_val, rhs_val = cast_int_int_div_trt_tensor(network, lhs_val, rhs_val, name)

    return convert_binary_elementwise(
        network, target, source_ir, name, trt.ElementWiseOperation.DIV, lhs_val, rhs_val
    )


def pow(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    if isinstance(lhs_val, TRTTensor) and isinstance(rhs_val, TRTTensor):
        lhs_val, rhs_val = cast_int_int_div_trt_tensor(network, lhs_val, rhs_val, name)

    return convert_binary_elementwise(
        network, target, source_ir, name, trt.ElementWiseOperation.POW, lhs_val, rhs_val
    )


def floor_divide(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        network,
        target,
        source_ir,
        name,
        trt.ElementWiseOperation.FLOOR_DIV,
        lhs_val,
        rhs_val,
    )


def logical_and(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    if isinstance(lhs_val, TRTTensor):
        lhs_val = cast_int_or_float_to_bool(network, name, lhs_val)

    if isinstance(rhs_val, TRTTensor):
        rhs_val = cast_int_or_float_to_bool(network, name, rhs_val)

    return convert_binary_elementwise(
        network, target, source_ir, name, trt.ElementWiseOperation.AND, lhs_val, rhs_val
    )


def logical_or(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    if isinstance(lhs_val, TRTTensor):
        lhs_val = cast_int_or_float_to_bool(network, name, lhs_val)

    if isinstance(rhs_val, TRTTensor):
        rhs_val = cast_int_or_float_to_bool(network, name, rhs_val)

    return convert_binary_elementwise(
        network, target, source_ir, name, trt.ElementWiseOperation.OR, lhs_val, rhs_val
    )


def logical_xor(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    if isinstance(lhs_val, TRTTensor):
        lhs_val = cast_int_or_float_to_bool(network, name, lhs_val)

    if isinstance(rhs_val, TRTTensor):
        rhs_val = cast_int_or_float_to_bool(network, name, rhs_val)

    return convert_binary_elementwise(
        network, target, source_ir, name, trt.ElementWiseOperation.XOR, lhs_val, rhs_val
    )


def eq(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        network,
        target,
        source_ir,
        name,
        trt.ElementWiseOperation.EQUAL,
        lhs_val,
        rhs_val,
    )


def gt(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        network,
        target,
        source_ir,
        name,
        trt.ElementWiseOperation.GREATER,
        lhs_val,
        rhs_val,
    )


def lt(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        network,
        target,
        source_ir,
        name,
        trt.ElementWiseOperation.LESS,
        lhs_val,
        rhs_val,
    )
