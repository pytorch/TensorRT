from typing import Optional, Union

import tensorrt as trt
import torch
import torch_tensorrt.dynamo.conversion.impl as impl
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_int_int_div_trt_tensor,
    cast_int_or_float_to_bool,
    get_trt_tensor,
)
from torch_tensorrt.dynamo.conversion.impl.elementwise.base import (
    convert_binary_elementwise,
)
from torch_tensorrt.dynamo.conversion.impl.unary import sign
from torch_tensorrt.dynamo.conversion.impl.unary.base import convert_unary
from torch_tensorrt.fx.types import TRTTensor
from torch_tensorrt.fx.utils import Frameworks, unified_dtype_converter


def trunc_div(
    ctx: ConversionContext,
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
        ctx: ConversionContext.
        target: node target
        source_ir (SourceIR): Source IR calling the function.
        name: namespace for the op
        input: divisor.
        other: dividend.

    Returns:
        A TensorRT tensor represent the result of trunc divide.
    """
    prod_output = convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        f"{name}_prod",
        trt.ElementWiseOperation.PROD,
        input,
        other,
    )

    sign_output = sign(
        ctx,
        target,
        source_ir,
        name,
        prod_output,
    )

    # Convert constant input into ITensor for UnaryOperation
    if not isinstance(input, trt.tensorrt.ITensor):
        input = get_trt_tensor(ctx, input, f"{name}_input")
    if not isinstance(other, trt.tensorrt.ITensor):
        other = get_trt_tensor(
            ctx,
            other,
            f"{name}_other",
            dtype=unified_dtype_converter(input.dtype, Frameworks.TORCH),
        )

    abs_input_output = convert_unary(
        ctx,
        target,
        source_ir,
        f"{name}_abs_input",
        trt.UnaryOperation.ABS,
        input,
    )
    abs_other_output = convert_unary(
        ctx,
        target,
        source_ir,
        f"{name}_abs_other",
        trt.UnaryOperation.ABS,
        other,
    )
    abs_floor_output = convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        f"{name}_floor_div",
        trt.ElementWiseOperation.FLOOR_DIV,
        abs_input_output,
        abs_other_output,
    )
    output = convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        f"{name}_output",
        trt.ElementWiseOperation.PROD,
        abs_floor_output,
        sign_output,
    )

    return output


def rsqrt(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
) -> TRTTensor:
    sqrt_trt_output = convert_unary(
        ctx,
        target,
        source_ir,
        f"{name}_sqrt",
        trt.UnaryOperation.SQRT,
        input,
    )

    output = convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        f"{name}_output",
        trt.ElementWiseOperation.DIV,
        1,
        sqrt_trt_output,
    )

    return output


def fmod(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    other: TRTTensor,
) -> TRTTensor:
    # NOTE: TRT doesnt currently implement fmod so we need multiple operations to perform it
    trunc_div_value = trunc_div(
        ctx,
        target,
        source_ir,
        name + "_trunc_div",
        input,
        other,
    )
    prod_value = convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        name + "_prod",
        trt.ElementWiseOperation.PROD,
        trunc_div_value,
        other,
    )
    sub_value = convert_binary_elementwise(
        ctx,
        target,
        SourceIR.ACC,
        name + "_sub",
        trt.ElementWiseOperation.SUB,
        input,
        prod_value,
    )
    return sub_value


def clamp(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    min_val: Optional[Union[int, float, TRTTensor]] = None,
    max_val: Optional[Union[int, float, TRTTensor]] = None,
) -> TRTTensor:
    clamped_val = input_val
    if min_val is not None:
        clamped_val = impl.elementwise.max(
            ctx, target, source_ir, f"{name}_max", clamped_val, min_val
        )

    if max_val is not None:
        clamped_val = impl.elementwise.min(
            ctx, target, source_ir, f"{name}_min", clamped_val, max_val
        )

    return clamped_val


def add(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.SUM, lhs_val, rhs_val
    )


def mul(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        name,
        trt.ElementWiseOperation.PROD,
        lhs_val,
        rhs_val,
    )


def max(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.MAX, lhs_val, rhs_val
    )


def min(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.MIN, lhs_val, rhs_val
    )


def sub(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.SUB, lhs_val, rhs_val
    )


def div(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    if isinstance(lhs_val, TRTTensor) and isinstance(rhs_val, TRTTensor):
        lhs_val, rhs_val = cast_int_int_div_trt_tensor(ctx, lhs_val, rhs_val, name)

    return convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.DIV, lhs_val, rhs_val
    )


def pow(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    if isinstance(lhs_val, TRTTensor) and isinstance(rhs_val, TRTTensor):
        lhs_val, rhs_val = cast_int_int_div_trt_tensor(ctx, lhs_val, rhs_val, name)

    return convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.POW, lhs_val, rhs_val
    )


def floor_divide(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float],
    rhs_val: Union[TRTTensor, int, float],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        name,
        trt.ElementWiseOperation.FLOOR_DIV,
        lhs_val,
        rhs_val,
    )


def logical_and(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float, bool],
    rhs_val: Union[TRTTensor, int, float, bool],
) -> TRTTensor:
    if isinstance(lhs_val, TRTTensor):
        lhs_val = cast_int_or_float_to_bool(ctx, name, lhs_val)

    if isinstance(rhs_val, TRTTensor):
        rhs_val = cast_int_or_float_to_bool(ctx, name, rhs_val)

    return convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.AND, lhs_val, rhs_val
    )


def logical_or(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float, bool],
    rhs_val: Union[TRTTensor, int, float, bool],
) -> TRTTensor:
    if isinstance(lhs_val, TRTTensor):
        lhs_val = cast_int_or_float_to_bool(ctx, name, lhs_val)

    if isinstance(rhs_val, TRTTensor):
        rhs_val = cast_int_or_float_to_bool(ctx, name, rhs_val)

    return convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.OR, lhs_val, rhs_val
    )


def logical_xor(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float, bool],
    rhs_val: Union[TRTTensor, int, float, bool],
) -> TRTTensor:
    if isinstance(lhs_val, TRTTensor):
        lhs_val = cast_int_or_float_to_bool(ctx, name, lhs_val)

    if isinstance(rhs_val, TRTTensor):
        rhs_val = cast_int_or_float_to_bool(ctx, name, rhs_val)

    return convert_binary_elementwise(
        ctx, target, source_ir, name, trt.ElementWiseOperation.XOR, lhs_val, rhs_val
    )


def bitwise_and(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float, torch.Tensor, bool],
    rhs_val: Union[TRTTensor, int, float, torch.Tensor, bool],
) -> TRTTensor:
    return logical_and(ctx, target, source_ir, f"{name}_logical_and", lhs_val, rhs_val)


def bitwise_or(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float, torch.Tensor, bool],
    rhs_val: Union[TRTTensor, int, float, torch.Tensor, bool],
) -> TRTTensor:
    return logical_or(ctx, target, source_ir, f"{name}_logical_or", lhs_val, rhs_val)


def bitwise_xor(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: Union[TRTTensor, int, float, torch.Tensor, bool],
    rhs_val: Union[TRTTensor, int, float, torch.Tensor, bool],
) -> TRTTensor:
    return logical_xor(ctx, target, source_ir, f"{name}_logical_xor", lhs_val, rhs_val)


def eq(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: TRTTensor,
    rhs_val: Union[TRTTensor, int, float, torch.Tensor],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        name,
        trt.ElementWiseOperation.EQUAL,
        lhs_val,
        rhs_val,
    )


def ne(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: TRTTensor,
    rhs_val: Union[TRTTensor, int, float, torch.Tensor],
) -> TRTTensor:
    return impl.unary.logical_not(
        ctx,
        target,
        source_ir,
        f"{name}_logical_not",
        eq(ctx, target, source_ir, f"{name}_eq", lhs_val, rhs_val),
    )


def gt(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: TRTTensor,
    rhs_val: Union[TRTTensor, int, float, torch.Tensor],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        name,
        trt.ElementWiseOperation.GREATER,
        lhs_val,
        rhs_val,
    )


def ge(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: TRTTensor,
    rhs_val: Union[TRTTensor, int, float, torch.Tensor],
) -> TRTTensor:
    return logical_or(
        ctx,
        target,
        source_ir,
        name,
        gt(ctx, target, source_ir, f"{name}_gt", lhs_val, rhs_val),
        eq(ctx, target, source_ir, f"{name}_eq", lhs_val, rhs_val),
    )


def lt(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: TRTTensor,
    rhs_val: Union[TRTTensor, int, float, torch.Tensor],
) -> TRTTensor:
    return convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        name,
        trt.ElementWiseOperation.LESS,
        lhs_val,
        rhs_val,
    )


def le(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: TRTTensor,
    rhs_val: Union[TRTTensor, int, float, torch.Tensor],
) -> TRTTensor:
    return logical_or(
        ctx,
        target,
        source_ir,
        name,
        lt(ctx, target, source_ir, f"{name}_lt", lhs_val, rhs_val),
        eq(ctx, target, source_ir, f"{name}_eq", lhs_val, rhs_val),
    )
