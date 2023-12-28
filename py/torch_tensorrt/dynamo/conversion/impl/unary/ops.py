from typing import Optional

import tensorrt as trt
import torch_tensorrt.dynamo.conversion.impl as impl
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_trt_tensor,
)
from torch_tensorrt.dynamo.conversion.impl.unary.base import convert_unary
from torch_tensorrt.fx.types import TRTTensor


def exp(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    """
    Args:
        ctx (ConversionContext): TensorRT ConversionContext object.
        target (Target): fx node target.
        source_ir (SourceIR): Source IR calling the function
        name (str): Name of the fx node with optional suffix.
        input_val (TRTTensor): The input tensor.

    Returns:
        TRTTensor: A TensorRT tensor represent the result of exp operator.
    """
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.EXP, input_val
    )


def log(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.LOG, input_val
    )


def sqrt(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.SQRT, input_val
    )


def recip(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.RECIP, input_val
    )


def abs(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.ABS, input_val
    )


def sin(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.SIN, input_val
    )


def cos(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.COS, input_val
    )


def tan(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.TAN, input_val
    )


def sinh(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.SINH, input_val
    )


def cosh(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.COSH, input_val
    )


def asin(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.ASIN, input_val
    )


def acos(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.ACOS, input_val
    )


def atan(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.ATAN, input_val
    )


def asinh(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.ASINH, input_val
    )


def acosh(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.ACOSH, input_val
    )


def atanh(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.ATANH, input_val
    )


def ceil(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.CEIL, input_val
    )


def floor(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.FLOOR, input_val
    )


def logical_not(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and input_val.dtype != trt.bool:
        input_val = cast_trt_tensor(ctx, input_val, trt.bool, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.NOT, input_val
    )


def bitwise_not(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    return impl.unary.logical_not(
        ctx, target, source_ir, f"{name}_logical_not", input_val
    )


def sign(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.SIGN, input_val
    )


def round(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.ROUND, input_val
    )


def isinf(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.ISINF, input_val
    )


def neg(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.NEG, input_val
    )


def erf(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    return convert_unary(
        ctx, target, source_ir, name, trt.UnaryOperation.ERF, input_val
    )


def trunc(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if input_val.dtype not in (trt.float16, trt.float32):
        return impl.cast.to_copy(
            ctx,
            target,
            source_ir,
            f"{name}_copy",
            input_val,
            input_val.dtype,
            force_layer=True,
        )

    dividend = get_trt_tensor(ctx, 1, f"{name}_dividend")
    return impl.elementwise.trunc_div(
        ctx, target, source_ir, f"{name}_trunc", input_val, dividend
    )
