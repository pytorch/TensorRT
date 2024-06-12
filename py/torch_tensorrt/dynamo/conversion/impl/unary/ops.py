from typing import Optional, Union

import numpy as np
import tensorrt as trt
import torch
import torch_tensorrt.dynamo.conversion.impl as impl
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_trt_tensor,
)
from torch_tensorrt.dynamo.conversion.impl.unary.base import convert_unary
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTDataType, TRTTensor


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


def expm1(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    """
    Computes e^x - 1 for each element of the input tensor.

    Args:
        ctx (ConversionContext): TensorRT ConversionContext object.
        target (Target): fx node target.
        source_ir (SourceIR): Source IR calling the function
        name (str): Name of the fx node with optional suffix.
        input_val (TRTTensor): The input tensor.

    Returns:
        TRTTensor: A TensorRT tensor represent the result of expm1 operator.
    """
    # Compute e^x for each element of the input tensor
    exp_result = exp(ctx, target, source_ir, f"{name}_exp", input_val)

    return impl.elementwise.sub(ctx, target, source_ir, f"{name}_sub", exp_result, 1)


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


def log10(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    log_layer_output = log(ctx, target, source_ir, f"{name}_log", input_val)

    ln10 = 2.302585092994046

    return impl.elementwise.div(
        ctx, target, source_ir, f"{name}_div", log_layer_output, ln10
    )


def log2(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    log_layer_output = log(ctx, target, source_ir, f"{name}_log", input_val)

    ln2 = 0.693147180559945309

    return impl.elementwise.div(
        ctx, target, source_ir, f"{name}_div", log_layer_output, ln2
    )


def log1p(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    """
    Computes log(1 + x) for each element of the input tensor.
    """
    one_plus_x = impl.elementwise.add(
        ctx, target, source_ir, f"{name}_add", input_val, 1
    )

    return log(ctx, target, source_ir, f"{name}_log", one_plus_x)


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


def scalar_tensor(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    scalar: Union[int, float, bool],
    dtype: Optional[Union[torch.dtype, np.dtype, TRTDataType]] = None,
) -> TRTTensor:
    tensor = get_trt_tensor(ctx, scalar, f"{name}_scalar_tensor", dtype)
    identity_layer = ctx.net.add_identity(tensor)
    set_layer_name(identity_layer, target, name, source_ir)
    return identity_layer.get_output(0)


def isnan(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
) -> TRTTensor:
    # False for NaN elements since NaN is not equal to anything, including itself.
    equality_result = impl.elementwise.eq(
        ctx, target, source_ir, f"{name}_eq_nan", input, input
    )

    # Invert equality_result to get a mask where NaN values are marked as True.
    nan_values_mask = logical_not(
        ctx, target, source_ir, f"{name}_logical_not", equality_result
    )

    return nan_values_mask


def local_scalar_dense(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
) -> TRTTensor:
    start = [0] * len(input.shape)
    shape = [1] * len(input.shape)  # Get one element from each dimension
    stride = [1] * len(input.shape)  # Step through each dimension by 1

    layer = ctx.net.add_slice(input=input, start=start, shape=shape, stride=stride)
    set_layer_name(layer, target, f"{name}_slice", source_ir)

    reshape_layer = ctx.net.add_shuffle(layer.get_output(0))
    reshape_layer.reshape_dims = [
        1,
    ]  # Reshape to a single-element tensor
    set_layer_name(reshape_layer, target, f"{name}_reshape", source_ir)

    return reshape_layer.get_output(0)
