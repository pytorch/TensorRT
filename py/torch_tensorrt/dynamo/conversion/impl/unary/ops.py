from typing import Optional

import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion.converter_utils import cast_trt_tensor
from torch_tensorrt.dynamo.conversion.impl.unary.base import convert_unary
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor


def exp(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    """
    Args:
        network (TRTNetwork): TensorRT network object.
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
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.EXP, input_val
    )


def log(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.LOG, input_val
    )


def sqrt(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.SQRT, input_val
    )


def recip(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.RECIP, input_val
    )


def abs(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.ABS, input_val
    )


def sin(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.SIN, input_val
    )


def cos(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.COS, input_val
    )


def tan(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.TAN, input_val
    )


def sinh(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.SINH, input_val
    )


def cosh(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.COSH, input_val
    )


def asin(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.ASIN, input_val
    )


def acos(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.ACOS, input_val
    )


def atan(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.ATAN, input_val
    )


def asinh(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.ASINH, input_val
    )


def acosh(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.ACOSH, input_val
    )


def atanh(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.ATANH, input_val
    )


def ceil(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.CEIL, input_val
    )


def floor(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.FLOOR, input_val
    )


def logical_not(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and input_val.dtype != trt.bool:
        input_val = cast_trt_tensor(network, input_val, trt.bool, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.NOT, input_val
    )


def sign(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.SIGN, input_val
    )


def round(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.ROUND, input_val
    )


def isinf(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.ISINF, input_val
    )


def neg(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)
    return convert_unary(
        network, target, source_ir, name, trt.UnaryOperation.NEG, input_val
    )
