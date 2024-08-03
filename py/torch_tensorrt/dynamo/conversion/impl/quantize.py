from typing import Optional

import numpy as np
import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTTensor


def quantize(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    quantize_type: str,
    input_tensor: TRTTensor,
    scale: np.ndarray,
) -> TRTTensor:
    """
    Adds quantize and dequantize ops (QDQ) which quantize to INT8 or FP8 based
    on the output_type set and dequantizes them back.
    """
    if (isinstance(input_tensor, TRTTensor)) and not (
        input_tensor.dtype == trt.float32 or input_tensor.dtype == trt.float16
    ):
        raise ValueError(
            f"quantize {quantize_type} converter received an input of {input_tensor.dtype} type. Supported types: float32 | float16"
        )
    if quantize_type not in ["fp8", "int8"]:
        raise ValueError(
            f"{quantize_type=} is not supported. Supported types: fp8 | int8"
        )
    scale = get_trt_tensor(ctx, scale, name + "_scale")
    # Add Q node
    quantize_layer = ctx.net.add_quantize(input_tensor, scale)
    if quantize_type == "int8":
        quantize_layer.set_output_type(0, trt.DataType.INT8)
    else:
        quantize_layer.set_output_type(0, trt.DataType.FP8)
    set_layer_name(quantize_layer, target, name + "_quantize", source_ir)
    q_output = quantize_layer.get_output(0)
    # Add DQ node
    dequantize_layer = ctx.net.add_dequantize(q_output, scale)
    set_layer_name(dequantize_layer, target, name + "_dequantize", source_ir)
    if quantize_type == "int8":
        dequantize_layer.precision = trt.DataType.INT8
    else:
        # Set DQ layer precision to FP8
        dequantize_layer.precision = trt.DataType.FP8
    dq_output = dequantize_layer.get_output(0)

    return dq_output


def quantize_fp8(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_tensor: TRTTensor,
    scale: np.ndarray,
) -> TRTTensor:
    return quantize(ctx, target, source_ir, name, "fp8", input_tensor, scale)


def quantize_int8(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_tensor: TRTTensor,
    scale: np.ndarray,
) -> TRTTensor:
    return quantize(ctx, target, source_ir, name, "int8", input_tensor, scale)
