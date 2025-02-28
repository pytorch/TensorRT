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
    input_tensor: TRTTensor,
    amax: np.ndarray,
    num_bits: int,
    exponent_bits: int,
) -> TRTTensor:
    """
    Adds quantize and dequantize ops (QDQ) which quantize to INT8 or FP8 based
    on the output_type set and dequantizes them back.
    """
    if isinstance(input_tensor, TRTTensor) and input_tensor.dtype not in (
        trt.DataType.FLOAT,
        trt.DataType.HALF,
    ):
        raise ValueError(
            f"quantize converter received an input of {input_tensor.dtype} type. Supported types: float32 | float16"
        )
    if num_bits != 8 or exponent_bits not in (0, 4):
        raise ValueError(
            f"quantize converter currently only accept INT8 or FP8 based quantize, got {num_bits=}, {exponent_bits=}"
        )
    if num_bits == 8 and exponent_bits == 0:
        max_bound = 127
    elif num_bits == 8 and exponent_bits == 4:
        max_bound = 448
    scale = np.divide(amax, max_bound)
    scale = get_trt_tensor(ctx, scale, name + "_scale")
    # Add Q node
    quantize_layer = ctx.net.add_quantize(input_tensor, scale)
    if num_bits == 8 and exponent_bits == 0:
        quantize_layer.set_output_type(0, trt.DataType.INT8)
    elif num_bits == 8 and exponent_bits == 4:
        quantize_layer.set_output_type(0, trt.DataType.FP8)

    set_layer_name(quantize_layer, target, name + "_quantize", source_ir)
    q_output = quantize_layer.get_output(0)
    # Add DQ node
    dequantize_layer = ctx.net.add_dequantize(q_output, scale)
    set_layer_name(dequantize_layer, target, name + "_dequantize", source_ir)
    if num_bits == 8 and exponent_bits == 0:
        dequantize_layer.precision = trt.DataType.INT8
    elif num_bits == 8 and exponent_bits == 4:
        # Set DQ layer precision to FP8
        dequantize_layer.precision = trt.DataType.FP8
    dq_output = dequantize_layer.get_output(0)

    return dq_output
