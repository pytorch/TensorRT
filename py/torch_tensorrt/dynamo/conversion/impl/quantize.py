from typing import Optional, Union

import numpy as np
import tensorrt as trt
import torch
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor, to_torch
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTTensor


def quantize(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_tensor: TRTTensor,
    amax: Union[np.ndarray, torch.Tensor],
    num_bits: int,
    exponent_bits: int,
) -> TRTTensor:
    """
    Adds quantize and dequantize ops (QDQ) which quantize to INT8 or FP8 based
    on the output_type set and dequantizes them back.
    """

    with unset_fake_temporarily():
        if not isinstance(input_tensor, TRTTensor):
            input_tensor = get_trt_tensor(ctx, input_tensor, name + "_quantize_input")
        if isinstance(input_tensor, TRTTensor) and input_tensor.dtype not in (
            trt.float32,
            trt.float16,
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

        amax = to_torch(amax, None)
        scale = torch.divide(amax, max_bound)
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


def dynamic_block_quantize(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_tensor: TRTTensor,
    block_size: int,
    amax: Union[np.ndarray, torch.Tensor],
    num_bits: int,
    exponent_bits: int,
    scale_num_bits: int,
    scale_exponent_bits: int,
) -> TRTTensor:
    """
    Adds quantize and dequantize ops (QDQ) which quantize to FP4 based
    on the output_type set and dequantizes them back.
    """
    print(
        f"dynamic_block_quantize entered: {target=} {source_ir=} {name=} {input_tensor.shape=} {input_tensor.dtype=} {block_size=} {amax=} {num_bits=} {exponent_bits=} {scale_num_bits=} {scale_exponent_bits=}"
    )
    with unset_fake_temporarily():
        if not isinstance(input_tensor, TRTTensor):
            input_tensor = get_trt_tensor(
                ctx, input_tensor, name + "_dynamic_quantize_input"
            )
        if isinstance(input_tensor, TRTTensor) and input_tensor.dtype not in (
            trt.float32,
            trt.float16,
            trt.bfloat16,
        ):
            raise ValueError(
                f"dynamic_block_quantize converter received an input of {input_tensor.dtype} type. Supported types: float32 | float16 | bfloat16"
            )
        if len(input_tensor.shape) not in (2, 3):
            raise ValueError(
                f"dynamic_block_quantize converter received an input of {input_tensor.shape} shape. Supported shapes: 2D or 3D"
            )
        max_bound = 6
        amax = to_torch(amax, None)
        scale = torch.divide(amax, max_bound)
        scale = get_trt_tensor(ctx, scale, name + "_scale")

        # Add Q node
        dynamic_quantize_layer = ctx.net.add_dynamic_quantize(
            input_tensor,
            axis=1,
            block_size=16,
            output_type=trt.DataType.FP4,
            scale_type=trt.DataType.FP8,
        )
        set_layer_name(
            dynamic_quantize_layer, target, name + "_dynamic_quantize", source_ir
        )
        q_output = dynamic_quantize_layer.get_output(0)
        q_scale = dynamic_quantize_layer.get_output(1)

        # Add double DQ node
        scale_dequantize_layer = ctx.net.add_dequantize(q_scale, scale)
        scale_dequantize_layer.axis = 0
        set_layer_name(
            scale_dequantize_layer, target, name + "_scale_dequantize", source_ir
        )
        scale_dequantize_layer.precision = trt.DataType.FP8
        scale_dq_output = scale_dequantize_layer.get_output(0)

        dequantize_layer = ctx.net.add_dequantize(q_output, scale_dq_output)
        dequantize_layer.axis = 1
        set_layer_name(dequantize_layer, target, name + "_dequantize", source_ir)
        dequantize_layer.precision = trt.DataType.FP4
        dq_output = dequantize_layer.get_output(0)

        return dq_output
