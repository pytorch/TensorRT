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


def nvfp4_quantize(
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
        f"lan added nvfp4_quantize entered: {target=} {source_ir=} {name=} {input_tensor.shape=} {input_tensor.dtype=} {block_size=} {amax=} {num_bits=} {exponent_bits=} {scale_num_bits=} {scale_exponent_bits=}"
    )
    with unset_fake_temporarily():
        if input_tensor.dtype not in (
            trt.float32,
            trt.float16,
            trt.bfloat16,
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ):
            raise ValueError(
                f"dynamic_block_quantize converter received an input of {input_tensor.dtype} type. Supported types: float32 | float16 | bfloat16"
            )
        if len(input_tensor.shape) not in (2, 3):
            raise ValueError(
                f"dynamic_block_quantize converter received an input of {input_tensor.shape} shape. Supported shapes: 2D or 3D"
            )
        axis = len(input_tensor.shape) - 1

        # TODO: ADD PADDING IF NEEDED
        # TODO: ADD DYNAMIC SHAPE SUPPORT

        global_scale = _calculate_global_scale(ctx, name, amax)

        if ".weight_quantizer" in name:
            block_scale = _calculate_block_scale(
                ctx,
                name,
                input_tensor,
                block_size,
            )
            input_tensor = get_trt_tensor(ctx, input_tensor, name + "_input")
            output = _static_double_quantize(
                ctx,
                target,
                source_ir,
                name,
                input_tensor,
                block_scale,
                global_scale,
            )
        elif ".input_quantizer" in name:
            # quantize input tensor to fp4, output should be data tensor in fp4 and block scale tensor in fp8
            output = _dynamic_double_quantize(
                ctx,
                target,
                source_ir,
                name,
                input_tensor,
                global_scale,
            )

        else:
            raise ValueError(
                f"quantizer received an input of {name}. Supported values: weight_quantizer | input_quantizer"
            )
        return output


def _dynamic_double_quantize(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_tensor: TRTTensor,
    global_scale: TRTTensor,
    axis: int = -1,
    block_size: int = 16,
    output_type: trt.DataType = trt.DataType.FP4,
    scale_type: trt.DataType = trt.DataType.FP8,
) -> TRTTensor:
    """
    quantize input tensor to fp4, output should be data tensor in fp4 and block scale tensor in fp8
    Parameters:
        ctx: ConversionContext,
        target: Target,
        source_ir: Optional[SourceIR]
        name: str
        input_tensor : Tensor (On GPU)
            The input tensor.
        global_scale : Tensor (On GPU)
            The global per-tensor scaling factor. It should contain only 1 element.
        axis : int
            The axis to quantize. Default is -1 (the last axis).
        block_size : int
            The block size for quantization. Default is 16.
        output_type : trt.DataType
            The data type for quantized data. Default is FP4.
        scale_type : trt.DataType
            The data type for block scale. Default is FP8.

    """
    # dynamic quantize input tensor to fp4
    dynamic_quantize_layer = ctx.net.add_dynamic_quantize(
        input_tensor,
        axis,
        block_size,
        output_type,
        scale_type,
    )
    dynamic_quantize_layer.set_input(1, global_scale)
    set_layer_name(
        dynamic_quantize_layer, target, name + "_dynamic_quantize", source_ir
    )
    quantized_data_in_fp4 = dynamic_quantize_layer.get_output(0)
    quantized_scale_in_fp8 = dynamic_quantize_layer.get_output(1)

    # dequantize scale from fp8 to orignal dtype(default is float32)
    dequantize_scale_layer = ctx.net.add_dequantize(
        quantized_scale_in_fp8, global_scale, input_tensor.dtype
    )
    set_layer_name(
        dequantize_scale_layer, target, name + "_dequantize_scale", source_ir
    )
    dequantized_scale = dequantize_scale_layer.get_output(0)

    # dequantize quantized_data_in_fp4 from  fp4 to orignal dtype(default is float32)
    dequantize_data_layer = ctx.net.add_dequantize(
        quantized_data_in_fp4, dequantized_scale, input_tensor.dtype
    )
    set_layer_name(dequantize_data_layer, target, name + "_dequantize_data", source_ir)
    dequantized_data = dequantize_data_layer.get_output(0)
    return dequantized_data


def _static_double_quantize(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_tensor: TRTTensor,
    block_scale: TRTTensor,
    global_scale: TRTTensor,
) -> TRTTensor:
    """
    Parameters:
        ctx: ConversionContext,
        target: Target,
        source_ir: Optional[SourceIR],
        name: str,
        input_tensor : Tensor (On GPU)
            The input tensor.
        block_scale : Tensor (On GPU)
            The per-block scaling factor.
        global_scale : Tensor (On GPU)
            The global per-tensor scaling factor. It should contain only 1 element.
    Returns:
        A tuple of two tensors: quantized data tensor in fp4 and quantized block scaling factor tensor in fp8
    """
    # quantize block scale to fp8
    block_scale_quantize_layer = ctx.net.add_quantize(block_scale, global_scale)
    set_layer_name(
        block_scale_quantize_layer,
        target,
        name + "_block_scale_quantize",
        source_ir,
    )
    block_scale_quantize_layer.set_output_type(0, trt.DataType.FP8)
    quantized_block_scale_in_fp8 = block_scale_quantize_layer.get_output(0)

    # dequantize block scale from fp8 to original dtype(default is float32)
    dequantize_block_scale_layer = ctx.net.add_dequantize(
        quantized_block_scale_in_fp8,
        global_scale,
        block_scale.dtype,
    )
    set_layer_name(
        dequantize_block_scale_layer,
        target,
        name + "_dequantize_block_scale",
        source_ir,
    )
    dequantized_block_scale = dequantize_block_scale_layer.get_output(0)

    # quantize input tensor to fp4
    data_quantize_layer = ctx.net.add_quantize(input_tensor, dequantized_block_scale)
    set_layer_name(data_quantize_layer, target, name + "_data_quantize", source_ir)
    data_quantize_layer.set_output_type(0, trt.DataType.FP4)
    quantized_data_in_fp4 = data_quantize_layer.get_output(0)

    # dequantize input tensor from fp4 to originaldtype(default is float32)
    dequantize_data_layer = ctx.net.add_dequantize(
        quantized_data_in_fp4,
        dequantized_block_scale,
        input_tensor.dtype,
    )
    set_layer_name(dequantize_data_layer, target, name + "_dequantize_data", source_ir)
    dequantized_data = dequantize_data_layer.get_output(0)
    return dequantized_data


def _calculate_global_scale(
    ctx: ConversionContext,
    name: str,
    amax: TRTTensor,
) -> TRTTensor:
    # calculate global scale (the global per-tensor scaling factor, should only contain 1 element)
    amax = to_torch(
        amax, None
    )  # amax is calculated from input_tensor.abs().amax().float()
    global_scale = torch.divide(amax, 6 * 448)
    global_scale = get_trt_tensor(ctx, global_scale, name + "_global_scale")
    return global_scale


def _calculate_block_scale(
    ctx: ConversionContext,
    name: str,
    input_tensor: TRTTensor,
    block_size: int,
) -> TRTTensor:

    [n, k] = input_tensor.shape[-2:]
    assert block_size != 0, "block_size must be non-zero"
    assert k % block_size == 0, "k must be a multiple of block_size"
    reshaped_input_tensor = input_tensor.reshape(
        tuple(input_tensor.shape[:-2]) + (n, k // block_size, block_size)
    )
    block_amax = reshaped_input_tensor.abs().amax(dim=-1).float()
    block_scale = torch.divide(block_amax, 6)

    block_scale = get_trt_tensor(ctx, block_scale, name + "_block_scale")
    return block_scale
