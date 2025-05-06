from typing import Optional, Tuple, Union

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
            input_tensor = get_trt_tensor(ctx, input_tensor, name + "_input")
        if isinstance(input_tensor, TRTTensor) and input_tensor.dtype not in (
            trt.float32,
            trt.float16,
            trt.bfloat16,
        ):
            raise ValueError(
                f"dynamic_block_quantize converter received an input of {input_tensor.dtype} type. Supported types: float32 | float16 | bfloat16"
            )

        # calculate global scale (the global per-tensor scaling factor, should only contain 1 element)
        max_bound = 6
        amax = to_torch(amax, None)
        global_scale = torch.divide(amax, max_bound)
        global_scale = get_trt_tensor(ctx, global_scale, name + "_global_scale")

        if ".weight_quantizer" in name:
            # static double quantization is used for weights
            q_output, q_scale = _static_double_quantize(
                ctx,
                target,
                source_ir,
                name,
                input_tensor,
                global_scale,
            )
            output = _block_double_dequantize(
                ctx,
                target,
                source_ir,
                name,
                q_output,
                q_scale,
                global_scale,
            )
        elif ".input_quantizer" in name:
            # dynamic double quantization is used for inputs
            # Add DYQ node
            q_output, q_scale = _dynamic_quantize(
                ctx,
                target,
                source_ir,
                name,
                input_tensor,
                global_scale,
            )
            # Add double DQ node
            output = _block_double_dequantize(
                ctx,
                target,
                source_ir,
                name,
                q_output,
                q_scale,
                global_scale,
            )
        else:
            raise ValueError(
                f"dynamic_block_quantize converter received an input of {name} name. Supported names: weight_quantizer | input_quantizer"
            )
        return output


def _dynamic_quantize(
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
) -> Tuple[TRTTensor, TRTTensor]:
    """
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
        data_qtype : trt.DataType
            The data type for quantized data. Default is FP4.
        scale_qtype : trt.DataType
            The data type for block scale. Default is FP8.
    Returns:
        A tuple of two tensors: quantized tensor in f4 and block scale tensor.
    """
    if len(input_tensor.shape) not in (2, 3):
        raise ValueError(
            f"dynamic_block_quantize converter received an input of {input_tensor.shape} shape. Supported shapes: 2D or 3D"
        )
    if axis < 0:
        axis = len(input_tensor.shape) + axis
    # Add DYQ node
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
    q_output = dynamic_quantize_layer.get_output(0)
    q_scale = dynamic_quantize_layer.get_output(1)
    return q_output, q_scale


def _block_double_dequantize(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_tensor: TRTTensor,
    scale: TRTTensor,
    global_scale: TRTTensor,
    dtype: trt.DataType = trt.DataType.FLOAT,
) -> TRTTensor:
    """
     Parameters:
     ctx: ConversionContext,
    target: Target,
        source_ir: Optional[SourceIR]
        name: str
        input_tensor : Tensor (On GPU)
            The input tensor.
        scale : Tensor (On GPU)
            The block scale tensor.
        global_scale : Tensor (On GPU)
            The global per-tensor scaling factor. It should contain only 1 element.
        dtype : trt.DataType | str
            The data type for dequantized data. Default is float32.
    Returns:
        The dequantized tensor.
    """
    # dequantize scale from fp8 to dtype(default is float32)
    dequantize_scale_layer = ctx.net.add_dequantize(scale, global_scale, dtype)
    set_layer_name(
        dequantize_scale_layer, target, name + "_dequantize_scale", source_ir
    )
    dequantized_scale = dequantize_scale_layer.get_output(0)

    # dequantize input_tensor from  fp4 to dtype(default is float32)
    dequantize_data_layer = ctx.net.add_dequantize(
        input_tensor, dequantized_scale, dtype
    )
    set_layer_name(dequantize_data_layer, target, name + "_dequantize_data", source_ir)
    dq_output = dequantize_data_layer.get_output(0)
    return dq_output


def _static_double_quantize(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_tensor: TRTTensor,
    global_scale: TRTTensor,
) -> Tuple[TRTTensor, TRTTensor]:
    """
    Parameters:
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
        input_tensor : Tensor (On GPU)
            The input tensor.
        global_scale : Tensor (On GPU)
            The global per-tensor scaling factor. It should contain only 1 element.
    Returns:
        A tuple of two tensors: quantized tensor and scaling factor tensor
    """
    pass
    return input_tensor, global_scale
    # quantize_layer = ctx.net.add_quantize(input_tensor, global_scale)
    # set_layer_name(quantize_layer, target, name + "_quantize", source_ir)
    # q_output = quantize_layer.get_output(0)
    # q_scale = quantize_layer.get_output(1)

    # return q_output, q_scale
