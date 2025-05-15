from typing import Optional, Union

import numpy as np
import tensorrt as trt
import torch
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_trt_tensor,
    to_torch,
)
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTTensor


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
    if len(input_tensor.shape) not in (2, 3):
        raise ValueError(
            f"nvfp4_quantize converter received an input of {input_tensor.shape} shape. Supported shapes: 2D or 3D"
        )
    with unset_fake_temporarily():
        axis = len(input_tensor.shape) - 1
        global_scale = _calculate_global_scale(ctx, name, amax)
        if ".weight_quantizer" in name:
            _test_weights_scaling_factor(input_tensor, global_scale)
            output = _static_double_quantize(
                ctx,
                target,
                source_ir,
                name,
                input_tensor,
                global_scale,
                axis,
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
                axis,
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
    input_tensor: torch.Tensor,
    global_scale: torch.Tensor,
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
    global_scale = get_trt_tensor(ctx, global_scale, name + "_global_scale")
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
    dequantize_data_layer.axis = axis
    set_layer_name(dequantize_data_layer, target, name + "_dequantize_data", source_ir)
    dequantized_data = dequantize_data_layer.get_output(0)
    return dequantized_data


# TODO: to remove it this is to make sure our global scale and block scale calculation is correct during debugging
def _test_weights_scaling_factor(
    weights_tensor: torch.Tensor, 
    global_scale: torch.Tensor
) -> None:

    import modelopt.core.torch.quantization.qtensor.nvfp4_tensor as nvfp4_tensor
    import modelopt.onnx.quantization.quant_utils as quant_utils

    weights_scaling_factor_2 = nvfp4_tensor.NVFP4QTensor.get_weights_scaling_factor_2(
        weights_tensor
    )
    torch.allclose(weights_scaling_factor_2, global_scale)

    block_scale_f32 = quant_utils.get_weights_scaling_factor(
        weights_tensor.numpy(), 16, np.float32(global_scale)
    )
    block_scale_f32 = torch.from_numpy(block_scale_f32)

    block_scale = nvfp4_tensor.NVFP4QTensor.get_weights_scaling_factor(
        weights_tensor, 16, global_scale, True
    )[0]
    torch.allclose(block_scale_f32, block_scale)
    block_scale_fp8 = block_scale.to(torch.float8_e4m3fn)


def _static_double_quantize(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    weights_tensor: torch.Tensor,
    global_scale: torch.Tensor,
    axis: int,
) -> TRTTensor:
    """
    Parameters:
        ctx: ConversionContext,
        target: Target,
        source_ir: Optional[SourceIR],
        name: str,
        weights_tensor : Tensor (On GPU)
            The input tensor for weights.
        global_scale : Tensor (On GPU)
            The global per-tensor scaling factor. It should contain only 1 element.
        axis: int
            The axis to quantize. Default is -1 (the last axis).
    Returns:
        quantized data tensor in fp4
    """

    import modelopt.core.torch.quantization.qtensor.nvfp4_tensor as nvfp4_tensor

    block_scale_fp8 = nvfp4_tensor.NVFP4QTensor.get_weights_scaling_factor(
        weights_tensor, 16, global_scale,
    )[0]

    weights_tensor_scaled = nvfp4_tensor.NVFP4QTensor.quantize(
        weights_tensor,
        16,
        block_scale_fp8,
        global_scale,
    )[0]._quantized_data

    block_scale_fp8 = get_trt_tensor(ctx, block_scale_fp8, name + "_block_scale_fp8")
    global_scale = to_torch(global_scale, None)
    global_scale = get_trt_tensor(ctx, global_scale, name + "_global_scale")
    weights_fp4_represented_in_uint8 = get_trt_tensor(ctx, weights_tensor_scaled, name + "_weights_fp4_represented_in_uint8")

    # dequantize block scale from fp8 to float32
    dequantize_block_scale_layer = ctx.net.add_dequantize(
        block_scale_fp8,
        global_scale,
        trt.DataType.FLOAT,
    )
    set_layer_name(
        dequantize_block_scale_layer,
        target,
        name + "_dequantize_block_scale",
        source_ir,
    )
    dequantized_block_scale = dequantize_block_scale_layer.get_output(0)

    # dequantize weights tensor from fp4 to originaldtype(default is float32)
    dequantize_data_layer = ctx.net.add_dequantize(
        weights_fp4_represented_in_uint8,
        dequantized_block_scale,
        trt.DataType.FLOAT,
    )
    dequantize_data_layer.precision = trt.DataType.FP4
    set_layer_name(dequantize_data_layer, target, name + "_dequantize_data", source_ir)
    dequantized_data = dequantize_data_layer.get_output(0)
    return dequantized_data


def _calculate_global_scale(
    ctx: ConversionContext,
    name: str,
    amax: torch.Tensor,
) -> torch.Tensor:
    # calculate global scale (the global per-tensor scaling factor, should only contain 1 element)
    if amax is None or amax == 0:
        amax = 1.0
    amax = to_torch(
        amax, None
    )  # amax is calculated from input_tensor.abs().amax().float()
    global_scale = torch.divide(amax, 6 * 448)
    if global_scale == 0:
        global_scale = 1.0
    return global_scale

