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
    with unset_fake_temporarily():
        axis = len(input_tensor.shape) - 1
        global_scale = _calculate_global_scale(ctx, name, amax)
        if ".weight_quantizer" in name:
            _test_weights_scaling_factor(input_tensor, global_scale)
            output = _static_double_quantize_without_constant_folding(
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
def _test_weights_scaling_factor(weights_tensor, global_scale):

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

    # import modelopt.onnx.quantization.quant_utils as quant_utils

    block_scale_fp32 = nvfp4_tensor.NVFP4QTensor.get_weights_scaling_factor(
        weights_tensor, 16, global_scale, True
    )[0]
    block_scale_fp8 = block_scale_fp32.to(torch.float8_e4m3fn)

    global_scale = to_torch(global_scale, None)

    # # TODO: issue1: not sure whether we need to quantize the weights tensor here, due to Icast layer does not support cast
    # IBuilder::buildSerializedNetwork: Error Code 4: API Usage Error (Cast ITensor linear1.weight_quantizer/dynamic_block_quantize_op_1_weights_tensor_scaled_output from DataType.FLOAT to DataType.FP4 - [unknown_ir_ops]-[linear1.weight_quantizer/dynamic_block_quantize_op_1_cast_weights_tensor_scaled_to_fp4]: unsupported input type and output type for ICastLayer, unsupported types are: {FP8, Int4, FP4}, current input type: Float, output type: FP4)
    # reference https://gitlab-master.nvidia.com/omniml/modelopt/-/blob/main/modelopt/onnx/quantization/qdq_utils.py#L955
    # weights_tensor_scaled = quant_utils.quantize(weights_tensor.numpy(), 16, block_scale_fp32.numpy(),global_scale.numpy())
    # weights_tensor_scaled = torch.from_numpy(weights_tensor_scaled)
    # weights_tensor_scaled = get_trt_tensor(ctx, weights_tensor_scaled, name + "_weights_tensor_scaled")
    # weights_fp4 = cast_trt_tensor(ctx, weights_tensor_scaled, trt.DataType.FP4, name + "_cast_weights_tensor_scaled_to_fp4")

    # # TODO: issue2: weights_tensor_scaled is in torch.uint8 format not sure how can this to be converted into float4_e2m1fn_x2
    # reference: https://gitlab-master.nvidia.com/omniml/modelopt/-/blob/main/modelopt/core/torch/quantization/qtensor/nvfp4_tensor.py#L136
    weights_tensor_scaled = nvfp4_tensor.NVFP4QTensor.quantize(
        weights_tensor,
        16,
        block_scale_fp32,
        global_scale,
    )[0]._quantized_data

    # # TODO: issue3: torch does not support convert to float4_e2m1fn_x2 directly got RuntimeError: "copy_kernel" not implemented for 'Float4_e2m1fn_x2'
    # weights_fp4 = weights_tensor_scaled.to(torch.float4_e2m1fn_x2)
    # weights_fp4 = get_trt_tensor(ctx, weights_fp4, name + "_weights_fp4")

    global_scale = get_trt_tensor(ctx, global_scale, name + "_global_scale")
    block_scale = get_trt_tensor(ctx, block_scale_fp32, name + "_block_scale")
    block_scale_fp8 = get_trt_tensor(ctx, block_scale_fp8, name + "_block_scale_fp8")
    # # quantize block scale to fp8
    # block_scale_quantize_layer = ctx.net.add_quantize(block_scale, global_scale)
    # set_layer_name(
    #     block_scale_quantize_layer,
    #     target,
    #     name + "_block_scale_quantize",
    #     source_ir,
    # )
    # block_scale_quantize_layer.set_output_type(0, trt.DataType.FP8)
    # quantized_block_scale_in_fp8 = block_scale_quantize_layer.get_output(0)

    # dequantize block scale from fp8 to float32
    dequantize_block_scale_layer = ctx.net.add_dequantize(
        block_scale_fp8,
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

    # dequantize weights tensor from fp4 to originaldtype(default is float32)
    dequantize_data_layer = ctx.net.add_dequantize(
        weights_fp4,
        dequantized_block_scale,
        trt.DataType.FLOAT,
    )
    set_layer_name(dequantize_data_layer, target, name + "_dequantize_data", source_ir)
    dequantized_data = dequantize_data_layer.get_output(0)
    return dequantized_data


def _static_double_quantize_without_constant_folding(
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

    # import modelopt.onnx.quantization.quant_utils as quant_utils

    block_scale = nvfp4_tensor.NVFP4QTensor.get_weights_scaling_factor(
        weights_tensor, 16, global_scale, True
    )[0]
    global_scale = to_torch(global_scale, None)

    # block_scale_fp8 = block_scale.to(torch.float8_e4m3fn)
    # block_scale_fp8 = get_trt_tensor(ctx, block_scale_fp8, name + "_block_scale_fp8")

    global_scale = get_trt_tensor(ctx, global_scale, name + "_global_scale")
    block_scale = get_trt_tensor(ctx, block_scale, name + "_block_scale")
    weights_tensor = get_trt_tensor(ctx, weights_tensor, name + "_weights_tensor")

    # quantize block scale to fp8
    block_scale_quantize_layer = ctx.net.add_quantize(block_scale, global_scale)
    set_layer_name(
        block_scale_quantize_layer,
        target,
        name + "_block_scale_quantize_to_fp8",
        source_ir,
    )
    block_scale_quantize_layer.set_output_type(0, trt.DataType.FP8)
    block_scale_fp8 = block_scale_quantize_layer.get_output(0)

    # dequantize block scale from fp8 to float32
    dequantize_block_scale_layer = ctx.net.add_dequantize(
        block_scale_fp8,
        global_scale,
        block_scale.dtype,
    )
    set_layer_name(
        dequantize_block_scale_layer,
        target,
        name + "_dequantize_block_scale_from_fp8",
        source_ir,
    )
    dequantized_block_scale = dequantize_block_scale_layer.get_output(0)

    # quantize weights tensor to fp4
    quantize_weights_layer = ctx.net.add_quantize(
        weights_tensor, dequantized_block_scale
    )
    set_layer_name(
        quantize_weights_layer,
        target,
        name + "_quantize_weights_to_fp4",
        source_ir,
    )
    quantize_weights_layer.set_output_type(0, trt.DataType.FP4)
    weights_fp4 = quantize_weights_layer.get_output(0)

    # dequantize weights tensor from fp4 to originaldtype(default is float32)
    dequantize_weights_layer = ctx.net.add_dequantize(
        weights_fp4,
        dequantized_block_scale,
        trt.DataType.FLOAT,
    )
    set_layer_name(
        dequantize_weights_layer,
        target,
        name + "_dequantize_weights_from_fp4",
        source_ir,
    )
    dequantized_data = dequantize_weights_layer.get_output(0)
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


def _calculate_block_scale(
    ctx: ConversionContext,
    name: str,
    weights_tensor: TRTTensor,
    block_size: int,
) -> TRTTensor:
    amax = weights_tensor.abs().amax().float()
    # reference: https://gitlab-master.nvidia.com/omniml/modelopt/-/blob/main/modelopt/onnx/quantization/quant_utils.py#L122
    weights_scaling_factor_2 = amax / 6 / 448
    if weights_scaling_factor_2 == 0:
        weights_scaling_factor_2 = 1.0

    # reference: https://gitlab-master.nvidia.com/omniml/modelopt/-/blob/main/modelopt/onnx/quantization/quant_utils.py#L131
    [n, k] = weights_tensor.shape[-2:]
    assert block_size != 0, "block_size must be non-zero"
    assert k % block_size == 0, "k must be a multiple of block_size"
    reshaped_input_tensor = weights_tensor.reshape(
        tuple(weights_tensor.shape[:-2]) + (n, k // block_size, block_size)
    )

    per_block_amax = reshaped_input_tensor.abs().amax(dim=-1).float()
    per_block_scale = torch.divide(per_block_amax, 6)
    q_per_block_scale = torch.divide(per_block_scale, weights_scaling_factor_2)
    # TODO:set all zero values in scale to 1.0
    # block_scale = get_trt_tensor(ctx, q_per_block_scale, name + "_block_scale")
    return q_per_block_scale
