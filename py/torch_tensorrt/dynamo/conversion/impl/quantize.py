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
        if isinstance(input_tensor, (torch.Tensor, TRTTensor)):
            if input_tensor.dtype not in (
                trt.float32,
                trt.float16,
                trt.bfloat16,
                torch.bfloat16,
                torch.float16,
                torch.float32,
            ):
                raise ValueError(
                    f"quantize converter received an input of {input_tensor.dtype} type. Supported types: float32 | float16 | bfloat16"
                )
            if num_bits != 8 or exponent_bits not in (0, 4):
                raise ValueError(
                    f"quantize converter currently only accept INT8 or FP8 based quantize, got {num_bits=}, {exponent_bits=}"
                )
        else:
            raise ValueError(
                f"quantize converter received an input of {type(input_tensor)} type. Supported types: torch.Tensor | TRTTensor"
            )

        if num_bits == 8 and exponent_bits == 0:
            dtype = trt.DataType.INT8
            max_bound = 127
        elif num_bits == 8 and exponent_bits == 4:
            dtype = trt.DataType.FP8
            max_bound = 448

        amax = to_torch(amax, None)
        axis = None
        # int8 weight quantization is per-channel quantization(it can have one or multiple amax values)
        if dtype == trt.DataType.INT8 and ".weight_quantizer" in name:
            # if the amax has more than one element, calculate the axis, otherwise axis value will be ignored
            if amax.numel() > 1:
                amax_init_shape = amax.shape
                amax = amax.squeeze().data
                assert (
                    len(amax.shape) == 1
                ), f"TensorRT does not support multi-axis quantization. {name=} {amax_init_shape=} {amax.shape=} "
                axis = list(amax_init_shape).index(list(amax.shape)[0])
                assert (
                    axis == 0
                ), f"{name=} {amax=} is per-channel quantization, expected axis to be 0, but got {axis=}"
        else:
            # int8 activation and fp8 weight/activation quantization is per-tensor quantization, it can only have single amax value
            assert (
                amax.numel() == 1
            ), f"{name=} is per-tensor quantization, expected amax is a singular value, but got {amax.shape=}"
        scale = torch.divide(amax, max_bound)
        scale.masked_fill_(scale == 0, 1.0)
        scale = get_trt_tensor(ctx, scale, name + "_scale")
        input_tensor = get_trt_tensor(ctx, input_tensor, name)

        # Add Q node
        quantize_layer = ctx.net.add_quantize(input_tensor, scale, dtype)
        if axis is not None:
            quantize_layer.axis = axis
        set_layer_name(quantize_layer, target, name + "_quantize", source_ir)
        q_output = quantize_layer.get_output(0)
        # Add DQ node
        dequantize_layer = ctx.net.add_dequantize(
            q_output, scale, output_type=input_tensor.dtype
        )
        dequantize_layer.to_type = input_tensor.dtype
        if axis is not None:
            dequantize_layer.axis = axis
        set_layer_name(dequantize_layer, target, name + "_dequantize", source_ir)
        dq_output = dequantize_layer.get_output(0)

        return dq_output
