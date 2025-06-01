from typing import Optional, Union

import numpy as np
import tensorrt as trt
import torch
import torch_tensorrt.dynamo.conversion.impl as impl
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
            input_tensor = get_trt_tensor(ctx, input_tensor, name)
            if input_tensor.dtype not in (
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
        else:
            raise ValueError(
                f"quantize converter received an input of {type(input_tensor)} type. Supported types: torch.Tensor | TRTTensor"
            )

        if num_bits == 8 and exponent_bits == 0:
            max_bound = 127
        elif num_bits == 8 and exponent_bits == 4:
            max_bound = 448

        if not isinstance(amax, trt.ITensor):
            # for the int8/fp8 quantization, it is per tensor quantization, so amax should be a singlular element
            # TODO: to confirm with ModelOpt team, why I am getting a amax with shape > 1
            if len(amax.shape) >= 1:
                amax = amax.abs().amax()
                if amax == 0.0:
                    amax = 1.0
            amax = to_torch(amax, None)
            scale = torch.divide(amax, max_bound)
            scale = get_trt_tensor(ctx, amax, name + "_scale")
        else:
            scale = impl.elementwise_div(
                ctx, target, source_ir, name + "_scale", amax, max_bound
            )

        if num_bits == 8 and exponent_bits == 0:
            dtype = trt.DataType.INT8
        elif num_bits == 8 and exponent_bits == 4:
            dtype = trt.DataType.FP8

        # Add Q node
        quantize_layer = ctx.net.add_quantize(input_tensor, scale, dtype)
        set_layer_name(quantize_layer, target, name + "_quantize", source_ir)
        q_output = quantize_layer.get_output(0)
        # Add DQ node
        dequantize_layer = ctx.net.add_dequantize(
            q_output, scale, output_type=input_tensor.dtype
        )
        dequantize_layer.to_type = input_tensor.dtype
        set_layer_name(dequantize_layer, target, name + "_dequantize", source_ir)
        dq_output = dequantize_layer.get_output(0)

        return dq_output
