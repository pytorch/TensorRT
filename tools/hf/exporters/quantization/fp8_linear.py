"""Export-friendly ModelOpt FP8 linear and TensorRT lowering."""

from __future__ import annotations

from typing import Any

import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.node import Argument, Target
from torch_tensorrt import _enums
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_trt_tensor,
    set_layer_name,
)

FP8_MAX = 448.0


@torch.library.custom_op("edge_export::fp8_linear", mutates_args=())
def fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Reference W8A8 linear used outside TensorRT.

    The TensorRT converter below replaces this implementation with explicit
    FP8 Q/DQ around MatMul. The eager implementation exists for parity checks.
    """
    output_dtype = bias.dtype
    x_fp8 = (x.float() / input_scale.float()).to(torch.float8_e4m3fn)
    x_dq = x_fp8.to(output_dtype) * input_scale.to(output_dtype)
    weight_dq = weight.to(output_dtype) * weight_scale.to(output_dtype)
    linear_bias = None if bias.numel() == 0 else bias.to(output_dtype)
    return F.linear(x_dq, weight_dq, linear_bias)


@fp8_linear.register_fake
def _fp8_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    del weight_scale, input_scale
    return torch.empty(
        (*x.shape[:-1], weight.shape[0]),
        dtype=bias.dtype,
        device=x.device,
    )


class FP8CheckpointLinear(nn.Module):
    """Linear backed by compressed ModelOpt E4M3 weights and scalar scales."""

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_amax: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "ModelOpt FP8 linear requires torch.float8_e4m3fn weight, "
                f"got {weight.dtype}"
            )
        if weight.ndim != 2:
            raise ValueError(f"FP8 linear weight must have rank 2, got {weight.shape}")
        if weight_scale.numel() != 1 or input_amax.numel() != 1:
            raise ValueError("Only per-tensor ModelOpt FP8 scales are supported")

        target = torch.device(device) if device is not None else weight.device
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.register_buffer("weight", weight.to(target).contiguous())
        self.register_buffer(
            "weight_scale",
            weight_scale.reshape(()).to(device=target, dtype=torch.float32),
        )
        self.register_buffer(
            "input_scale",
            (input_amax.reshape(()).float() / FP8_MAX).to(
                device=target, dtype=torch.float32
            ),
        )
        self.register_buffer(
            "bias",
            (
                bias.to(device=target, dtype=dtype).contiguous()
                if bias is not None
                else torch.empty(0, device=target, dtype=dtype)
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.edge_export.fp8_linear.default(
            x,
            self.weight,
            self.weight_scale,
            self.input_scale,
            self.bias,
        )


@dynamo_tensorrt_converter(
    torch.ops.edge_export.fp8_linear.default,
    supports_dynamic_shapes=True,
)
def convert_fp8_linear(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> Any:
    """Lower compressed W8A8 linear to TensorRT explicit FP8 Q/DQ."""
    del kwargs
    x, weight, weight_scale, input_scale, bias = args
    x_trt = get_trt_tensor(ctx, x, f"{name}_input")
    if not isinstance(weight, torch.Tensor):
        raise TypeError("FP8 linear weight must be a frozen torch.Tensor")
    weight_shape = (1,) * (len(x_trt.shape) - 2) + tuple(weight.shape)
    weight_trt = get_trt_tensor(
        ctx,
        weight.reshape(weight_shape),
        f"{name}_weight",
    )
    input_scale_trt = get_trt_tensor(
        ctx,
        input_scale,
        f"{name}_input_scale",
        dtype=torch.float32,
    )
    weight_scale_trt = get_trt_tensor(
        ctx,
        weight_scale,
        f"{name}_weight_scale",
        dtype=torch.float32,
    )

    quantize = ctx.net.add_quantize(x_trt, input_scale_trt, trt.DataType.FP8)
    set_layer_name(quantize, target, f"{name}_input_quantize", SourceIR.ATEN)
    input_dequantize = ctx.net.add_dequantize(
        quantize.get_output(0),
        input_scale_trt,
        output_type=x_trt.dtype,
    )
    set_layer_name(
        input_dequantize,
        target,
        f"{name}_input_dequantize",
        SourceIR.ATEN,
    )

    weight_dequantize = ctx.net.add_dequantize(
        weight_trt,
        weight_scale_trt,
        output_type=x_trt.dtype,
    )
    set_layer_name(
        weight_dequantize,
        target,
        f"{name}_weight_dequantize",
        SourceIR.ATEN,
    )

    matmul = ctx.net.add_matrix_multiply(
        input_dequantize.get_output(0),
        trt.MatrixOperation.NONE,
        weight_dequantize.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    )
    set_layer_name(matmul, target, f"{name}_matmul", SourceIR.ATEN)
    output = matmul.get_output(0)

    if isinstance(bias, torch.Tensor) and bias.numel() != 0:
        bias_shape = (1,) * (len(output.shape) - 1) + (bias.numel(),)
        bias_trt = get_trt_tensor(
            ctx,
            bias.reshape(bias_shape),
            f"{name}_bias",
            dtype=_enums.dtype._from(x_trt.dtype).to(torch.dtype),
        )
        bias_add = ctx.net.add_elementwise(
            output,
            bias_trt,
            trt.ElementWiseOperation.SUM,
        )
        set_layer_name(bias_add, target, f"{name}_bias_add", SourceIR.ATEN)
        output = bias_add.get_output(0)
    output_dtype = _enums.dtype._from(bias.dtype).to(trt.DataType)
    if output.dtype != output_dtype:
        output = cast_trt_tensor(
            ctx,
            output,
            output_dtype,
            f"{name}_output_cast",
            target,
            SourceIR.ATEN,
        )
    return output
