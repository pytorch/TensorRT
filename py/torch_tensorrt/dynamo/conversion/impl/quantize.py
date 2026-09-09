from typing import Optional, Sequence, Union

import numpy as np
import tensorrt as trt
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch.fx.node import Target
from torch_tensorrt import _enums
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_trt_tensor,
    set_layer_name,
    to_torch,
    to_trt_weights,
)


def get_ir(target: Target) -> SourceIR:
    target_module = getattr(target, "__module__", "None")
    if any(
        target_module.startswith(prefix)
        for prefix in ("torch.ops.aten", "torch._ops.aten")
    ):
        return SourceIR.ATEN
    elif any(
        target_module.startswith(prefix)
        for prefix in ("torch.ops.prims", "torch._ops.prims")
    ):
        return SourceIR.PRIM
    elif target_module.startswith("torch.nn"):
        return SourceIR.NN

    return SourceIR.UNKNOWN


def quantize(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_tensor: TRTTensor,
    amax: Union[np.ndarray, torch.Tensor, TRTTensor],
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

        axis = None
        # Dynamic amax (TRT ITensor) is always treated as per-tensor; numel()/shape
        # checks only apply to constant torch/numpy amax values.
        if not isinstance(amax, trt.ITensor):
            # int8 weight quantization is per-channel quantization(it can have one or multiple amax values)
            if dtype == trt.DataType.INT8 and amax.numel() > 1:
                # if the amax has more than one element, calculate the axis, otherwise axis value will be ignored
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

            amax = to_torch(amax, None)
            scale = torch.divide(amax, max_bound)
            scale = get_trt_tensor(ctx, scale, name + "_scale", dtype=torch.float32)
        else:
            scale = impl.elementwise.div(
                ctx,
                target,
                get_ir(target),
                name,
                amax,
                max_bound,
            )
            scale = get_trt_tensor(ctx, scale, name + "_scale", dtype=torch.float32)

        # Add Q node
        if num_bits == 8 and exponent_bits == 0:
            dtype = trt.DataType.INT8
        elif num_bits == 8 and exponent_bits == 4:
            dtype = trt.DataType.FP8

        if not isinstance(input_tensor, TRTTensor):
            input_tensor = get_trt_tensor(ctx, input_tensor, name + "_quantize_input")

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
        if axis is not None:
            dequantize_layer.axis = axis
        set_layer_name(dequantize_layer, target, name + "_dequantize", source_ir)
        dq_output = dequantize_layer.get_output(0)

        return dq_output


def _block_size_as_ints(block_size: Sequence[object]) -> list[int]:
    dims: list[int] = []
    for bs in block_size:
        if isinstance(bs, torch.Tensor):
            dims.append(int(bs.item()))
        elif isinstance(bs, (int, float)):
            dims.append(int(bs))
        else:
            raise TypeError(f"Unsupported block_size dim type {type(bs)}: {bs!r}")
    return dims


def _pack_int4_nibbles(qdata: torch.Tensor) -> torch.Tensor:
    """Pack adjacent INT4 values into bytes (mslk / TensorRT nibble order).

    Even columns go in the low nibble, odd columns in the high nibble.
    """
    if qdata.dtype != torch.int8:
        raise ValueError(f"expected int8 qdata for INT4 packing, got {qdata.dtype}")
    if qdata.shape[-1] % 2 != 0:
        raise ValueError(
            f"K dim must be even for INT4 packing, got {tuple(qdata.shape)}"
        )
    low = torch.bitwise_and(qdata[..., ::2], 0x0F)
    high = torch.bitwise_left_shift(qdata[..., 1::2], 4)
    return torch.bitwise_or(low, high).contiguous()


def _is_groupwise_int4(
    qdata: Union[torch.Tensor, TRTTensor],
    block_size: Sequence[object],
) -> bool:
    """True when DQ is group-wise INT4 (blocked scale along the last dim).

    Per-row INT8 uses block_size=[1, K] with K == qdata.shape[-1], which
    also has block_size[-1] > 1. That is *not* INT4 groupwise.
    """
    if not isinstance(qdata, torch.Tensor) or qdata.dtype != torch.int8:
        return False
    try:
        bs = _block_size_as_ints(block_size)
    except TypeError:
        return False
    if not bs or bs[-1] <= 1:
        return False
    if bs[-1] >= qdata.shape[-1]:
        return False
    return True


def _add_int4_constant(
    ctx: ConversionContext, qdata: torch.Tensor, name: str
) -> TRTTensor:
    """Create a TRT constant with logical shape (N, K) and packed INT4 weights."""
    packed = _pack_int4_nibbles(qdata.contiguous().cpu())
    weights = to_trt_weights(
        ctx,
        packed,
        name,
        "CONSTANT",
        "CONSTANT",
        dtype=trt.DataType.INT4,
        count=qdata.numel(),
    )
    constant = ctx.net.add_constant(list(qdata.shape), weights)
    constant.name = name
    return constant.get_output(0)


def _zero_point_is_nonzero(zero_point: object) -> bool:
    if zero_point is None:
        return False
    if isinstance(zero_point, torch.Tensor):
        return bool(torch.any(zero_point != 0).item())
    return True


def dequantize_affine(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    qdata: Union[torch.Tensor, TRTTensor],
    block_size: Sequence[object],
    scale: Union[np.ndarray, torch.Tensor, TRTTensor],
    output_dtype: torch.dtype,
    input_dtype: Optional[torch.dtype] = None,
    zero_point: Optional[object] = None,
) -> TRTTensor:
    """Map torchao.dequantize_affine to TensorRT IDequantizeLayer.

    Used by TorchAO weight-only quantization (FP8/INT8/INT4). The quantized
    weight stays a low-precision constant and is dequantized at the GEMM
    boundary so Myelin can fuse DQ into the matmul prologue instead of
    folding it into a dense high-precision weight.

    Group-wise INT4 WOQ unpacks to int8 in the PyTorch graph; this converter
    re-packs those nibbles into a trt.DataType.INT4 constant. TensorRT
    blocked INT4 DQ currently wants an FP32 DQ output (cast back to BF16
    after). Nonzero zero-points are rejected for that path.
    """
    use_int4 = _is_groupwise_int4(qdata, block_size)

    if use_int4:
        if _zero_point_is_nonzero(zero_point):
            raise RuntimeError(
                "TensorRT IDequantizeLayer rejects nonzero zero_point; "
                "group-wise INT4 WOQ must use symmetric quantization "
                f"(got nonzero zero_point for '{name}')"
            )
        assert isinstance(qdata, torch.Tensor)
        qdata_trt = _add_int4_constant(ctx, qdata, f"{name}_qdata_int4")
        # Blocked along the group dimension (last dim of block_size > 1).
        axis: Optional[int] = len(_block_size_as_ints(block_size)) - 1
        scale_for_trt = scale
        # Myelin currently matches INT4 blocked DQ -> FP32, not BF16.
        trt_output_dtype = trt.DataType.FLOAT
    else:
        qdata_trt = get_trt_tensor(ctx, qdata, f"{name}_qdata", dtype=input_dtype)

        axis = None
        if isinstance(scale, torch.Tensor):
            scale_for_trt = scale.squeeze()
            if scale_for_trt.numel() != 1:
                # Per-channel axis is the dimension whose block size is 1
                # (quantized independently per slice). Example: weight (3072, 64)
                # with block_size [3072, 1] → axis 1.
                bs = _block_size_as_ints(block_size)
                try:
                    axis = next(i for i, dim in enumerate(bs) if dim == 1)
                except StopIteration as exc:
                    raise ValueError(
                        f"Unable to derive IDequantizeLayer axis from block_size={bs} "
                        f"and scale shape {tuple(scale.shape)}"
                    ) from exc
        else:
            scale_for_trt = scale

        trt_output_dtype = _enums.dtype._from(output_dtype).to(trt.DataType)

    scale_trt = get_trt_tensor(ctx, scale_for_trt, f"{name}_scale", dtype=torch.float32)

    dequantize_layer = ctx.net.add_dequantize(
        qdata_trt,
        scale_trt,
        output_type=trt_output_dtype,
    )
    if axis is not None:
        dequantize_layer.axis = axis
    set_layer_name(dequantize_layer, target, f"{name}_dequantize", source_ir)
    return dequantize_layer.get_output(0)


def _fp8_scale_and_axis(
    ctx: ConversionContext,
    scale: Union[np.ndarray, torch.Tensor, TRTTensor],
    name: str,
) -> tuple[TRTTensor, Optional[int]]:
    axis = None
    if isinstance(scale, torch.Tensor):
        scale_for_trt = scale.squeeze()
        if scale_for_trt.numel() != 1:
            axis = 0
    else:
        scale_for_trt = scale
    scale_trt = get_trt_tensor(ctx, scale_for_trt, f"{name}_scale", dtype=torch.float32)
    return scale_trt, axis


def quantize_affine_float8(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_tensor: Union[torch.Tensor, TRTTensor],
    scale: Union[np.ndarray, torch.Tensor, TRTTensor],
) -> TRTTensor:
    """Map TorchAO quantize_affine_float8_non_decomposed to IQuantizeLayer."""
    input_trt = get_trt_tensor(ctx, input_tensor, f"{name}_input")
    scale_trt, axis = _fp8_scale_and_axis(ctx, scale, name)
    quantize_layer = ctx.net.add_quantize(input_trt, scale_trt, trt.DataType.FP8)
    if axis is not None:
        quantize_layer.axis = axis
    set_layer_name(quantize_layer, target, f"{name}_quantize", source_ir)
    return quantize_layer.get_output(0)


def dequantize_affine_float8(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_tensor: Union[torch.Tensor, TRTTensor],
    scale: Union[np.ndarray, torch.Tensor, TRTTensor],
    output_dtype: torch.dtype = torch.bfloat16,
) -> TRTTensor:
    """Map TorchAO dequantize_affine_float8_non_decomposed to IDequantizeLayer."""
    input_trt = get_trt_tensor(ctx, input_tensor, f"{name}_input")
    scale_trt, axis = _fp8_scale_and_axis(ctx, scale, name)
    trt_output_dtype = _enums.dtype._from(output_dtype).to(trt.DataType)
    dequantize_layer = ctx.net.add_dequantize(
        input_trt,
        scale_trt,
        output_type=trt_output_dtype,
    )
    if axis is not None:
        dequantize_layer.axis = axis
    set_layer_name(dequantize_layer, target, f"{name}_dequantize", source_ir)
    return dequantize_layer.get_output(0)
