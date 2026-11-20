# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

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
    cast_trt_tensor,
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
    unsigned: bool = False,
    narrow_range: bool = True,
) -> TRTTensor:
    """
    Adds quantize and dequantize ops (QDQ) which quantize to INT8 or FP8 based
    on the output_type set and dequantizes them back. A constant weight quantized
    to INT4 gets only a dequantize, see _int4_weight.
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
            if (
                num_bits == 4
                and exponent_bits == 0
                and isinstance(input_tensor, torch.Tensor)
            ):
                return _int4_weight(
                    ctx,
                    target,
                    source_ir,
                    name,
                    input_tensor,
                    amax,
                    unsigned,
                    narrow_range,
                )
            if num_bits != 8 or exponent_bits not in (0, 4):
                raise ValueError(
                    f"quantize converter currently only accept INT8 or FP8 based quantize, or INT4 for a constant weight, got {num_bits=}, {exponent_bits=}"
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
        zero_amax_bound = max_bound / torch.finfo(torch.float32).max
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
            assert amax is not None
            # As in ModelOpt, the scale is computed in float32, so a small float16
            # amax does not round it to 0.
            amax = amax.float()
            if dtype == trt.DataType.FP8:
                # An FP8 scale of 0 zeroes the input or gives NaN. ModelOpt 0.48's FP8
                # fake quantization uses an amax of 1 at or below 448 / float32 max
                # (earlier versions use 2**-24).
                amax = torch.where(amax <= zero_amax_bound, torch.ones_like(amax), amax)
            scale = torch.divide(amax, max_bound)
            scale = get_trt_tensor(ctx, scale, name + "_scale", dtype=torch.float32)
        else:
            amax = cast_trt_tensor(
                ctx, amax, trt.float32, f"{name}_amax_float", target, get_ir(target)
            )
            if dtype == trt.DataType.FP8:
                amax = impl.condition.where(
                    ctx,
                    target,
                    get_ir(target),
                    f"{name}_amax_or_one",
                    torch.tensor(1.0, dtype=torch.float32),
                    amax,
                    impl.elementwise.le(
                        ctx,
                        target,
                        get_ir(target),
                        f"{name}_zero_amax",
                        amax,
                        zero_amax_bound,
                    ),
                )
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


def _int4_weight(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    weight: torch.Tensor,
    amax: Union[np.ndarray, torch.Tensor],
    unsigned: bool,
    narrow_range: bool,
) -> TRTTensor:
    """Quantize a constant weight to INT4 now and add only a dequantize layer.

    TensorRT has no INT4 activations, so it keeps 4-bit weights only when they come
    as an INT4 constant in front of an IDequantizeLayer (weight-only quantization).
    ModelOpt passes the weight as rows of one block each, with one amax per row, and
    reshapes the result back to the weight's shape. TensorRT fuses the dequantize
    into the matmul only when the constant has the matmul weight's shape and the
    scale and output have the weight's type, so the constant takes the shape of that
    reshape. Otherwise it dequantizes the whole weight into scratch memory first.
    """
    if unsigned:
        raise ValueError(
            f"INT4 quantize of {name} is unsigned, but TensorRT INT4 is signed. "
            "Use signed INT4 quantization."
        )
    amax = to_torch(amax, None)
    assert amax is not None
    if (
        weight.dim() != 2
        or amax.numel() != weight.shape[0]
        or (amax.dim() > 0 and amax.shape[0] != weight.shape[0])
    ):
        raise ValueError(
            f"INT4 quantize of {name} expects one amax per block row, got weight "
            f"{tuple(weight.shape)} and amax {tuple(amax.shape)}"
        )
    # Multiplying by the bound before dividing by amax reproduces how ModelOpt's CUDA
    # fake quantization rounds bf16 and fp16 weights; w * (7 / amax) rounds some ties
    # the other way. As in ModelOpt's CUDA kernel, an amax below 2**-24 quantizes its
    # block to 0. So does an amax whose scale rounds to 0 in the weight's type, so
    # that no scale is 0.
    amax = amax.float().reshape(-1, 1)
    max_bound = 7.0
    min_bound = -max_bound if narrow_range else -max_bound - 1
    is_zero = (amax < 2**-24) | ((amax / max_bound).to(weight.dtype) == 0)
    qdata = weight.to(torch.float32, copy=True)
    qdata.mul_(max_bound).div_(amax).round_().clamp_(min_bound, max_bound)
    qdata = qdata.masked_fill_(is_zero, 0.0).to(torch.int8)
    scale = torch.where(is_zero, 1.0, amax / max_bound)

    # ModelOpt reshapes the block rows back to the weight's shape right after the
    # quantize. Without that reshape the weight was not split into blocks, so it
    # already has its own shape.
    shape = tuple(weight.shape)
    users = list(ctx.current_node.users) if ctx.current_node is not None else []
    reshapes = [
        user
        for user in users
        if user.target
        in (
            torch.ops.aten._reshape_copy.default,
            torch.ops.aten._unsafe_view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten.view.default,
        )
    ]
    if reshapes:
        user_shapes = {
            tuple(getattr(user.meta.get("val"), "shape", ())) for user in reshapes
        }
        user_shape = user_shapes.pop() if len(user_shapes) == 1 else ()
        if not (
            len(reshapes) == len(users)
            and user_shape
            and all(isinstance(dim, int) for dim in user_shape)
            and user_shape[-1] % shape[1] == 0
            and qdata.numel() == int(np.prod(user_shape))
        ):
            raise ValueError(
                f"INT4 quantize of {name} could not find the weight's shape from "
                f"the reshape after it ({[str(user.target) for user in users]})."
            )
        shape = user_shape
    # TensorRT has weight-only kernels for matmul weights only. ModelOpt pads a conv
    # kernel's last dim to a whole block, so a 3x3 conv weight would grow many times.
    if len(shape) != 2:
        raise ValueError(
            f"INT4 quantize of {name} supports matmul weights only, got shape {shape}. "
            "Leave this layer out of the INT4 quantization config."
        )
    qdata = qdata.reshape(shape)
    scale = scale.reshape(*shape[:-1], -1)

    # A refittable engine fails to build with a 16-bit INT4 dequantize, so there it
    # dequantizes in float32 and casts, which TensorRT does not fuse.
    dq_dtype = (
        weight.dtype if ctx.compilation_settings.immutable_weights else torch.float32
    )
    qdata_trt = _add_int4_constant(ctx, qdata, f"{name}_int4_weight")
    scale_trt = get_trt_tensor(ctx, scale.to(dq_dtype), f"{name}_scale")
    dequantize_layer = ctx.net.add_dequantize(
        qdata_trt,
        scale_trt,
        output_type=_enums.dtype._from(dq_dtype).to(trt.DataType),
    )
    dequantize_layer.axis = len(shape) - 1
    set_layer_name(dequantize_layer, target, f"{name}_dequantize", source_ir)
    output = dequantize_layer.get_output(0)
    if dq_dtype != weight.dtype:
        output = cast_trt_tensor(
            ctx, output, weight.dtype, f"{name}_cast_output", target, source_ir
        )
    return impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_block_rows", output, tuple(weight.shape)
    )


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
    folding it into a dense high-precision weight. An INT8 weight is the
    exception: TensorRT folds it back into a high-precision weight unless the
    matmul's activations are INT8 too.

    Group-wise INT4 WOQ unpacks to int8 in the PyTorch graph; this converter
    re-packs those nibbles into a trt.DataType.INT4 constant and dequantizes
    it to FP32. Nonzero zero-points are rejected for that path.
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
        # A bf16 output fails to build in a refittable engine.
        trt_output_dtype = trt.DataType.FLOAT
    else:
        int8_qdata = torch.int8 in (input_dtype, getattr(qdata, "dtype", None))
        qdata_trt = get_trt_tensor(
            ctx,
            qdata,
            f"{name}_qdata",
            dtype=input_dtype,
            target_quantized_type=trt.DataType.INT8 if int8_qdata else None,
        )

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
    input_trt: TRTTensor,
    scale: Union[np.ndarray, torch.Tensor, TRTTensor],
    name: str,
) -> tuple[TRTTensor, Optional[int]]:
    axis = None
    scale_for_trt = scale

    if isinstance(scale, torch.Tensor):
        scale_for_trt = scale.squeeze()
        scale_numel = scale_for_trt.numel()
        scale_shape = tuple(scale.shape)
    elif isinstance(scale, np.ndarray):
        scale_for_trt = scale.squeeze()
        scale_numel = scale_for_trt.size
        scale_shape = scale.shape
    else:
        scale_numel = None
        scale_shape = None

    if scale_numel is not None and scale_numel != 1:
        input_shape = tuple(input_trt.shape)
        assert scale_shape is not None
        non_singleton_dims = [dim for dim, size in enumerate(scale_shape) if size != 1]

        if len(scale_shape) == len(input_shape) and len(non_singleton_dims) == 1:
            axis = non_singleton_dims[0]
            input_axis_size = input_shape[axis]
            scale_axis_size = scale_shape[axis]
            if input_axis_size not in (-1, scale_axis_size):
                raise ValueError(
                    f"FP8 scale shape {scale_shape} is incompatible with input "
                    f"shape {input_shape} at axis {axis} for '{name}'"
                )
        else:
            matching_dims = [
                dim for dim, size in enumerate(input_shape) if size == scale_numel
            ]
            if len(matching_dims) != 1:
                raise ValueError(
                    "Unable to derive a unique FP8 quantization axis for "
                    f"'{name}' from input shape {input_shape} and scale shape "
                    f"{scale_shape}; matching input dimensions: {matching_dims}"
                )
            axis = matching_dims[0]

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
    scale_trt, axis = _fp8_scale_and_axis(ctx, input_trt, scale, name)
    quantize_layer = ctx.net.add_quantize(input_trt, scale_trt, trt.DataType.FP8)
    if axis is not None:
        quantize_layer.axis = axis
    set_layer_name(quantize_layer, target, f"{name}_quantize", source_ir)
    return quantize_layer.get_output(0)


def _as_dim_int(value: Union[int, torch.Tensor]) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return value


def dequantize_nvfp4(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    qdata: Union[torch.Tensor, TRTTensor],
    block_scale: Union[torch.Tensor, TRTTensor],
    per_tensor_scale: Union[torch.Tensor, TRTTensor],
    rows: Union[int, torch.Tensor],
    cols: Union[int, torch.Tensor],
    output_dtype: torch.dtype,
) -> TRTTensor:
    """Map torchao_trt.dequantize_nvfp4 to two-level TensorRT dequantize.

    TorchAO stores NVFP4 block scales in a padded/swizzled layout. TensorRT
    wants the logical [N, K/16] FP8 block-scale tensor, an FP4 weight
    constant, and a FP32 global scale. Activations stay high precision; this
    is weight-only storage, not native FP4 MMA.
    """
    if not all(
        isinstance(x, torch.Tensor) for x in (qdata, block_scale, per_tensor_scale)
    ):
        raise ValueError("NVFP4 weight DQ requires constant qdata and scales")

    from torchao.prototype.mx_formats.utils import from_blocked

    rows_i = _as_dim_int(rows)
    cols_i = _as_dim_int(cols)

    with unset_fake_temporarily():
        logical_scale = from_blocked(block_scale, rows_i, cols_i // 16).contiguous()

    qdata_trt = get_trt_tensor(
        ctx,
        qdata,
        f"{name}_qdata_fp4",
        target_quantized_type=trt.DataType.FP4,
    )
    block_scale_trt = get_trt_tensor(
        ctx,
        logical_scale,
        f"{name}_block_scale_fp8",
        target_quantized_type=trt.DataType.FP8,
    )
    global_scale_trt = get_trt_tensor(
        ctx,
        per_tensor_scale.float(),
        f"{name}_global_scale",
        dtype=torch.float32,
        min_rank=0,
    )
    trt_output_dtype = _enums.dtype._from(output_dtype).to(trt.DataType)

    scale_dq = ctx.net.add_dequantize(
        block_scale_trt,
        global_scale_trt,
        output_type=trt_output_dtype,
    )
    scale_dq.axis = -1
    set_layer_name(scale_dq, target, f"{name}_dequantize_scale", source_ir)

    data_dq = ctx.net.add_dequantize(
        qdata_trt,
        scale_dq.get_output(0),
        output_type=trt_output_dtype,
    )
    data_dq.axis = -1
    set_layer_name(data_dq, target, f"{name}_dequantize_data", source_ir)
    return data_dq.get_output(0)


def _e8m0_constant(
    ctx: ConversionContext,
    scale_e8m0: torch.Tensor,
    name: str,
) -> TRTTensor:
    """Create a TRT E8M0 constant from TorchAO float8_e8m0fnu scales."""
    # get_trt_tensor maps uint8 to FP4; build E8M0 weights directly and
    # record the torch buffer so the data_ptr stays alive through build.
    u8 = scale_e8m0.detach().contiguous().view(torch.uint8).cpu()
    weights = to_trt_weights(
        ctx,
        u8,
        name,
        "CONSTANT",
        "CONSTANT",
        dtype=trt.DataType.E8M0,
    )
    constant = ctx.net.add_constant(tuple(u8.shape), weights)
    constant.name = name
    return constant.get_output(0)


def dequantize_mxfp4(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    qdata: Union[torch.Tensor, TRTTensor],
    block_scale_e8m0: Union[torch.Tensor, TRTTensor],
    rows: Union[int, torch.Tensor],
    cols: Union[int, torch.Tensor],
    block_size: Union[int, torch.Tensor],
    output_dtype: torch.dtype,
) -> TRTTensor:
    """Map torchao_trt.dequantize_mxfp4 to TensorRT dequantize.

    TorchAO stores MX scales in a padded/swizzled layout. TensorRT wants
    logical [N, K/block_size] E8M0 scales, an FP4 weight constant, and
    IDequantizeLayer. Myelin accepts FP4 + E8M0 at block size 32; it
    rejects FP4 + float32 scales at that block size. Activations stay high
    precision; this is MXFP4 weight storage, not native MXFP4xMXFP4 MMA.
    """
    if not all(isinstance(x, torch.Tensor) for x in (qdata, block_scale_e8m0)):
        raise ValueError("MXFP4 weight DQ requires constant qdata and scales")

    from torchao.prototype.mx_formats.utils import from_blocked

    rows_i = _as_dim_int(rows)
    cols_i = _as_dim_int(cols)
    block_size_i = _as_dim_int(block_size)
    if cols_i % block_size_i != 0:
        raise ValueError(
            f"cols ({cols_i}) must be divisible by block_size ({block_size_i})"
        )
    n_blocks = cols_i // block_size_i

    with unset_fake_temporarily():
        logical_e8m0 = from_blocked(block_scale_e8m0, rows_i, n_blocks).contiguous()

    qdata_trt = get_trt_tensor(
        ctx,
        qdata,
        f"{name}_qdata_fp4",
        target_quantized_type=trt.DataType.FP4,
    )
    block_scale_trt = _e8m0_constant(
        ctx,
        logical_e8m0,
        f"{name}_block_scale_e8m0",
    )
    trt_output_dtype = _enums.dtype._from(output_dtype).to(trt.DataType)

    data_dq = ctx.net.add_dequantize(
        qdata_trt,
        block_scale_trt,
        output_type=trt_output_dtype,
    )
    data_dq.axis = -1
    set_layer_name(data_dq, target, f"{name}_dequantize_data", source_ir)
    return data_dq.get_output(0)


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
    scale_trt, axis = _fp8_scale_and_axis(ctx, input_trt, scale, name)
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
