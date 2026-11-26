# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import FrozenSet, Optional, Sequence, Tuple, Union

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

_LOGGER: logging.Logger = logging.getLogger(__name__)

# The quantized_decomposed ops that PT2E quantization emits. Their converters keep the integer
# results of a quantize in float32, where every int8 and uint8 value is exact. That is only
# correct while those values stay inside one engine and reach a dequantize or an engine
# output, which TRTInterpreter casts back to the op's dtype using its output_dtypes. Code
# that relies on this: quantized_decomposed_validator, makes_uint8_tensor, the torch.cond
# subgraph check, and the partitioners, which keep a quantize with its dequantizes (see
# quantize_read_by).
QUANTIZED_DECOMPOSED_QUANTIZE_OPS: FrozenSet[torch._ops.OpOverload] = frozenset()
QUANTIZED_DECOMPOSED_DEQUANTIZE_OPS: FrozenSet[torch._ops.OpOverload] = frozenset()
QUANTIZED_DECOMPOSED_PER_CHANNEL_OPS: FrozenSet[torch._ops.OpOverload] = frozenset()
try:
    # Importing the module registers the ops. It lives in torch.ao.quantization, which PyTorch
    # is moving out to torchao, so any failure here only leaves the ops unconverted.
    import torch.ao.quantization.fx._decomposed  # noqa: F401

    _qd = torch.ops.quantized_decomposed
    (
        QUANTIZED_DECOMPOSED_QUANTIZE_OPS,
        QUANTIZED_DECOMPOSED_DEQUANTIZE_OPS,
        QUANTIZED_DECOMPOSED_PER_CHANNEL_OPS,
    ) = (
        frozenset(
            {
                _qd.quantize_per_tensor.default,
                _qd.quantize_per_tensor.tensor,
                _qd.quantize_per_channel.default,
            }
        ),
        frozenset(
            {
                _qd.dequantize_per_tensor.default,
                _qd.dequantize_per_tensor.tensor,
                _qd.dequantize_per_channel.default,
            }
        ),
        frozenset(
            {_qd.quantize_per_channel.default, _qd.dequantize_per_channel.default}
        ),
    )
except Exception as e:
    _LOGGER.debug(
        f"quantized_decomposed ops are not available, so PyTorch runs them: {e}"
    )


def quantize_read_by(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """Return the quantize whose result the dequantize node reads, or None."""
    if node.target not in QUANTIZED_DECOMPOSED_DEQUANTIZE_OPS:
        return None
    source = node.args[0]
    if (
        isinstance(source, torch.fx.Node)
        and source.target in QUANTIZED_DECOMPOSED_QUANTIZE_OPS
    ):
        return source
    return None


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


def _affine_constants(
    scale: Union[float, torch.Tensor],
    zero_point: Union[int, torch.Tensor, None],
    axis: Optional[int],
    rank: int,
    reciprocal: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return the float32 scale, or its reciprocal, and zero point of a quantized_decomposed op,
    shaped to broadcast against its input, rounded to float32 the way PyTorch rounds them.
    """
    with unset_fake_temporarily():
        if axis is None:
            # The .tensor overloads read their one value with item(), as PyTorch does.
            scale_value = scale.item() if isinstance(scale, torch.Tensor) else scale
            zero_point_value = (
                zero_point.item()
                if isinstance(zero_point, torch.Tensor)
                else zero_point
            )
            return torch.tensor(
                1.0 / scale_value if reciprocal else scale_value, dtype=torch.float32
            ), torch.tensor(zero_point_value, dtype=torch.float32)
        assert isinstance(scale, torch.Tensor)
        shape = [1] * rank
        shape[axis] = -1
        # PyTorch computes 1.0 / scales in the dtype of the scales, which the validator
        # requires to be float32.
        scale_tensor = 1.0 / scale if reciprocal else scale
        zero_point_tensor = (
            None
            if zero_point is None
            else torch.as_tensor(zero_point).to(torch.float32).reshape(shape)
        )
        return scale_tensor.reshape(shape), zero_point_tensor


def quantize_decomposed(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: Union[torch.Tensor, TRTTensor],
    scale: Union[float, torch.Tensor],
    zero_point: Union[int, torch.Tensor],
    quant_min: int,
    quant_max: int,
    axis: Optional[int] = None,
) -> TRTTensor:
    """Lower quantized_decomposed.quantize_per_tensor and quantize_per_channel to their
    definition, clamp(round(x * (1 / scale)) + zero_point, quant_min, quant_max).

    TensorRT's quantize layer is symmetric and has no uint8 output, so it cannot express the
    asymmetric uint8 quantization PT2E often uses. The result keeps the integer values in
    float32, where every int8 and uint8 value is exact; it is the value PyTorch's final cast
    gives. A result that leaves the engine is cast to the op's dtype there.

    When a dequantize follows in the same engine, TensorRT can fuse the two and multiply by the
    scalar scale constant one unit in the last place off, so a value that close to a halfway
    point can round to the neighboring integer. A quantize alone matched PyTorch exactly in
    testing.

    A NaN input gives 0, the value PyTorch's cast to int8 or uint8 gives it on CPU and CUDA,
    although C++ leaves that cast undefined. That select also keeps a version compatible
    engine buildable: TensorRT 11.4 cannot cast a clamp's result to an int8 engine output.
    """
    input_trt = get_trt_tensor(ctx, input_val, f"{name}_input")
    # PyTorch quantizes float16 and bfloat16 inputs in float32.
    x = cast_trt_tensor(
        ctx, input_trt, trt.float32, f"{name}_input_float32", target, source_ir
    )
    inv_scale, zero_point_float = _affine_constants(
        scale, zero_point, axis, len(x.shape), reciprocal=True
    )
    x = impl.elementwise.mul(
        ctx,
        target,
        source_ir,
        f"{name}_mul_inv_scale",
        x,
        get_trt_tensor(ctx, inv_scale, f"{name}_inv_scale"),
    )
    # Like torch.round, TensorRT's ROUND rounds halfway values to the nearest even integer.
    x = impl.unary.round(ctx, target, source_ir, f"{name}_round", x)
    assert zero_point_float is not None
    x = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_add_zero_point",
        x,
        get_trt_tensor(ctx, zero_point_float, f"{name}_zero_point"),
    )
    x = impl.elementwise.clamp(
        ctx, target, source_ir, f"{name}_clamp", x, float(quant_min), float(quant_max)
    )
    return impl.condition.where(
        ctx,
        target,
        source_ir,
        f"{name}_nan_to_zero",
        x,
        get_trt_tensor(ctx, 0.0, f"{name}_nan_value", torch.float32, min_rank=0),
        impl.elementwise.eq(ctx, target, source_ir, f"{name}_not_nan", x, x),
    )


def _quantized_values_as_float32(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: Union[torch.Tensor, TRTTensor],
) -> TRTTensor:
    """Return the integer values a dequantize reads as a float32 tensor."""
    if not isinstance(input_val, TRTTensor):
        with unset_fake_temporarily():
            return get_trt_tensor(ctx, input_val.to(torch.float32), name)
    if input_val.dtype == trt.int8:
        # TensorRT reads an int8 tensor only through a dequantize layer. A scale of 1 keeps
        # the values exact.
        one = get_trt_tensor(ctx, 1.0, f"{name}_unit_scale", torch.float32, min_rank=0)
        layer = ctx.net.add_dequantize(input_val, one, output_type=trt.float32)
        set_layer_name(layer, target, f"{name}_dequantize", source_ir)
        return layer.get_output(0)
    # A converted quantize already gives float32; an engine input may be uint8.
    return cast_trt_tensor(ctx, input_val, trt.float32, name, target, source_ir)


def dequantize_decomposed(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: Union[torch.Tensor, TRTTensor],
    scale: Union[float, torch.Tensor],
    zero_point: Union[int, torch.Tensor, None],
    out_dtype: Optional[torch.dtype],
    axis: Optional[int] = None,
) -> TRTTensor:
    """Lower quantized_decomposed.dequantize_per_tensor and dequantize_per_channel to their
    definition, (q - zero_point) * scale, computed in float32 and cast to out_dtype. Like
    PyTorch, the per-tensor op first rounds q - zero_point to out_dtype.

    The input is the float32 result of a converted quantize, an int8 or uint8 engine input,
    or a constant.
    """
    out_dtype = out_dtype or torch.float32
    x = _quantized_values_as_float32(ctx, target, source_ir, f"{name}_q", input_val)
    scale_float, zero_point_float = _affine_constants(
        scale, zero_point, axis, len(x.shape), reciprocal=False
    )
    if zero_point_float is not None:
        x = impl.elementwise.sub(
            ctx,
            target,
            source_ir,
            f"{name}_sub_zero_point",
            x,
            get_trt_tensor(ctx, zero_point_float, f"{name}_zero_point"),
        )
    if axis is None and out_dtype != torch.float32:
        # The per-tensor op subtracts in out_dtype and rounds there before it multiplies by the
        # float32 scale; the per-channel op computes in float32 and casts at the end.
        x = cast_trt_tensor(
            ctx, x, out_dtype, f"{name}_round_to_out_dtype", target, source_ir
        )
        x = cast_trt_tensor(
            ctx, x, trt.float32, f"{name}_back_to_float32", target, source_ir
        )
    x = impl.elementwise.mul(
        ctx,
        target,
        source_ir,
        f"{name}_mul_scale",
        x,
        get_trt_tensor(ctx, scale_float, f"{name}_scale"),
    )
    return cast_trt_tensor(ctx, x, out_dtype, f"{name}_out", target, source_ir)
