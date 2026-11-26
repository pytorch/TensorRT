"""Ops for mixture-of-experts weights stored in a block-quantized format.

A model keeps its expert weights packed and dequantizes them with one of the
``dequantize_*_experts`` ops before ``aten._grouped_mm``. Exported as is, the
program computes exactly what the dequantized model computes. With
``grouped_mm_kernel`` set, Torch-TensorRT replaces each dequantize followed by a
grouped matrix multiply with a ``grouped_mm_*`` op, whose kernel reads the packed
weights and dequantizes them as it multiplies, so the engine never holds the
16-bit weights.

Weights are ``[groups, N, K]`` (each group's output rows along N), matching
``grouped_mm(mat1, weight.transpose(-2, -1))``.

- NVFP4: two E2M1 codes per byte, the even K index in the low nibble; one
  float8_e4m3fn scale per 16 values along K; one float32 scale per group.
- MXFP8: float8_e4m3fn values; one power-of-two scale per 32 values along K,
  stored as its biased exponent (E8M0) in a uint8.
"""

from __future__ import annotations

import torch

_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _check_nvfp4(
    weight: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2: torch.Tensor
) -> None:
    groups, n, half_k = weight.shape
    if weight.dtype != torch.uint8 or weight_scale.dtype != torch.float8_e4m3fn:
        raise ValueError("NVFP4 experts need uint8 weights and float8_e4m3fn scales")
    if tuple(weight_scale.shape) != (groups, n, half_k * 2 // 16):
        raise ValueError(
            f"NVFP4 scales must be {(groups, n, half_k * 2 // 16)}, got {tuple(weight_scale.shape)}"
        )
    if tuple(weight_scale_2.shape) != (groups,):
        raise ValueError(f"the NVFP4 per-group scale must have shape {(groups,)}")


def _check_mxfp8(weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
    groups, n, k = weight.shape
    if weight.dtype != torch.float8_e4m3fn or weight_scale.dtype != torch.uint8:
        raise ValueError(
            "MXFP8 experts need float8_e4m3fn weights and uint8 (E8M0) scales"
        )
    if tuple(weight_scale.shape) != (groups, n, k // 32):
        raise ValueError(
            f"MXFP8 scales must be {(groups, n, k // 32)}, got {tuple(weight_scale.shape)}"
        )


def _nvfp4_to_float(
    weight: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2: torch.Tensor
) -> torch.Tensor:
    lut = torch.tensor(_E2M1_VALUES, device=weight.device)
    codes = weight.long()
    values = torch.stack((lut[codes & 0xF], lut[codes >> 4]), dim=-1)
    values = values.view(*weight_scale.shape, 16) * weight_scale.float()[..., None]
    values = values * weight_scale_2.float()[:, None, None, None]
    return values.reshape(weight.shape[0], weight.shape[1], -1)


def _mxfp8_to_float(weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    values = (
        weight.float().view(*weight_scale.shape, 32)
        * torch.exp2(weight_scale.float() - 127)[..., None]
    )
    return values.reshape(weight.shape)


@torch.library.custom_op("tensorrt::dequantize_nvfp4_experts", mutates_args=())  # type: ignore[misc]
def dequantize_nvfp4_experts(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    _check_nvfp4(weight, weight_scale, weight_scale_2)
    return _nvfp4_to_float(weight, weight_scale, weight_scale_2).to(output_dtype)


@dequantize_nvfp4_experts.register_fake  # type: ignore[misc]
def _(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return weight.new_empty(
        (weight.shape[0], weight.shape[1], weight.shape[2] * 2), dtype=output_dtype
    )


@torch.library.custom_op("tensorrt::dequantize_mxfp8_experts", mutates_args=())  # type: ignore[misc]
def dequantize_mxfp8_experts(
    weight: torch.Tensor, weight_scale: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor:
    _check_mxfp8(weight, weight_scale)
    return _mxfp8_to_float(weight, weight_scale).to(output_dtype)


@dequantize_mxfp8_experts.register_fake  # type: ignore[misc]
def _(
    weight: torch.Tensor, weight_scale: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor:
    return weight.new_empty(weight.shape, dtype=output_dtype)


# The fused forms. Their results equal dequantizing, then aten._grouped_mm.
@torch.library.custom_op("tensorrt::grouped_mm_nvfp4", mutates_args=())  # type: ignore[misc]
def grouped_mm_nvfp4(
    mat1: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    mat2 = dequantize_nvfp4_experts(weight, weight_scale, weight_scale_2, mat1.dtype)
    return torch._grouped_mm(mat1, mat2.transpose(-2, -1), offs=offs)


@grouped_mm_nvfp4.register_fake  # type: ignore[misc]
def _(
    mat1: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    return mat1.new_empty((mat1.shape[0], weight.shape[1]))


@torch.library.custom_op("tensorrt::grouped_mm_mxfp8", mutates_args=())  # type: ignore[misc]
def grouped_mm_mxfp8(
    mat1: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    mat2 = dequantize_mxfp8_experts(weight, weight_scale, mat1.dtype)
    return torch._grouped_mm(mat1, mat2.transpose(-2, -1), offs=offs)


@grouped_mm_mxfp8.register_fake  # type: ignore[misc]
def _(
    mat1: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    return mat1.new_empty((mat1.shape[0], weight.shape[1]))


# Each row multiplied by the expert its index names; equal to batching the selected dequantized weights.
@torch.library.custom_op("tensorrt::gather_mm_nvfp4", mutates_args=())  # type: ignore[misc]
def gather_mm_nvfp4(
    mat1: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    mat2 = dequantize_nvfp4_experts(weight, weight_scale, weight_scale_2, mat1.dtype)
    return torch.bmm(mat2[index], mat1.unsqueeze(-1)).squeeze(-1)


@gather_mm_nvfp4.register_fake  # type: ignore[misc]
def _(
    mat1: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    return mat1.new_empty((mat1.shape[0], weight.shape[1]))


@torch.library.custom_op("tensorrt::gather_mm_mxfp8", mutates_args=())  # type: ignore[misc]
def gather_mm_mxfp8(
    mat1: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    mat2 = dequantize_mxfp8_experts(weight, weight_scale, mat1.dtype)
    return torch.bmm(mat2[index], mat1.unsqueeze(-1)).squeeze(-1)


@gather_mm_mxfp8.register_fake  # type: ignore[misc]
def _(
    mat1: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    return mat1.new_empty((mat1.shape[0], weight.shape[1]))
