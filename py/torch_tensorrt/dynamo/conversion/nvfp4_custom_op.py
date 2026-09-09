"""Register ``torchao_trt.dequantize_nvfp4`` when TorchAO NVFP4 kernels exist.

TorchAO ``NVFP4Tensor`` does not emit a single DQ op that TensorRT can map.
This custom op keeps packed FP4 qdata, swizzled FP8 block scales, and the
global scale visible through ``torch.export`` so the converter can emit
TensorRT FP4 / FP8 constants and two-level ``IDequantizeLayer``.
"""

from __future__ import annotations

import torch

try:
    from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4
    from torchao.prototype.mx_formats.utils import from_blocked

    _NVFP4_KERNELS_AVAILABLE = True
except Exception:
    _NVFP4_KERNELS_AVAILABLE = False


if _NVFP4_KERNELS_AVAILABLE:

    @torch.library.custom_op(  # type: ignore[misc]
        "torchao_trt::dequantize_nvfp4", mutates_args=()
    )
    def dequantize_nvfp4(
        qdata: torch.Tensor,
        block_scale: torch.Tensor,
        per_tensor_scale: torch.Tensor,
        rows: int,
        cols: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize packed FP4 + swizzled FP8 scales for eager validation."""
        unpacked = unpack_uint4(qdata.contiguous().view(torch.uint8))
        data_f32 = f4_unpacked_to_f32(unpacked)
        data_f32 = data_f32.view(rows, cols // 16, 16)

        scale = from_blocked(block_scale, rows, cols // 16)
        scale = scale.to(torch.float32) * per_tensor_scale.to(torch.float32)
        return (
            (data_f32 * scale.view(rows, cols // 16, 1))
            .view(rows, cols)
            .to(output_dtype)
        )

    @dequantize_nvfp4.register_fake  # type: ignore[misc]
    def _dequantize_nvfp4_fake(
        qdata: torch.Tensor,
        block_scale: torch.Tensor,
        per_tensor_scale: torch.Tensor,
        rows: int,
        cols: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.empty((rows, cols), device=qdata.device, dtype=output_dtype)

else:

    def dequantize_nvfp4(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError(
            "torchao_trt.dequantize_nvfp4 requires torchao NVFP4 kernels "
            "(torchao.prototype.mx_formats)"
        )
