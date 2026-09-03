"""Register torchao_trt.dequantize_mxfp4 when TorchAO MX kernels exist.

TorchAO MXTensor does not emit a single DQ op that TensorRT can map.
This custom op keeps packed FP4 qdata and swizzled E8M0 block scales visible
through torch.export so the converter can emit TensorRT FP4 / E8M0 constants
and IDequantizeLayer.

MXFP4 uses block size 32 (NVFP4 uses 16) and E8M0 scales (NVFP4 uses FP8
E4M3 plus a per-tensor scale). There is no TorchAO MXFP4 weight-only config;
the public recipe is MXDynamicActivationMXWeightConfig. This op only
dequantizes weights so TensorRT can keep an FP4 constant.
"""

from __future__ import annotations

import torch

try:
    from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4
    from torchao.prototype.mx_formats.mx_tensor import get_fp_scale
    from torchao.prototype.mx_formats.utils import from_blocked

    _MXFP4_KERNELS_AVAILABLE = True
except Exception:
    _MXFP4_KERNELS_AVAILABLE = False


if _MXFP4_KERNELS_AVAILABLE:

    @torch.library.custom_op(  # type: ignore[misc]
        "torchao_trt::dequantize_mxfp4", mutates_args=()
    )
    def dequantize_mxfp4(
        qdata: torch.Tensor,
        block_scale_e8m0: torch.Tensor,
        rows: int,
        cols: int,
        block_size: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize packed FP4 + swizzled E8M0 block scales for eager validation."""
        if cols % block_size != 0:
            raise ValueError(
                f"cols ({cols}) must be divisible by block_size ({block_size})"
            )
        n_blocks = cols // block_size

        unpacked = unpack_uint4(qdata.contiguous().view(torch.uint8))
        data_f32 = f4_unpacked_to_f32(unpacked).view(rows, n_blocks, block_size)

        scale = from_blocked(block_scale_e8m0, rows, n_blocks)
        scale_f32 = get_fp_scale(scale).to(torch.float32)
        return (
            (data_f32 * scale_f32.view(rows, n_blocks, 1))
            .view(rows, cols)
            .to(output_dtype)
        )

    @dequantize_mxfp4.register_fake  # type: ignore[misc]
    def _dequantize_mxfp4_fake(
        qdata: torch.Tensor,
        block_scale_e8m0: torch.Tensor,
        rows: int,
        cols: int,
        block_size: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.empty((rows, cols), device=qdata.device, dtype=output_dtype)

else:

    def dequantize_mxfp4(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError(
            "torchao_trt.dequantize_mxfp4 requires torchao MX kernels "
            "(torchao.prototype.mx_formats)"
        )
