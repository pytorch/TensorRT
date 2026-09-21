"""Helpers for TorchAO NVFP4 weight-only quantization examples.

TorchAO's default NVFP4 Linear path dequantizes in Python and export would
see an opaque weight. NVFP4TensorNonDecomposed emits
``torchao_trt.dequantize_nvfp4`` so Torch-TensorRT can map packed FP4 +
swizzled FP8 block scales to TensorRT constants and IDequantizeLayer.

NVFP4 requires the last two weight dims to be divisible by 16.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch_tensorrt.dynamo.conversion.nvfp4_custom_op import dequantize_nvfp4
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    per_tensor_amax_to_scale,
)
from torchao.utils import _dispatch__torch_function__

aten = torch.ops.aten


class NVFP4TensorNonDecomposed(NVFP4Tensor):
    """NVFP4Tensor that dequantizes via explicit ``dequantize_nvfp4``.

    Parent NVFP4Tensor disables ``__torch_function__``; re-enable it so
    functional linear is intercepted before export treats the parameter as
    opaque.
    """

    __torch_function__ = classmethod(_dispatch__torch_function__)


implements = NVFP4TensorNonDecomposed.implements
implements_torch_function = NVFP4TensorNonDecomposed.implements_torch_function


@implements([aten.linear.default])
@implements_torch_function([torch.nn.functional.linear])
def _nvfp4_non_decomposed_linear(func, types, args, kwargs):
    """Run Linear as dequantize + F.linear so export sees dequantize_nvfp4."""
    input_tensor = args[0]
    weight_tensor = args[1]
    bias = args[2] if len(args) > 2 else None

    assert isinstance(weight_tensor, NVFP4TensorNonDecomposed)
    assert weight_tensor.per_tensor_scale is not None
    assert weight_tensor.is_swizzled_scales

    rows, cols = weight_tensor.shape
    weight_hp = dequantize_nvfp4(
        weight_tensor.qdata,
        weight_tensor.scale,
        weight_tensor.per_tensor_scale,
        rows,
        cols,
        input_tensor.dtype,
    )
    if bias is not None and bias.dtype != input_tensor.dtype:
        bias = bias.to(input_tensor.dtype)
    return torch.nn.functional.linear(input_tensor, weight_hp, bias)


def quantize_linear_nvfp4(
    model: torch.nn.Module,
    verbose: bool = False,
    compute_device: Optional[str] = None,
) -> torch.nn.Module:
    """Replace Linear weights with NVFP4 (packed FP4 E2M1 + FP8 block scales).

    Skips layers whose last two dims are not divisible by 16. Each layer is
    quantized on ``compute_device`` then moved back so VRAM stays bounded.
    """
    if compute_device is None:
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"

    quantized = 0
    skipped = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = module.weight
        if getattr(weight, "is_meta", False):
            raise RuntimeError(
                f"{name}.weight is on the meta device (no data to quantize). "
                "Load the model without device_map/offloading before quantizing."
            )

        if weight.shape[-2] % 16 != 0 or weight.shape[-1] % 16 != 0:
            if verbose:
                print(
                    f"[skip] {name}: shape {tuple(weight.shape)} " "not divisible by 16"
                )
            skipped += 1
            continue

        original_device = weight.device
        hp = weight.detach().to(device=compute_device, dtype=torch.bfloat16)
        tensor_amax = torch.max(torch.abs(hp))
        per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)
        qw = NVFP4Tensor.to_nvfp4(
            hp,
            per_tensor_scale=per_tensor_scale,
            is_swizzled_scales=True,
            act_quant_kwargs=None,
        )
        if qw.device != original_device:
            qw = qw.to(original_device)
        module.weight = torch.nn.Parameter(qw, requires_grad=False)
        quantized += 1
        del hp
        if verbose:
            print(
                f"[nvfp4] {name}: {tuple(weight.shape)} "
                f"-> qdata {tuple(qw.qdata.shape)} scale {tuple(qw.scale.shape)}"
            )

    if compute_device == "cuda":
        torch.cuda.empty_cache()
    if verbose:
        print(
            f"Quantized {quantized} Linear layers to NVFP4 "
            f"(skipped {skipped}; block_size=16, swizzled FP8 scales)"
        )
    return model


def convert_nvfp4_to_non_decomposed(model: torch.nn.Module) -> torch.nn.Module:
    """Promote all NVFP4Tensor parameters to NVFP4TensorNonDecomposed."""
    for param in model.parameters():
        if isinstance(param, NVFP4Tensor) and not isinstance(
            param, NVFP4TensorNonDecomposed
        ):
            param.__class__ = NVFP4TensorNonDecomposed
            param.requires_grad_(False)
    return model


def pre_process_model_for_export(model: torch.nn.Module) -> torch.nn.Module:
    """Promote NVFP4Tensor parameters so export emits dequantize_nvfp4."""
    return convert_nvfp4_to_non_decomposed(model)
