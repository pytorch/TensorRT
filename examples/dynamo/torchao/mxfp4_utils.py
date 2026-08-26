"""Helpers for TorchAO MXFP4 (MXDynamicActivationMXWeightConfig) examples.

TorchAO's default MXTensor Linear path dequantizes in Python and export would
see an opaque weight. MXTensorNonDecomposed emits
torchao_trt.dequantize_mxfp4 so Torch-TensorRT can map packed FP4 +
swizzled E8M0 block scales to TensorRT constants and IDequantizeLayer.

There is no TorchAO MXFP4 weight-only config. Last two weight dims must be
divisible by 32. Native MXFP4xMXFP4 kernels need B200/B300; this path uses
emulated storage and a weight DQ prologue into a BF16 GEMM.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch_tensorrt.dynamo.conversion.mxfp4_custom_op import dequantize_mxfp4
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import MXTensor, QuantizeTensorToMXKwargs
from torchao.quantization.quantize_.common import KernelPreference
from torchao.utils import _dispatch__torch_function__

aten = torch.ops.aten

MX_BLOCK_SIZE = 32


class MXTensorNonDecomposed(MXTensor):
    """MXTensor that dequantizes via explicit dequantize_mxfp4.

    Parent MXTensor disables __torch_function__; re-enable it so
    functional linear is intercepted before export treats the parameter as
    opaque. Dynamic activation MX quant from the config is not lowered —
    activations stay high precision so TensorRT can keep the FP4 weight.
    """

    __torch_function__ = classmethod(_dispatch__torch_function__)


implements = MXTensorNonDecomposed.implements
implements_torch_function = MXTensorNonDecomposed.implements_torch_function


@implements([aten.linear.default])
@implements_torch_function([torch.nn.functional.linear])
def _mxfp4_non_decomposed_linear(func, types, args, kwargs):
    """Run Linear as dequantize + F.linear so export sees dequantize_mxfp4."""
    input_tensor = args[0]
    weight_tensor = args[1]
    bias = args[2] if len(args) > 2 else None

    assert isinstance(weight_tensor, MXTensorNonDecomposed)
    assert weight_tensor.elem_dtype == torch.float4_e2m1fn_x2
    assert weight_tensor.is_swizzled_scales

    rows, cols = weight_tensor.shape
    weight_hp = dequantize_mxfp4(
        weight_tensor.qdata,
        weight_tensor.scale,
        rows,
        cols,
        int(weight_tensor.block_size),
        input_tensor.dtype,
    )
    if bias is not None and bias.dtype != input_tensor.dtype:
        bias = bias.to(input_tensor.dtype)
    return torch.nn.functional.linear(input_tensor, weight_hp, bias)


def quantize_linear_mxfp4(
    model: torch.nn.Module,
    verbose: bool = False,
    compute_device: Optional[str] = None,
    block_size: int = MX_BLOCK_SIZE,
) -> torch.nn.Module:
    """Replace Linear weights with MXFP4 (packed FP4 E2M1 + E8M0 block scales).

    Uses KernelPreference.EMULATED (AUTO needs B200/B300). Skips layers
    whose last two dims are not divisible by block_size (default 32).
    Each layer is quantized on compute_device then moved back.
    """
    if compute_device is None:
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"

    act_quant_kwargs = QuantizeTensorToMXKwargs(
        elem_dtype=torch.float4_e2m1fn_x2,
        block_size=block_size,
        kernel_preference=KernelPreference.EMULATED,
        is_swizzled_scales=True,
        scaling_mode=ScaleCalculationMode.RCEIL,
    )

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

        if weight.shape[-2] % block_size != 0 or weight.shape[-1] % block_size != 0:
            if verbose:
                print(
                    f"[skip] {name}: shape {tuple(weight.shape)} "
                    f"not divisible by {block_size}"
                )
            skipped += 1
            continue

        original_device = weight.device
        hp = weight.detach().to(device=compute_device, dtype=torch.bfloat16)
        qw = MXTensor.to_mx(
            hp,
            torch.float4_e2m1fn_x2,
            block_size=block_size,
            kernel_preference=KernelPreference.EMULATED,
            act_quant_kwargs=act_quant_kwargs,
            is_swizzled_scales=True,
            scaling_mode=ScaleCalculationMode.RCEIL,
        )
        if qw.device != original_device:
            qw = qw.to(original_device)
        module.weight = torch.nn.Parameter(qw, requires_grad=False)
        quantized += 1
        del hp
        if verbose:
            print(
                f"[mxfp4] {name}: {tuple(weight.shape)} "
                f"-> qdata {tuple(qw.qdata.shape)} scale {tuple(qw.scale.shape)}"
            )

    if compute_device == "cuda":
        torch.cuda.empty_cache()
    if verbose:
        print(
            f"Quantized {quantized} Linear layers to MXFP4 "
            f"(skipped {skipped}; block_size={block_size}, E8M0 scales, EMULATED)"
        )
    return model


def convert_mxfp4_to_non_decomposed(model: torch.nn.Module) -> torch.nn.Module:
    """Promote all MXTensor parameters to MXTensorNonDecomposed."""
    for param in model.parameters():
        if isinstance(param, MXTensor) and not isinstance(param, MXTensorNonDecomposed):
            param.__class__ = MXTensorNonDecomposed
            param.requires_grad_(False)
    return model


def pre_process_model_for_export(model: torch.nn.Module) -> torch.nn.Module:
    """Promote MXTensor parameters so export emits dequantize_mxfp4."""
    return convert_mxfp4_to_non_decomposed(model)
