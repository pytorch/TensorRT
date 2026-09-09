"""Helpers for TorchAO INT4 weight-only quantization examples.

TorchAO's default Int4Tensor Linear path uses fused mslk INT4 kernels and
never emits dequantize_affine. Int4TensorNonDecomposed unpacks nibbles
and dequantizes explicitly so Torch-TensorRT can map DQ to IDequantizeLayer
and keep an INT4 weight constant.

TensorRT's IDequantizeLayer rejects nonzero zero-points, so weights must be
quantized with the symmetric Int4Tensor.from_hp path (float8-activation
branch, then restore BF16 as the activation dtype).
"""

from __future__ import annotations

from typing import Optional

import torch
from torchao.quantization import dequantize_affine
from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

aten = torch.ops.aten


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of mslk pack_int4 (low nibble = even index, sign-extended)."""
    low = torch.bitwise_right_shift(torch.bitwise_left_shift(packed, 4), 4)
    high = torch.bitwise_right_shift(packed, 4)
    return torch.stack([low, high], dim=-1).flatten(-2).contiguous()


class Int4TensorNonDecomposed(Int4Tensor):
    """Int4Tensor that dequantizes via explicit dequantize_affine.

    Always emits zero_point=None. FP32 DQ output is what Myelin currently
    accepts for INT4 blocked DQ; eager Linear casts back to the activation dtype.
    """

    def dequantize(self, output_dtype=None):
        if output_dtype is None:
            output_dtype = (
                torch.bfloat16
                if self.activation_dtype == torch.float8_e4m3fn
                else self.activation_dtype
            )

        qdata = unpack_int4(self.qdata)
        scale = self.scale.transpose(-2, -1).contiguous()
        weight_hp = dequantize_affine(
            qdata,
            tuple(self.block_size),
            scale,
            None,
            qdata.dtype,
            output_dtype=torch.float32,
        )
        return weight_hp.to(output_dtype)


implements = Int4TensorNonDecomposed.implements
implements_torch_function = Int4TensorNonDecomposed.implements_torch_function


@implements([aten.linear.default])
@implements_torch_function([torch.nn.functional.linear])
def _int4_non_decomposed_linear(func, types, args, kwargs):
    """Run Linear as dequantize + F.linear so export sees dequantize_affine.

    Parent Int4Tensor already owns Linear dispatch, but that path never
    calls dequantize(). It feeds packed INT4 qdata into fused mslk kernels
    (bf16i4bf16_rowwise). Export would then show an mslk op, which TensorRT
    cannot map to IDequantizeLayer.

    FP8 does not need this: Float8Tensor Linear already does
    weight.dequantize() then matmul. INT4's parent never takes that branch,
    so this handler steals Linear and calls our dequantize().
    """
    input_tensor = args[0]
    weight_tensor = args[1]
    bias = args[2] if len(args) > 2 else None

    assert isinstance(weight_tensor, Int4TensorNonDecomposed)

    weight_hp = weight_tensor.dequantize(output_dtype=input_tensor.dtype)
    if bias is not None and bias.dtype != input_tensor.dtype:
        bias = bias.to(dtype=input_tensor.dtype)
    return torch.nn.functional.linear(input_tensor, weight_hp, bias)


def _dequant_int4_tile_packed_to_4d(
    weight: torch.Tensor,
    compute_device: str,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Recover dense BF16 weights from Int4TilePackedTo4dTensor.

    That subclass has no real dequantize() (Tensor.dequantize dispatches
    to unimplemented aten.dequantize). Reconstruct via chunked
    F.linear using the same tinygemm path as inference.
    """
    w = weight
    if w.device.type != "cuda":
        w = w.to(compute_device)
    out_f, in_f = int(w.shape[0]), int(w.shape[1])
    device = w.device
    pieces: list[torch.Tensor] = []
    for start in range(0, in_f, chunk_size):
        end = min(start + chunk_size, in_f)
        rows = end - start
        x = torch.zeros(rows, in_f, dtype=torch.bfloat16, device=device)
        x[
            torch.arange(rows, device=device), torch.arange(start, end, device=device)
        ] = 1
        pieces.append(torch.nn.functional.linear(x, w))
    return (
        torch.cat(pieces, dim=0)
        .T.contiguous()
        .to(device=compute_device, dtype=torch.bfloat16)
    )


def _weight_to_bf16_hp(
    weight: torch.Tensor, compute_device: str
) -> tuple[torch.Tensor, torch.device]:
    """Materialize a Linear weight as dense BF16 on compute_device.

    Hub TorchAO checkpoints may use Int4TilePackedTo4dTensor, Int4Tensor,
    AffineQuantizedTensor, etc. Prefer a class-defined dequantize; fall
    back to the tile-packed linear reconstruct path. Do **not** call bare
    Tensor.dequantize — it hits unimplemented aten.dequantize.
    """
    original_device = getattr(weight, "device", torch.device(compute_device))
    cls_name = type(weight).__name__

    qdata = getattr(weight, "qdata", None)
    if cls_name == "Int4TilePackedTo4dTensor" or (
        qdata is not None and getattr(qdata, "ndim", 0) == 4
    ):
        hp = _dequant_int4_tile_packed_to_4d(weight, compute_device)
        return hp, original_device

    dq = type(weight).__dict__.get("dequantize")
    if callable(dq):
        hp = dq(weight)
        if not isinstance(hp, torch.Tensor):
            raise TypeError(f"dequantize() returned {type(hp)}, expected torch.Tensor")
        return (
            hp.detach().to(device=compute_device, dtype=torch.bfloat16),
            original_device,
        )

    hp = weight.detach().to(device=compute_device, dtype=torch.bfloat16)
    return hp, original_device


def quantize_linear_int4_symmetric(
    model: torch.nn.Module,
    group_size: int = 128,
    verbose: bool = False,
    compute_device: Optional[str] = None,
) -> torch.nn.Module:
    """Replace Linear weights with symmetric group-wise INT4 (TRT-compatible).

    TorchAO's default Int4WeightOnlyConfig path for BF16 activations is
    asymmetric (nonzero zero_point). TRT rejects that, so this uses TorchAO's
    symmetric quantizer (float8-activation branch of Int4Tensor.from_hp) and
    restores BF16 as the activation dtype for the DQ → Linear path.

    Accepts dense BF16 weights **or** already-quantized Hub TorchAO weights
    (dequantized first). Each layer is handled on compute_device then moved
    back so VRAM stays bounded.
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

        hp_weight, original_device = _weight_to_bf16_hp(weight, compute_device)
        if hp_weight.shape[-1] % group_size != 0:
            if verbose:
                print(
                    f"[skip] {name}: shape {tuple(hp_weight.shape)} not divisible "
                    f"by group_size={group_size}"
                )
            module.weight = torch.nn.Parameter(
                hp_weight.to(original_device), requires_grad=False
            )
            skipped += 1
            continue

        block_size = [1] * (hp_weight.ndim - 1) + [group_size]
        qw = Int4Tensor.from_hp(
            hp_weight,
            block_size,
            activation_dtype=torch.float8_e4m3fn,
        )
        qw.activation_dtype = torch.bfloat16
        if not torch.all(qw.zero_point == 0).item():
            raise AssertionError(f"{name}: expected symmetric INT4 (zero zero_point)")
        if qw.device != original_device:
            qw = qw.to(original_device)
        module.weight = torch.nn.Parameter(qw, requires_grad=False)
        quantized += 1
        del hp_weight
        if verbose:
            print(
                f"[int4] {name}: -> qdata {tuple(qw.qdata.shape)} "
                f"(symmetric, group_size={group_size})"
            )
    if compute_device == "cuda":
        torch.cuda.empty_cache()
    if verbose:
        print(
            f"Quantized {quantized} Linear layers to symmetric INT4 "
            f"(skipped {skipped})"
        )
    return model


def convert_hub_int4_to_symmetric_trt(
    model: torch.nn.Module,
    group_size: int = 128,
    verbose: bool = False,
) -> torch.nn.Module:
    """Hub INT4 (HQQ / tile-packed / asymmetric) → TRT-ready symmetric Int4Tensor."""
    return quantize_linear_int4_symmetric(model, group_size=group_size, verbose=verbose)


def convert_int4_to_int4_non_decomposed(model: torch.nn.Module) -> torch.nn.Module:
    """Promote all Int4Tensor parameters to Int4TensorNonDecomposed."""
    for param in model.parameters():
        if isinstance(param, Int4Tensor) and not isinstance(
            param, Int4TensorNonDecomposed
        ):
            param.__class__ = Int4TensorNonDecomposed
            param.requires_grad_(False)
    return model


def pre_process_model_for_export(model: torch.nn.Module) -> torch.nn.Module:
    """Promote Int4Tensor parameters so export emits dequantize_affine."""
    return convert_int4_to_int4_non_decomposed(model)
