"""Helpers for TorchAO FP8 weight-only quantization examples.

TorchAO's default ``Float8Tensor.dequantize`` decomposes into primitive ops.
``Float8TensorNonDecomposed`` keeps an explicit ``dequantize_affine`` in the
exported graph so Torch-TensorRT can map it to ``IDequantizeLayer``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torchao.quantization import dequantize_affine
from torchao.quantization.quantize_.workflows import Float8Tensor


class Float8TensorNonDecomposed(Float8Tensor):
    """``Float8Tensor`` that dequantizes via explicit ``dequantize_affine``."""

    def dequantize(self, output_dtype=None):
        if output_dtype is None:
            output_dtype = torch.bfloat16
        return dequantize_affine(
            self.qdata,
            self.block_size,
            self.scale,
            None,
            self.qdata.dtype,
            output_dtype=output_dtype,
        )


def pre_process_model_for_export(model: torch.nn.Module) -> torch.nn.Module:
    """Promote ``Float8Tensor`` parameters so export emits ``dequantize_affine``."""
    for param in model.parameters():
        if isinstance(param, Float8Tensor) and not isinstance(
            param, Float8TensorNonDecomposed
        ):
            param.__class__ = Float8TensorNonDecomposed
            param.requires_grad_(False)
    return model


@contextmanager
def exclude_dq_from_constant_folding() -> Iterator[None]:
    """Keep ``dequantize_affine`` out of inductor constant folding during export."""
    from torch._inductor.constant_folding import (
        _dont_constant_fold,
        add_dont_constant_fold,
    )

    op = torch.ops.torchao.dequantize_affine.default
    add_dont_constant_fold(op)
    try:
        yield
    finally:
        if op in _dont_constant_fold:
            _dont_constant_fold.remove(op)
