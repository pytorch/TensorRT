from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@torch.no_grad()
def ensure_valid_rope_inv_freq(model: nn.Module) -> None:
    """Repair deterministic RoPE buffers broken by remote-code version drift."""
    decoder = model if hasattr(model, "layers") else model.model
    repaired = 0

    for layer in decoder.layers:
        attention = layer.self_attn
        rotary = attention.rotary_emb
        device = attention.q_proj.weight.device
        dim = int(rotary.dim)
        base = float(rotary.base)
        exponent = torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
        expected = torch.pow(
            torch.tensor(base, device=device, dtype=torch.float32),
            -exponent,
        )

        current = getattr(rotary, "inv_freq", None)
        valid = (
            isinstance(current, torch.Tensor)
            and current.device.type != "meta"
            and current.shape == expected.shape
            and bool(torch.isfinite(current).all())
            and torch.allclose(
                current.to(device=device, dtype=torch.float32),
                expected,
            )
        )
        if valid:
            continue

        if current is None:
            rotary.register_buffer("inv_freq", expected, persistent=False)
        else:
            rotary.inv_freq = expected
        repaired += 1

    if repaired:
        logger.warning(
            "Reinitialized invalid Nanbeige RoPE buffers in %d attention layers",
            repaired,
        )
