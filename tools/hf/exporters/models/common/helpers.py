from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


def causal_lm_flat(
    language: nn.Module,
    inputs_embeds: torch.Tensor,
    *,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    seq_len: int | None = None,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """inputs_embeds, rope, ctx, kv_start, last_token_ids, ds_stack, *kvs."""
    decoder = getattr(language, "model", language)
    cfg = language.config
    bsz, prompt_len, hidden = inputs_embeds.shape
    seq_len = int(seq_len or prompt_len)
    num_layers = len(decoder.layers)
    num_kv = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    try:
        from ...rope import make_rope_rotary_cos_sin

        rope = make_rope_rotary_cos_sin(
            cfg, int(max_seq_len), device, language_model=language
        )
    except ImportError:
        rope = torch.zeros(int(max_seq_len), 2, 1, head_dim, device=device, dtype=dtype)

    ctx_len = torch.full((bsz,), seq_len, device=device, dtype=torch.int32)
    last_token_ids = torch.full((bsz, 1), seq_len - 1, device=device, dtype=torch.int64)
    kv_start = torch.empty(0, dtype=torch.int32, device=device)
    ds_stack = torch.zeros(0, bsz, seq_len, hidden, device=device, dtype=dtype)
    kvs = [
        torch.zeros(
            bsz, 2, num_kv, int(max_seq_len), head_dim, device=device, dtype=dtype
        )
        for _ in range(num_layers)
    ]
    flat = (inputs_embeds, rope, ctx_len, kv_start, last_token_ids, ds_stack, *kvs)
    names = [
        "inputs_embeds",
        "rope_rotary_cos_sin",
        "context_lengths",
        "kvcache_start_index",
        "last_token_ids",
        "ds_stack",
        *[f"past_key_values_{i}" for i in range(num_layers)],
    ]
    return flat, {
        "input_names": names,
        "num_layers": num_layers,
        "hidden_size": hidden,
        "head_dim": head_dim,
        "num_key_value_heads": num_kv,
    }


def kv_kwargs(
    sample: Mapping[str, Any], prefix: str = "past_key_values_"
) -> list[torch.Tensor]:
    tensors = []
    idx = 0
    while f"{prefix}{idx}" in sample:
        tensors.append(sample[f"{prefix}{idx}"])
        idx += 1
    return tensors


def split_flat_to_kwargs(
    flat: tuple[torch.Tensor, ...], names: list[str]
) -> dict[str, torch.Tensor]:
    return dict(zip(names, flat))
