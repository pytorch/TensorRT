from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _decoder(model: nn.Module) -> nn.Module:
    return getattr(model, "backbone", None) or model.model


def _kind(mixer: nn.Module) -> str:
    name = type(mixer).__name__
    if "Mamba" in name:
        return "mamba"
    if "Attention" in name:
        return "attention"
    if "MoE" in name or "Moe" in name:
        return "moe"
    return "mlp"


def allocate_plugin_states(
    model: nn.Module,
    config: Any,
    batch: int,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    kinds = [_kind(block.mixer) for block in _decoder(model).layers]
    head_dim = int(
        getattr(config, "head_dim", 0)
        or config.hidden_size // config.num_attention_heads
    )
    conv_dim = int(config.mamba_num_heads) * int(config.mamba_head_dim) + 2 * int(
        config.n_groups
    ) * int(config.ssm_state_size)
    conv_kernel = int(getattr(config, "conv_kernel", 4))
    kvs, convs, ssms = [], [], []
    for kind in kinds:
        if kind == "attention":
            kvs.append(
                torch.zeros(
                    batch,
                    2,
                    int(config.num_key_value_heads),
                    max_seq_len,
                    head_dim,
                    device=device,
                    dtype=dtype,
                )
            )
        elif kind == "mamba":
            convs.append(
                torch.zeros(batch, conv_dim, conv_kernel, device=device, dtype=dtype)
            )
            ssms.append(
                torch.zeros(
                    batch,
                    int(config.mamba_num_heads),
                    int(config.mamba_head_dim),
                    int(config.ssm_state_size),
                    device=device,
                    dtype=dtype,
                )
            )
    return kvs, convs, ssms
