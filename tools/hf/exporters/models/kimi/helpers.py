from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def language_model(model: nn.Module) -> nn.Module:
    language = getattr(model, "language_model", None)
    if not isinstance(language, nn.Module):
        raise AttributeError(f"{type(model).__name__} has no language_model")
    return language


def decoder_model(model: nn.Module) -> nn.Module:
    decoder = getattr(language_model(model), "model", None)
    if not isinstance(decoder, nn.Module) or not hasattr(decoder, "layers"):
        raise AttributeError(f"{type(model).__name__} has no Kimi decoder layers")
    return decoder


def kda_layer_indices(model: nn.Module) -> list[int]:
    return [
        index
        for index, layer in enumerate(decoder_model(model).layers)
        if bool(getattr(layer, "is_linear_attn", False))
    ]


def allocate_kda_states(
    model: nn.Module,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], list[str]]:
    """Allocate Q/K/V convolution state plus V-first recurrent state per KDA layer."""
    states: list[torch.Tensor] = []
    names: list[str] = []
    decoder = decoder_model(model)

    for layer_index in kda_layer_indices(model):
        attention = decoder.layers[layer_index].self_attn
        num_heads = int(attention.num_heads)
        head_dim = int(attention.head_dim)
        projection_size = num_heads * head_dim
        conv_size = int(attention.conv_size)

        for suffix in ("q", "k", "v"):
            states.append(
                torch.zeros(
                    batch_size,
                    projection_size,
                    conv_size,
                    device=device,
                    dtype=dtype,
                )
            )
            names.append(f"kda_conv_{suffix}_{layer_index}")

        states.append(
            torch.zeros(
                batch_size,
                num_heads,
                head_dim,
                head_dim,
                device=device,
                dtype=torch.float32,
            )
        )
        names.append(f"kda_recurrent_{layer_index}")

    return tuple(states), names


def text_config(model: nn.Module) -> Any:
    return language_model(model).config
