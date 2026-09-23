from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def alpamayo_vlm(model: nn.Module) -> nn.Module:
    """Return the Qwen3-VL conditional-generation module."""
    vlm = getattr(model, "vlm", None)
    if not isinstance(vlm, nn.Module):
        raise AttributeError(f"{type(model).__name__} has no Alpamayo VLM")
    return vlm


def alpamayo_vlm_core(model: nn.Module) -> nn.Module:
    """Return the Qwen3-VL model containing visual and language towers."""
    core = getattr(alpamayo_vlm(model), "model", None)
    if not isinstance(core, nn.Module):
        raise AttributeError("Alpamayo VLM has no model")
    return core


def alpamayo_visual(model: nn.Module) -> nn.Module:
    visual = getattr(alpamayo_vlm_core(model), "visual", None)
    if not isinstance(visual, nn.Module):
        raise AttributeError("Alpamayo VLM has no visual tower")
    return visual


def alpamayo_language(model: nn.Module) -> nn.Module:
    language = getattr(alpamayo_vlm_core(model), "language_model", None)
    if not isinstance(language, nn.Module):
        language = getattr(alpamayo_vlm(model), "language_model", None)
    if not isinstance(language, nn.Module):
        raise AttributeError("Alpamayo VLM has no language model")
    return language


def stack_deepstack_features(features: Any) -> torch.Tensor:
    """Normalize Qwen3-VL deepstack features to ``[N, tokens, hidden]``."""
    if isinstance(features, torch.Tensor):
        return features
    if not isinstance(features, (tuple, list)) or not features:
        raise ValueError("Alpamayo visual tower returned no deepstack features")
    return torch.stack(tuple(features), dim=0)


def scatter_visual_tokens(
    visual: torch.Tensor,
    text_embeds: torch.Tensor,
    image_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Insert flattened visual features into Qwen image-token positions."""
    hidden = int(text_embeds.shape[-1])
    flat = text_embeds.reshape(-1, hidden).clone()
    mask = image_token_mask.reshape(-1)
    values = visual.reshape(-1, hidden).to(device=flat.device, dtype=flat.dtype)
    count = int(mask.sum().item())
    if count != int(values.shape[0]):
        raise ValueError(
            "Alpamayo image token count does not match visual features: "
            f"{count} tokens vs {values.shape[0]} features"
        )
    flat[mask] = values
    return flat.reshape_as(text_embeds)


def make_deepstack_tensor(
    features: torch.Tensor,
    image_token_mask: torch.Tensor,
    *,
    num_layers: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand sparse Qwen deepstack features into the common dense layout."""
    dense = torch.zeros(
        num_layers,
        batch_size,
        seq_len,
        hidden_size,
        device=device,
        dtype=dtype,
    )
    zero_text = torch.zeros(
        batch_size,
        seq_len,
        hidden_size,
        device=device,
        dtype=dtype,
    )
    if int(features.shape[0]) > num_layers:
        raise ValueError(
            f"{features.shape[0]} deepstack stages exceed {num_layers} language layers"
        )
    for layer_index in range(int(features.shape[0])):
        dense[layer_index] = scatter_visual_tokens(
            features[layer_index],
            zero_text,
            image_token_mask,
        )
    return dense
