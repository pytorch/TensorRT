from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def unpack_visual_output(output: Any) -> tuple[torch.Tensor, Any]:
    """Return visual embeddings and DeepStack features across HF versions."""
    pooler_output = getattr(output, "pooler_output", None)
    deepstack_features = getattr(output, "deepstack_features", None)
    if pooler_output is not None and deepstack_features is not None:
        return pooler_output, deepstack_features
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        return output[0], output[1]
    raise TypeError(
        "Unsupported Qwen3-VL vision output; expected a tuple or "
        "BaseModelOutputWithDeepstackFeatures"
    )


def prepare_edge_visual_inputs(
    visual: nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Build Qwen3-VL inputs matching Edge's Qwen3VLViTRunner bindings."""
    from transformers.vision_utils import (
        get_vision_bilinear_indices_and_weights,
        get_vision_cu_seqlens,
        get_vision_position_ids,
    )

    fast_indices, fast_weights = get_vision_bilinear_indices_and_weights(
        grid_thw,
        num_grid_per_side=visual.num_grid_per_side,
        spatial_merge_size=visual.spatial_merge_size,
    )
    position_ids = get_vision_position_ids(
        grid_thw,
        visual.spatial_merge_size,
    )
    rotary_pos_emb = visual.rotary_pos_emb(position_ids).reshape(
        pixel_values.shape[0],
        -1,
    )
    cu_seqlens = get_vision_cu_seqlens(grid_thw)
    sequence_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = int(sequence_lengths.max().item())
    max_seqlen_carrier = torch.zeros(
        max_seqlen,
        device=pixel_values.device,
        dtype=torch.int32,
    )
    return (
        pixel_values,
        rotary_pos_emb.to(device=pixel_values.device, dtype=torch.float32),
        cu_seqlens.to(device=pixel_values.device, dtype=torch.int32),
        fast_indices.to(device=pixel_values.device, dtype=torch.int64),
        fast_weights.to(device=pixel_values.device, dtype=pixel_values.dtype),
        max_seqlen_carrier,
    )


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


def prepare_fixed_grid_vision(
    visual: nn.Module,
    grid_thw: torch.Tensor,
) -> None:
    """Attach fixed-grid tensors consumed by the temporary vision patch."""
    with torch.no_grad():
        pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
        rotary = visual.rot_pos_emb(grid_thw)
        seq_len = int(pos_embeds.shape[0])
        rotary = rotary.reshape(seq_len, -1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        lengths = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0],
        )
        cu_seqlens = F.pad(
            lengths.cumsum(dim=0, dtype=torch.int32),
            (1, 0),
            value=0,
        )
        static_lengths = [int(value) for value in lengths.cpu().tolist()]

    fixed_buffers = {
        "_edge_pos_embeds": pos_embeds,
        "_edge_cos": rotary.cos(),
        "_edge_sin": rotary.sin(),
        "_edge_cu_seqlens": cu_seqlens,
    }
    for name, value in fixed_buffers.items():
        if name in visual._buffers:
            visual._buffers[name] = value
        else:
            visual.register_buffer(name, value, persistent=False)
    for block in visual.blocks:
        block.attn._edge_static_lengths = static_lengths
