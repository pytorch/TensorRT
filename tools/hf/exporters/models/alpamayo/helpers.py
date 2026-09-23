from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...prefix_cache import PrefixKVCache


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


class VisualFixedGrid(nn.Module):
    """Qwen3-VL visual tower with grid-dependent tensors baked as buffers."""

    def __init__(self, visual: nn.Module, grid_thw: torch.Tensor):
        super().__init__()
        self.visual = visual.eval()
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
            for block in visual.blocks:
                block.attn._static_lengths = static_lengths

        self.register_buffer("pos_embeds", pos_embeds, persistent=False)
        self.register_buffer("cos", rotary.cos(), persistent=False)
        self.register_buffer("sin", rotary.sin(), persistent=False)
        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.visual.patch_embed(pixel_values)
        hidden = hidden + self.pos_embeds.to(hidden.dtype)
        position_embeddings = (
            self.cos.to(hidden.dtype),
            self.sin.to(hidden.dtype),
        )
        deepstack = []
        for layer_index, block in enumerate(self.visual.blocks):
            hidden = block(
                hidden,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_index in self.visual.deepstack_visual_indexes:
                merger_index = self.visual.deepstack_visual_indexes.index(layer_index)
                deepstack.append(
                    self.visual.deepstack_merger_list[merger_index](hidden)
                )
        return self.visual.merger(hidden), torch.stack(tuple(deepstack), dim=0)


class StaticKVDiffusionStepModule(nn.Module):
    """Fused Alpamayo action projection, expert, and output projection."""

    def __init__(
        self,
        action_in_proj: nn.Module,
        expert: nn.Module,
        action_out_proj: nn.Module,
        action_space_dims: tuple[int, ...],
    ):
        super().__init__()
        self.action_in_proj = action_in_proj
        self.expert = expert
        self.action_out_proj = action_out_proj
        self.action_space_dims = action_space_dims
        self.n_diffusion_tokens = int(action_space_dims[0])

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = noisy_action.shape[0]
        action_embeds = self.action_in_proj(noisy_action, timestep)
        if action_embeds.dim() == 2:
            action_embeds = action_embeds.view(
                batch_size,
                self.n_diffusion_tokens,
                -1,
            )
        expert = self.expert(
            inputs_embeds=action_embeds,
            position_ids=position_ids,
            past_key_values=PrefixKVCache(prefix_k, prefix_v),
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = expert.last_hidden_state[:, -self.n_diffusion_tokens :]
        return self.action_out_proj(hidden).reshape(
            batch_size,
            *self.action_space_dims,
        )
