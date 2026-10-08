"""Qwen3-VL vision tower for GR00T N1.7 with grid tensors as engine inputs."""

from __future__ import annotations

import torch
import torch.nn as nn


def vision_rope(
    rotary: nn.Module, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``Qwen3VLVisionRotaryEmbedding`` without its autocast / no_grad wrappers."""
    freqs = position_ids[..., None].float() * rotary.inv_freq.float()
    freqs = torch.cat([freqs[:, 0], freqs[:, 1]], dim=-1)
    freqs = torch.cat([freqs, freqs], dim=-1)
    scale = float(rotary.attention_scaling)
    return freqs.cos() * scale, freqs.sin() * scale


def _vit_plugin_attention(
    attn: nn.Module,
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen_carrier: torch.Tensor,
) -> torch.Tensor:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        apply_rotary_pos_emb_vision,
    )

    tokens = hidden.shape[0]
    num_heads = int(attn.num_heads)
    head_dim = int(attn.head_dim)
    q, k, v = (
        attn.qkv(hidden)
        .reshape(tokens, 3, num_heads, head_dim)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    out = torch.ops.trt.vit_attention_plugin.default(
        q.to(torch.float16).contiguous(),
        k.to(torch.float16).contiguous(),
        v.to(torch.float16).contiguous(),
        cu_seqlens,
        max_seqlen_carrier,
        num_heads,
        head_dim,
    )
    out = out.reshape(tokens, num_heads * head_dim).to(attn.proj.weight.dtype)
    return attn.proj(out)


class GrootQwen3Vision(nn.Module):
    """``Qwen3VLVisionModel.forward`` with precomputed ``vision_grid_inputs``.

    Returns merged image tokens ``[N, H]`` and stacked deepstack features
    ``[num_deepstack, N, H]`` for the language ``ds_stack``.
    """

    def __init__(self, visual: nn.Module) -> None:
        super().__init__()
        self.visual = visual

    def forward(
        self,
        pixel_values: torch.Tensor,
        interp_indices: torch.Tensor,
        interp_weights: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.visual
        hidden = visual.patch_embed(pixel_values)
        pos_embeds = (
            visual.pos_embed(interp_indices) * interp_weights[:, :, None]
        ).sum(1)
        hidden = hidden + pos_embeds.to(hidden.dtype)
        cos, sin = vision_rope(visual.rotary_pos_emb, position_ids)

        deepstack = []
        for layer_num, block in enumerate(visual.blocks):
            hidden = hidden + _vit_plugin_attention(
                block.attn,
                block.norm1(hidden),
                cos,
                sin,
                cu_seqlens,
                max_seqlen_carrier,
            )
            hidden = hidden + block.mlp(block.norm2(hidden))
            if layer_num in visual.deepstack_visual_indexes:
                merger = visual.deepstack_merger_list[
                    visual.deepstack_visual_indexes.index(layer_num)
                ]
                deepstack.append(merger(hidden))
        return visual.merger(hidden), torch.stack(deepstack, dim=0)
