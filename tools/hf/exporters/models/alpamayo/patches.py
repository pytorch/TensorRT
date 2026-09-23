"""Alpamayo setattr replacements used only while compiling Edge engines."""

from __future__ import annotations

from typing import Any, Callable

import torch

from ...plugin.attn_patches import (
    _patch_language_attention,
    register_patch,
)
from ..common.patches import causal_lm_plugin_forward

ALPAMAYO = "alpamayo"


@register_patch(
    ALPAMAYO,
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionAttention.forward",
)
def _patch_qwen3_vl_vision_attention(original: Callable) -> Callable:
    """Use static sequence splits baked by ``VisualFixedGrid``."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb=None,
        position_embeddings=None,
        **kwargs: Any,
    ) -> torch.Tensor:
        static_lengths = getattr(self, "_static_lengths", None)
        if static_lengths is None or position_embeddings is None:
            return original(
                self,
                hidden_states,
                cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb_vision,
            eager_attention_forward,
        )

        seq_len = int(hidden_states.shape[0])
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_len, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        splits = [torch.split(tensor, static_lengths, dim=2) for tensor in (q, k, v)]
        outputs = [
            eager_attention_forward(
                self,
                q_part,
                k_part,
                v_part,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0,
                is_causal=False,
                **kwargs,
            )[0]
            for q_part, k_part, v_part in zip(*splits)
        ]
        output = (
            torch.cat(outputs, dim=1)
            .reshape(seq_len, -1)
            .contiguous()
            .to(hidden_states.dtype)
        )
        return self.proj(output)

    return forward


register_patch(
    ALPAMAYO,
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward",
)(_patch_language_attention)


@register_patch(
    ALPAMAYO,
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLForConditionalGeneration.forward",
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextModel.forward",
)
def _patch_qwen3_vl_language(original: Callable) -> Callable:
    """Use flattened AttentionPlugin I/O for language prefill."""

    def forward(
        self,
        inputs_embeds=None,
        rope_rotary_cos_sin=None,
        context_lengths=None,
        kvcache_start_index=None,
        last_token_ids=None,
        ds_stack=None,
        *past_key_values,
        **kwargs: Any,
    ):
        if rope_rotary_cos_sin is None:
            return original(self, inputs_embeds=inputs_embeds, **kwargs)

        root = getattr(self, "model", None)
        decoder = getattr(root, "language_model", None)
        if decoder is None:
            decoder = self
        return causal_lm_plugin_forward(
            decoder,
            inputs_embeds,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            last_token_ids,
            ds_stack,
            *past_key_values,
            lm_head=getattr(self, "lm_head", None),
        )

    return forward
