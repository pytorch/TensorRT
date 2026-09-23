"""Alpamayo setattr replacements used only while compiling Edge engines."""

from __future__ import annotations

from typing import Any, Callable

import torch

from ...plugin.attn_patches import (
    _patch_language_attention,
    register_patch,
)
from ...prefix_cache import PrefixKVCache
from ..common.patches import causal_lm_plugin_forward

ALPAMAYO = "alpamayo"


@register_patch(
    ALPAMAYO,
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionAttention.forward",
)
def _patch_qwen3_vl_vision_attention(original: Callable) -> Callable:
    """Route Qwen3-VL vision attention through the Edge ViT plugin."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb=None,
        position_embeddings=None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del rotary_pos_emb, kwargs
        if position_embeddings is None:
            return original(
                self,
                hidden_states,
                cu_seqlens,
                position_embeddings=position_embeddings,
            )

        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb_vision,
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
        q = q.to(torch.float16).contiguous()
        k = k.to(torch.float16).contiguous()
        v = v.to(torch.float16).contiguous()

        # The carrier's length communicates a safe maximum sequence length.
        max_seqlen_carrier = torch.zeros(
            hidden_states.shape[0],
            device=hidden_states.device,
            dtype=torch.int32,
        )
        output = torch.ops.trt.vit_attention_plugin.default(
            q,
            k,
            v,
            cu_seqlens.to(torch.int32),
            max_seqlen_carrier,
            int(self.num_heads),
            int(q.shape[-1]),
        )
        output = output.reshape(seq_len, -1).to(hidden_states.dtype)
        return self.proj(output)

    return forward


@register_patch(
    ALPAMAYO,
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionModel.forward",
)
def _patch_qwen3_vl_vision_model(original: Callable) -> Callable:
    """Return a tensor rather than a Python list for deepstack outputs."""

    def forward(self, hidden_states, grid_thw, **kwargs: Any):
        visual, deepstack = original(
            self,
            hidden_states,
            grid_thw,
            **kwargs,
        )
        return visual, torch.stack(tuple(deepstack), dim=0)

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


@register_patch(
    ALPAMAYO,
    "alpamayo1_5.models.alpamayo1_5.Alpamayo1_5.forward",
)
def _patch_alpamayo_action_step(original: Callable) -> Callable:
    """Compile one Alpamayo diffusion velocity step with prefix KV tensors."""

    def forward(
        self,
        noisy_action,
        timestep=None,
        prefix_k=None,
        prefix_v=None,
        position_ids=None,
        attention_mask=None,
        *args,
        **kwargs: Any,
    ):
        if prefix_k is None or getattr(prefix_k, "ndim", 0) != 5:
            return original(self, noisy_action, timestep, *args, **kwargs)

        action_embeds = self.action_in_proj(noisy_action, timestep)
        expert_kwargs: dict[str, Any] = {}
        if self.config.expert_non_causal_attention:
            expert_kwargs["is_causal"] = False
        expert = self.expert(
            inputs_embeds=action_embeds,
            position_ids=position_ids,
            past_key_values=PrefixKVCache(prefix_k, prefix_v),
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            **expert_kwargs,
        )
        hidden = expert.last_hidden_state
        hidden = hidden[:, -int(noisy_action.shape[1]) :]
        return self.action_out_proj(hidden).reshape_as(noisy_action)

    return forward
