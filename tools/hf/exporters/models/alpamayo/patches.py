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
    """Use static sequence splits prepared on the original vision instance."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb=None,
        position_embeddings=None,
        **kwargs: Any,
    ) -> torch.Tensor:
        static_lengths = getattr(self, "_edge_static_lengths", None)
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


@register_patch(
    ALPAMAYO,
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionModel.forward",
)
def _patch_qwen3_vl_vision_model(original: Callable) -> Callable:
    """Run Qwen3-VL vision with grid-dependent values prepared on the instance."""

    def forward(self, hidden_states, grid_thw=None, **kwargs: Any):
        if not hasattr(self, "_edge_pos_embeds"):
            return original(self, hidden_states, grid_thw, **kwargs)

        del grid_thw
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + self._edge_pos_embeds.to(hidden_states.dtype)
        position_embeddings = (
            self._edge_cos.to(hidden_states.dtype),
            self._edge_sin.to(hidden_states.dtype),
        )
        deepstack = []
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=self._edge_cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_index in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_index)
                deepstack.append(
                    self.deepstack_merger_list[merger_index](hidden_states)
                )
        return self.merger(hidden_states), torch.stack(tuple(deepstack), dim=0)

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
    """Run one diffusion step when explicit stacked prefix KV is supplied."""

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

        n_diffusion_tokens = int(noisy_action.shape[1])
        action_embeds = self.action_in_proj(noisy_action, timestep).to(
            dtype=noisy_action.dtype
        )
        if action_embeds.dim() == 2:
            action_embeds = action_embeds.view(
                noisy_action.shape[0],
                n_diffusion_tokens,
                -1,
            )
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
        hidden = expert.last_hidden_state[:, -n_diffusion_tokens:]
        return self.action_out_proj(hidden).reshape_as(noisy_action)

    return forward
