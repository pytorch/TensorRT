"""Alpamayo setattr replacements used only while compiling Edge engines."""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn.functional as F

from ...plugin.attn_patches import (
    _patch_language_attention,
    register_patch,
)
from ...prefix_cache import PrefixKVCache
from ..common.patches import causal_lm_plugin_forward
from . import action_ops as _action_ops  # noqa: F401

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
        max_seqlen_carrier = kwargs.pop("max_seqlen_carrier", None)
        static_lengths = getattr(self, "_edge_static_lengths", None)
        if (
            max_seqlen_carrier is not None
            and position_embeddings is not None
            and static_lengths is None
        ):
            seq_len = int(hidden_states.shape[0])
            q, k, v = (
                self.qkv(hidden_states)
                .reshape(seq_len, 3, self.num_heads, -1)
                .permute(1, 0, 2, 3)
                .unbind(0)
            )
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                apply_rotary_pos_emb_vision,
            )

            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
            output = torch.ops.trt.vit_attention_plugin.default(
                q.to(torch.float16),
                k.to(torch.float16),
                v.to(torch.float16),
                cu_seqlens,
                max_seqlen_carrier,
                int(self.num_heads),
                int(q.shape[-1]),
            )
            return self.proj(output.reshape(seq_len, -1).to(hidden_states.dtype))

        if static_lengths is None or position_embeddings is None:
            return original(
                self,
                hidden_states,
                cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        if max_seqlen_carrier is not None:
            dependency = (
                cu_seqlens[0].to(hidden_states.dtype)
                + max_seqlen_carrier[0].to(hidden_states.dtype)
            ) * 0
            hidden_states = hidden_states + dependency

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

    def forward(
        self,
        hidden_states,
        grid_thw=None,
        cu_seqlens=None,
        fast_pos_embed_idx=None,
        fast_pos_embed_weight=None,
        max_seqlen_carrier=None,
        **kwargs: Any,
    ):
        if fast_pos_embed_idx is not None:
            rotary_pos_emb = grid_thw
            hidden_states = self.patch_embed(hidden_states)
            pos_embeds = (
                self.pos_embed(fast_pos_embed_idx) * fast_pos_embed_weight[:, :, None]
            ).sum(0)
            hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
            rotary = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (rotary.cos(), rotary.sin())
            deepstack = []
            for layer_index, block in enumerate(self.blocks):
                hidden_states = block(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    max_seqlen_carrier=max_seqlen_carrier,
                    **kwargs,
                )
                if layer_index in self.deepstack_visual_indexes:
                    merger_index = self.deepstack_visual_indexes.index(layer_index)
                    deepstack.append(
                        self.deepstack_merger_list[merger_index](hidden_states)
                    )
            return (self.merger(hidden_states), *deepstack)

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
        context_mask_selector=None,
        last_token_ids=None,
        deepstack_embeds_0=None,
        deepstack_embeds_1=None,
        deepstack_embeds_2=None,
        kv_page_table=None,
        *past_key_values,
        **kwargs: Any,
    ):
        if rope_rotary_cos_sin is None:
            return original(self, inputs_embeds=inputs_embeds, **kwargs)

        root = getattr(self, "model", None)
        decoder = getattr(root, "language_model", None)
        if decoder is None:
            decoder = self
        if kv_page_table is not None:
            deepstack_embeds = tuple(
                tensor
                for tensor in (
                    deepstack_embeds_0,
                    deepstack_embeds_1,
                    deepstack_embeds_2,
                )
                if tensor is not None
            )
            hidden = inputs_embeds.to(dtype=next(decoder.parameters()).dtype)
            present_key_values = []
            for layer_index, layer in enumerate(decoder.layers):
                residual = hidden
                hidden = layer.input_layernorm(hidden)
                hidden, present = layer.self_attn(
                    hidden_states=hidden,
                    rope_rotary_cos_sin=rope_rotary_cos_sin,
                    past_key_value=past_key_values[layer_index],
                    ctx_len=context_lengths,
                    kvcache_start_index=kvcache_start_index,
                    context_mask_selector=context_mask_selector,
                    kv_page_table=kv_page_table,
                )
                hidden = residual + hidden

                residual = hidden
                hidden = layer.post_attention_layernorm(hidden)
                hidden = residual + layer.mlp(hidden)
                present_key_values.append(present)
                if layer_index < len(deepstack_embeds):
                    hidden = hidden + deepstack_embeds[layer_index].to(hidden.dtype)

            hidden = decoder.norm(hidden)
            token_indices = (
                last_token_ids
                if last_token_ids.ndim == 1
                else last_token_ids.squeeze(-1)
            )
            last_hidden = hidden[
                torch.arange(
                    hidden.shape[0],
                    device=hidden.device,
                    dtype=torch.long,
                ),
                token_indices,
            ]
            logits = getattr(self, "lm_head")(last_hidden).float()
            return logits, *present_key_values

        return causal_lm_plugin_forward(
            decoder,
            inputs_embeds,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            context_mask_selector,
            last_token_ids,
            deepstack_embeds_0,
            *past_key_values,
            lm_head=getattr(self, "lm_head", None),
        )

    return forward


@register_patch(
    ALPAMAYO,
    "alpamayo1_5.models.alpamayo1_5.Alpamayo1_5.forward",
)
def _patch_alpamayo_action_step(original: Callable) -> Callable:
    """Dispatch Alpamayo action calls to unified or Edge runtime ABIs."""

    def forward(
        self,
        noisy_action,
        timestep=None,
        *args,
        **kwargs: Any,
    ):
        if len(args) < 4:
            return original(self, noisy_action, timestep, *args, **kwargs)

        # Existing unified-export ABI:
        # noisy_action, timestep, prefix_k, prefix_v, position_ids, mask.
        if getattr(args[0], "ndim", 0) == 5:
            prefix_k, prefix_v, position_ids, attention_mask = args[:4]
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

        # TensorRT-Edge-LLM action ABI:
        # noise, t0, t1, kv_start, rope, position_ids, *K, *V.
        time_steps_t1, kv_start, rope, attention_pos_id = args[:4]
        cache_tensors = args[4:]
        num_layers = int(self.expert.config.num_hidden_layers)
        if len(cache_tensors) != 2 * num_layers:
            raise ValueError(
                f"Expected {2 * num_layers} action KV tensors, "
                f"got {len(cache_tensors)}"
            )

        n_diffusion_tokens = int(noisy_action.shape[1])
        noise = noisy_action.to(torch.float16)
        time_steps_t0 = timestep.to(torch.float16)
        time_steps_t1 = time_steps_t1.to(torch.float16)
        dt = (time_steps_t1 - time_steps_t0).view(-1, 1, 1)
        time_steps_t0 = time_steps_t0.view(-1, 1, 1)
        action_embeds = self.action_in_proj(noise, time_steps_t0).to(dtype=noise.dtype)
        if action_embeds.dim() == 2:
            action_embeds = action_embeds.view(
                noise.shape[0],
                n_diffusion_tokens,
                -1,
            )

        k_caches = cache_tensors[:num_layers]
        v_caches = cache_tensors[num_layers:]
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb,
        )

        hidden = action_embeds
        present_k = []
        present_v = []
        cache_capacity = int(k_caches[0].shape[2])
        rope_indices = (
            attention_pos_id.to(torch.long)
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                rope.shape[-1],
            )
        )
        selected_rope = torch.gather(rope, 1, rope_indices)
        half_dim = int(selected_rope.shape[-1]) // 2
        cos_half = selected_rope[..., :half_dim]
        sin_half = selected_rope[..., half_dim:]
        cos = torch.cat((cos_half, cos_half), dim=-1)
        sin = torch.cat((sin_half, sin_half), dim=-1)

        for layer_index, layer in enumerate(self.expert.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            attention = layer.self_attn
            input_shape = hidden.shape[:-1]
            hidden_shape = (*input_shape, -1, attention.head_dim)
            query = attention.q_norm(
                attention.q_proj(hidden).view(hidden_shape)
            ).transpose(1, 2)
            key = attention.k_norm(
                attention.k_proj(hidden).view(hidden_shape)
            ).transpose(1, 2)
            value = attention.v_proj(hidden).view(hidden_shape).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            query = query.to(hidden.dtype)
            key = key.to(hidden.dtype)

            updated_k = torch.ops.edge_export.action_kv_cache_update.default(
                k_caches[layer_index],
                key,
                kv_start,
            )
            updated_v = torch.ops.edge_export.action_kv_cache_update.default(
                v_caches[layer_index],
                value,
                kv_start,
            )
            present_k.append(updated_k)
            present_v.append(updated_v)

            num_heads = int(attention.config.num_attention_heads)
            num_key_value_heads = int(attention.config.num_key_value_heads)
            groups = num_heads // num_key_value_heads
            expanded_k = updated_k.repeat_interleave(groups, dim=1)
            expanded_v = updated_v.repeat_interleave(groups, dim=1)
            scores = torch.matmul(
                query,
                expanded_k.transpose(-1, -2),
            ) * float(
                getattr(
                    attention,
                    "scaling",
                    1.0 / math.sqrt(attention.head_dim),
                )
            )
            valid_lengths = (kv_start.to(torch.long) + n_diffusion_tokens).view(
                -1, 1, 1, 1
            )
            cache_positions = torch.arange(
                cache_capacity,
                device=scores.device,
                dtype=torch.long,
            ).view(1, 1, 1, -1)
            scores = scores.masked_fill(
                cache_positions >= valid_lengths,
                float("-inf"),
            )
            probabilities = F.softmax(scores.float(), dim=-1).to(query.dtype)
            attention_output = torch.matmul(probabilities, expanded_v)
            attention_output = attention_output.transpose(1, 2).reshape(
                *input_shape,
                -1,
            )
            hidden = residual + attention.o_proj(attention_output)

            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = residual + layer.mlp(hidden)

        hidden = self.expert.norm(hidden)
        velocity = self.action_out_proj(hidden[:, -n_diffusion_tokens:]).reshape_as(
            noise
        )
        denoised = noisy_action + dt.float() * velocity.float()
        return denoised, *present_k, *present_v

    return forward
