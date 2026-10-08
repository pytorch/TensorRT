from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Iterator

import torch
import torch.nn.functional as F

from ...plugin.attn_patches import patch_attribute
from .helpers import decoder_model, language_model


def _conv_bias(conv, projected: torch.Tensor) -> torch.Tensor:
    bias = getattr(conv, "bias", None)
    if bias is not None:
        return bias.to(device=projected.device, dtype=projected.dtype)
    return torch.zeros(
        int(conv.weight.shape[0]),
        device=projected.device,
        dtype=projected.dtype,
    )


def _plugin_convolution(
    conv,
    projected: torch.Tensor,
    state: torch.Tensor,
    context_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_size = int(conv.weight.shape[-1])
    output, next_state = torch.ops.trt.causal_conv1d.default(
        projected,
        conv.weight,
        _conv_bias(conv, projected),
        state,
        context_lengths,
        1,
        kernel_size - 1,
        1,
        int(projected.shape[-1]),
    )
    return F.silu(output), next_state


def _patch_kda_attention(original: Callable) -> Callable:
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        cache_params=None,
        *,
        context_lengths=None,
        conv_state_q=None,
        conv_state_k=None,
        conv_state_v=None,
        recurrent_state=None,
        **kwargs,
    ):
        if recurrent_state is None:
            return original(
                self,
                hidden_states,
                attention_mask=attention_mask,
                cache_params=cache_params,
                **kwargs,
            )

        del attention_mask, cache_params, kwargs
        if context_lengths is None:
            raise ValueError("Kimi KDA export requires context_lengths")
        if conv_state_q is None or conv_state_k is None or conv_state_v is None:
            raise ValueError("Kimi KDA export requires Q/K/V convolution states")

        q, _ = _plugin_convolution(
            self.q_conv1d,
            self.q_proj(hidden_states),
            conv_state_q,
            context_lengths,
        )
        k, _ = _plugin_convolution(
            self.k_conv1d,
            self.k_proj(hidden_states),
            conv_state_k,
            context_lengths,
        )
        v, _ = _plugin_convolution(
            self.v_conv1d,
            self.v_proj(hidden_states),
            conv_state_v,
            context_lengths,
        )

        batch_size, seq_len, _ = hidden_states.shape
        num_heads = int(self.num_heads)
        head_dim = int(self.head_dim)
        q = q.reshape(batch_size, seq_len, num_heads, head_dim)
        k = k.reshape(batch_size, seq_len, num_heads, head_dim)
        v = v.reshape(batch_size, seq_len, num_heads, head_dim)
        gate = self.f_b_proj(self.f_a_proj(hidden_states)).reshape(
            batch_size, seq_len, num_heads, head_dim
        )
        beta = self.b_proj(hidden_states).to(dtype=hidden_states.dtype)
        lower_bound = getattr(self, "gate_lower_bound", None)

        output, _ = torch.ops.trt.kimi_kda_plugin.default(
            q,
            k,
            v,
            gate,
            beta,
            self.A_log.float(),
            self.dt_bias.reshape(num_heads, head_dim).to(hidden_states.dtype),
            recurrent_state,
            context_lengths,
            float(lower_bound if lower_bound is not None else -5.0),
            lower_bound is not None,
        )

        output_gate = (
            self.g_proj(hidden_states)
            if self.use_full_rank_gate
            else self.g_b_proj(self.g_a_proj(hidden_states))
        ).reshape(batch_size, seq_len, num_heads, head_dim)

        output_float = output.float()
        eps = float(
            getattr(
                self.o_norm,
                "variance_epsilon",
                getattr(self.o_norm, "eps", 1e-5),
            )
        )
        output = (
            output_float
            * torch.rsqrt(output_float.square().mean(-1, keepdim=True) + eps)
        ).to(hidden_states.dtype)
        output = output * self.o_norm.weight * output_gate.sigmoid()
        return self.o_proj(output.reshape(batch_size, seq_len, -1))

    return forward


def _patch_mla_attention(original: Callable) -> Callable:
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        del position_ids, past_key_values, kwargs
        batch_size, seq_len, _ = hidden_states.shape
        query_shape = (batch_size, seq_len, -1, self.q_head_dim)
        key_shape = (
            batch_size,
            seq_len,
            -1,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        if self.q_lora_rank is not None:
            query = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            query = self.q_proj(hidden_states)
        query = query.reshape(query_shape).transpose(1, 2)
        query_nope, query_position = torch.split(
            query,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        key_latent, key_position = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        key_value = self.kv_b_proj(self.kv_a_layernorm(key_latent))
        key_value = key_value.reshape(key_shape).transpose(1, 2)
        key_nope, value = torch.split(
            key_value,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        key_position = key_position.reshape(
            batch_size, 1, seq_len, self.qk_rope_head_dim
        ).expand(*key_nope.shape[:-1], -1)
        query = torch.cat((query_nope, query_position), dim=-1)
        key = torch.cat((key_nope, key_position), dim=-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) * float(self.scaling)
        if attention_mask is not None:
            scores = scores + attention_mask[..., : key.shape[-2]]
        probabilities = scores.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, -1)
        if self.use_output_gate:
            output = output * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(output)

    return forward


def _patch_sparse_moe(original: Callable) -> Callable:
    def forward(self, hidden_states):
        identity = hidden_states
        original_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        flattened = hidden_states.reshape(-1, hidden_states.shape[-1])

        if self.use_latent_moe:
            flattened = self.routed_expert_down_proj(flattened)

        expert_outputs = torch.stack(
            [expert(flattened) for expert in self.experts],
            dim=1,
        )
        routing = F.one_hot(
            topk_indices,
            num_classes=int(self.num_experts),
        ).to(topk_weights.dtype)
        routing = (routing * topk_weights.unsqueeze(-1)).sum(dim=1)
        output = (expert_outputs * routing.unsqueeze(-1)).sum(dim=1)
        output = output.to(flattened.dtype)

        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                output = self.routed_expert_norm(output)
            output = self.routed_expert_up_proj(output)
        output = output.reshape(original_shape)
        if self.config.num_shared_experts is not None:
            output = output + self.shared_experts(identity).to(output.dtype)
        return output

    return forward


def _causal_mask(
    context_lengths: torch.Tensor,
    seq_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.arange(seq_len, device=context_lengths.device)
    causal = positions.unsqueeze(0) <= positions.unsqueeze(1)
    valid_keys = positions.unsqueeze(0) < context_lengths.unsqueeze(1)
    allowed = causal.unsqueeze(0) & valid_keys.unsqueeze(1)
    mask = torch.zeros(
        context_lengths.shape[0],
        1,
        seq_len,
        seq_len,
        device=context_lengths.device,
        dtype=dtype,
    )
    return mask.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)


def _patch_language_forward(original: Callable) -> Callable:
    def forward(
        self,
        inputs_embeds=None,
        context_lengths=None,
        last_token_ids=None,
        *kda_states,
        **kwargs,
    ):
        if context_lengths is None:
            return original(self, inputs_embeds=inputs_embeds, **kwargs)
        if inputs_embeds is None or last_token_ids is None:
            raise ValueError("Kimi export requires embeddings and last-token indices")

        decoder = self.model
        hidden = inputs_embeds.to(dtype=next(decoder.parameters()).dtype)
        batch_size, seq_len, hidden_size = hidden.shape
        causal_mask = _causal_mask(context_lengths, seq_len, hidden.dtype)
        block_residual = hidden.new_zeros(batch_size * seq_len, 0, hidden_size)
        state_index = 0

        for layer in decoder.layers:
            layer_kwargs: dict[str, Any] = {}
            layer_mask = causal_mask
            if layer.is_linear_attn:
                if state_index + 4 > len(kda_states):
                    raise ValueError("Missing flattened Kimi KDA states")
                layer_kwargs = {
                    "context_lengths": context_lengths,
                    "conv_state_q": kda_states[state_index],
                    "conv_state_k": kda_states[state_index + 1],
                    "conv_state_v": kda_states[state_index + 2],
                    "recurrent_state": kda_states[state_index + 3],
                }
                state_index += 4
                layer_mask = None

            hidden, block_residual = layer(
                hidden,
                attention_mask=layer_mask,
                block_residual=block_residual,
                **layer_kwargs,
            )

        hidden = decoder._apply_output_attn_res(hidden, block_residual)
        hidden = decoder.norm(hidden)
        indices = (
            last_token_ids if last_token_ids.ndim == 1 else last_token_ids.squeeze(-1)
        )
        selected = hidden[
            torch.arange(batch_size, device=hidden.device, dtype=torch.long),
            indices,
        ]
        return self.lm_head(selected).float()

    return forward


@contextmanager
def apply_kimik3_patches(model: Any | None = None) -> Iterator[None]:
    if model is None:
        yield
        return

    language = language_model(model)
    decoder = decoder_model(model)
    kda_class = mla_class = moe_class = None
    for layer in decoder.layers:
        if layer.is_linear_attn:
            kda_class = type(layer.self_attn)
        else:
            mla_class = type(layer.self_attn)
        if hasattr(layer, "block_sparse_moe"):
            moe_class = type(layer.block_sparse_moe)

    with ExitStack() as stack:
        stack.enter_context(
            patch_attribute(type(language), "forward", _patch_language_forward)
        )
        if kda_class is not None:
            stack.enter_context(
                patch_attribute(kda_class, "forward", _patch_kda_attention)
            )
        if mla_class is not None:
            stack.enter_context(
                patch_attribute(mla_class, "forward", _patch_mla_attention)
            )
        if moe_class is not None:
            stack.enter_context(
                patch_attribute(moe_class, "forward", _patch_sparse_moe)
            )
        yield
