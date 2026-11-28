from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_tensor(x):
    """Unwrap tuple/list outputs from patched attention modules."""
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


def language_decoder(language: nn.Module) -> nn.Module:
    """Inner module that owns ``.layers``.

    Paligemma / PI05 store layers on ``language_model`` itself. HF
    ``*ForCausalLM`` stores them on ``language_model.model``. Prefer ``.layers``
    so a stray ``.model`` attribute cannot silently pick the wrong submodule.
    """
    if hasattr(language, "layers"):
        return language
    inner = getattr(language, "model", None)
    if isinstance(inner, nn.Module) and hasattr(inner, "layers"):
        return inner
    raise AttributeError(f"{type(language).__name__} has no decoder .layers")


def causal_lm_plugin_forward(
    lm: nn.Module,
    inputs_embeds: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    context_lengths: torch.Tensor,
    kvcache_start_index: torch.Tensor,
    last_token_ids: torch.Tensor,
    ds_stack: torch.Tensor,
    *past_key_values: torch.Tensor,
    lm_head: nn.Module | None = None,
    select_layer: int = -1,
):
    """Prefill loop used by Edge language engines (plugin attention + prefix KV).

    ``ds_stack`` is ``[num_layers, B, S, H]``. Each layer adds its slice; PI05
    passes zeros so the add does not change hidden.
    """
    lm_dtype = next(lm.parameters()).dtype
    hidden = _as_tensor(inputs_embeds).to(dtype=lm_dtype)
    seq_len = inputs_embeds.shape[1]
    context_hidden = hidden if select_layer == 0 else None
    new_kvs = []

    for i, layer in enumerate(lm.layers):
        residual = hidden
        hidden = _as_tensor(layer.input_layernorm(hidden))
        hidden, kv = layer.self_attn(
            hidden_states=hidden,
            rope_rotary_cos_sin=rope_rotary_cos_sin,
            past_key_value=past_key_values[i],
            ctx_len=context_lengths,
            kvcache_start_index=kvcache_start_index,
        )
        hidden = _as_tensor(hidden)
        hidden = residual + hidden

        residual = hidden
        hidden = _as_tensor(layer.post_attention_layernorm(hidden))
        hidden = _as_tensor(layer.mlp(hidden))
        hidden = residual + hidden
        new_kvs.append(kv)

        hidden = hidden + ds_stack[i, :, :seq_len, :].to(dtype=hidden.dtype)

        if select_layer > 0 and (i + 1) == select_layer:
            context_hidden = hidden

    hidden = _as_tensor(lm.norm(hidden))
    if context_hidden is None:
        context_hidden = hidden

    indices = last_token_ids if last_token_ids.ndim == 1 else last_token_ids.squeeze(-1)
    last_hidden = hidden[
        torch.arange(hidden.shape[0], device=hidden.device, dtype=torch.long),
        indices,
    ]
    if lm_head is not None:
        logits = lm_head(last_hidden).float()
    else:
        embed = getattr(lm, "embed_tokens", None)
        logits = F.linear(last_hidden, embed.weight).float()
    prefix_k = torch.stack([kv[:, 0, :, :seq_len, :] for kv in new_kvs], dim=0)
    prefix_v = torch.stack([kv[:, 1, :, :seq_len, :] for kv in new_kvs], dim=0)
    return logits, context_hidden, prefix_k, prefix_v
