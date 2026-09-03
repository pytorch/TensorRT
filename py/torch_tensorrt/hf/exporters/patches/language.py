from __future__ import annotations

import torch
import torch.nn as nn


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


def gather_last_token_hidden(
    hidden_states: torch.Tensor,
    last_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather [B, S, H] at last_token_ids [B] or [B, 1] -> [B, H] for lm_head."""
    if last_token_ids.ndim == 1:
        indices = last_token_ids
    else:
        indices = last_token_ids.squeeze(-1)
    batch_idx = torch.arange(
        hidden_states.shape[0],
        device=hidden_states.device,
        dtype=torch.long,
    )
    return hidden_states[batch_idx, indices]


class CausalLMPatch(nn.Module):
    """Edge-LLM causal LM: manual decoder loop -> logits + lm_hidden_states + prefix KV.

    External RoPE and KV-cache controls match ``LLMEngineRunner`` prefill/decode.
    ``select_layer=-1`` (default for GR00T TRT) uses final RMSNorm hidden for
    context; positive values capture an intermediate layer output pre-norm.

    ``ds_stack`` is always an input (``[num_ds, B, S, H]``). Models without
    deepstack pass ``num_ds=0`` (empty leading dim); the add is a no-op.
    """

    def __init__(
        self,
        lm: nn.Module,
        lm_head: nn.Module,
        *,
        select_layer: int = -1,
    ):
        super().__init__()
        self.lm = lm
        self.lm_head = lm_head
        self.select_layer = int(select_layer)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        last_token_ids: torch.Tensor,
        ds_stack: torch.Tensor,
        *past_key_values: torch.Tensor,
    ):
        lm_dtype = next(self.lm.parameters()).dtype
        hidden = _as_tensor(inputs_embeds).to(dtype=lm_dtype)
        seq_len = inputs_embeds.shape[1]
        num_ds = int(ds_stack.shape[0])
        context_hidden = hidden if self.select_layer == 0 else None
        new_kvs = []

        for i, layer in enumerate(self.lm.layers):
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

            if i < num_ds:
                hidden = hidden + ds_stack[i, :, :seq_len, :].to(dtype=hidden.dtype)

            if self.select_layer > 0 and (i + 1) == self.select_layer:
                context_hidden = hidden

        hidden = _as_tensor(self.lm.norm(hidden))
        if context_hidden is None:
            context_hidden = hidden

        last_hidden = gather_last_token_hidden(hidden, last_token_ids)
        logits = self.lm_head(last_hidden).float()

        prefix_k = torch.stack(
            [kv[:, 0, :, :seq_len, :] for kv in new_kvs],
            dim=0,
        )
        prefix_v = torch.stack(
            [kv[:, 1, :, :seq_len, :] for kv in new_kvs],
            dim=0,
        )
        return logits, context_hidden, prefix_k, prefix_v


# specific to gr00t, before action another project is required for context embeddings
class ContextProjectionPatch(nn.Module):
    """eagle_linear -> vlln -> vl_self_attention (matches eager context path)."""

    def __init__(self, eagle_linear, vlln, vl_self_attention):
        super().__init__()
        self.eagle_linear = eagle_linear
        self.vlln = vlln
        self.vl_self_attention = vl_self_attention

    def forward(self, hidden_states: torch.Tensor):
        context_embs = self.eagle_linear(hidden_states)

        vlln_weight = getattr(self.vlln, "weight", None)
        if vlln_weight is not None:
            context_embs = context_embs.to(dtype=vlln_weight.dtype)

        context_embs = self.vlln(context_embs)
        context_embs = self.vl_self_attention(context_embs)
        return context_embs
