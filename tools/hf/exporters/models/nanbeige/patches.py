from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

import torch

from ...plugin.attn_patches import _patch_language_attention, patch_attribute


def _num_loops(config: Any) -> int:
    loop_weights = getattr(config, "loop_loss_weights", None)
    if loop_weights:
        return len(loop_weights) + 1
    return max(int(getattr(config, "num_loops", 1)), 1)


def _patch_nanbeige_attention(original: Callable) -> Callable:
    """Use the Edge language-attention plugin for the loaded remote-code class."""
    return _patch_language_attention(original)


def _patch_nanbeige_language_model(original: Callable) -> Callable:
    """Edge prefill when rope is present; otherwise HF forward."""

    def forward(
        self,
        inputs_embeds=None,
        rope_rotary_cos_sin=None,
        context_lengths=None,
        kvcache_start_index=None,
        last_token_ids=None,
        *past_key_values,
        **kwargs: Any,
    ):
        lm = self if hasattr(self, "layers") else self.model
        lm_head = getattr(self, "lm_head", None)

        hidden = inputs_embeds.to(dtype=next(lm.parameters()).dtype)
        seq_len = inputs_embeds.shape[1]
        physical = len(lm.layers)
        num_loops = _num_loops(lm.config)
        skip_loop_norm = bool(getattr(lm.config, "skip_loop_final_norm", False))
        new_kvs = []

        for loop_idx in range(num_loops):
            for layer_idx, layer in enumerate(lm.layers):
                logical = layer_idx + loop_idx * physical
                residual = hidden
                hidden = layer.input_layernorm(hidden)
                hidden, kv = layer.self_attn(
                    hidden_states=hidden,
                    rope_rotary_cos_sin=rope_rotary_cos_sin,
                    past_key_value=past_key_values[logical],
                    ctx_len=context_lengths,
                    kvcache_start_index=kvcache_start_index,
                )
                hidden = residual + hidden
                residual = hidden
                hidden = layer.post_attention_layernorm(hidden)
                hidden = layer.mlp(hidden)
                hidden = residual + hidden
                new_kvs.append(kv)

            if not skip_loop_norm or loop_idx == num_loops - 1:
                hidden = lm.norm(hidden)

        indices = (
            last_token_ids if last_token_ids.ndim == 1 else last_token_ids.squeeze(-1)
        )
        last_hidden = hidden[
            torch.arange(hidden.shape[0], device=hidden.device, dtype=torch.long),
            indices,
        ]
        logits = lm_head(last_hidden).float()
        prefix_k = torch.stack([kv[:, 0, :, :seq_len, :] for kv in new_kvs], dim=0)
        prefix_v = torch.stack([kv[:, 1, :, :seq_len, :] for kv in new_kvs], dim=0)
        return logits, hidden, prefix_k, prefix_v

    return forward


@contextmanager
def apply_nanbeige_patches(model: Any | None = None) -> Iterator[None]:
    """Patch the live trust-remote-code classes only while compiling."""
    # Nanbeige is loaded with trust_remote_code, so its revision-specific module
    # path is not stable enough for @register_patch; patch the loaded classes.
    if model is None:
        yield
        return
    if not hasattr(model, "lm_head"):
        raise TypeError("Nanbeige export expects NanbeigeForCausalLM")

    decoder = model if hasattr(model, "layers") else model.model
    if not getattr(decoder, "layers", None):
        raise AttributeError("Nanbeige decoder has no layers")

    attn_cls = type(decoder.layers[0].self_attn)
    with patch_attribute(attn_cls, "forward", _patch_nanbeige_attention):
        with patch_attribute(type(model), "forward", _patch_nanbeige_language_model):
            yield
