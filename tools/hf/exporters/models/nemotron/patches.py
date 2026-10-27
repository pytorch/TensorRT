"""Nemotron setattr replacements. Installed for the whole export via ``NemotronSpec.apply_patches``."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

from ...plugin.attn_patches import (
    apply_patches,
    register_patch,
)
from ..common.patches import gather_last_token_hidden
from .helpers import _decoder, _kind

NEMOTRON = "nemotron"


@register_patch(
    NEMOTRON,
    "transformers.models.nemotron_h.modeling_nemotron_h.NemotronHForCausalLM.forward",
)
def _patch_nemotron_causal_lm(original: Callable) -> Callable:
    """Hybrid plugin prefill when rope is present; otherwise HF causal-LM forward."""

    def forward(
        self,
        inputs_embeds=None,
        rope_rotary_cos_sin=None,
        context_lengths=None,
        kvcache_start_index=None,
        last_token_ids=None,
        *states,
        **kwargs: Any,
    ):
        if rope_rotary_cos_sin is None:
            return original(self, inputs_embeds=inputs_embeds, **kwargs)
        decoder = _decoder(self)
        kinds = [_kind(block.mixer) for block in decoder.layers]
        na, nm = kinds.count("attention"), kinds.count("mamba")
        kvs = list(states[:na])
        convs = list(states[na : na + nm])
        ssms = list(states[na + nm : na + 2 * nm])
        kv_i = conv_i = 0
        hidden = inputs_embeds
        present_kv, present_conv, present_ssm = [], [], []
        for block, kind in zip(decoder.layers, kinds):
            residual = hidden
            hidden = block.norm(hidden)
            mixer = block.mixer
            if kind == "attention":
                hidden, kv = mixer(
                    hidden_states=hidden,
                    rope_rotary_cos_sin=rope_rotary_cos_sin,
                    past_key_value=kvs[kv_i],
                    ctx_len=context_lengths,
                    kvcache_start_index=kvcache_start_index,
                )
                present_kv.append(kv)
                kv_i += 1
            elif kind == "mamba":
                hidden, conv_out, ssm_out = mixer(
                    hidden, convs[conv_i], ssms[conv_i], context_lengths
                )
                present_conv.append(conv_out)
                present_ssm.append(ssm_out)
                conv_i += 1
            else:
                hidden = mixer(hidden)
            hidden = residual + hidden
        hidden = decoder.norm_f(hidden)
        last = gather_last_token_hidden(hidden, last_token_ids)
        logits = self.lm_head(last).float()
        return (logits, *present_kv, *present_conv, *present_ssm)

    return forward


@contextmanager
def apply_nemotron_patches(model: Any | None = None) -> Iterator[None]:
    """Class setattr plus mixer plugin wrappers (MoE packing needs the instance)."""
    from ...plugin.plugin_utils import (
        patch_nemotron_mixers,
        restore_attention,
    )

    restore = []
    try:
        if model is not None:
            restore = patch_nemotron_mixers(model, model.config)
            for block in _decoder(model).layers:
                prepare = getattr(block.mixer, "prepare_for_export", None)
                if callable(prepare):
                    prepare()
        with apply_patches(NEMOTRON):
            yield
    finally:
        if restore:
            restore_attention(restore)
