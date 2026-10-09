from __future__ import annotations

import torch
import torch.nn as nn
from torch_tensorrt.hf.exporters.models.common.patches import gather_last_token_hidden
from torch_tensorrt.hf.exporters.models.nemotron.helpers import _decoder, _kind


class NemotronPatch(nn.Module):  # type: ignore[misc]
    """Hybrid decoder: plugin attention / mamba / moe, native MLP."""

    def __init__(self, model: nn.Module):
        super().__init__()
        decoder = _decoder(model)
        self.layers = decoder.layers
        self.norm = decoder.norm_f
        self.lm_head = model.lm_head
        self.kinds = [_kind(block.mixer) for block in self.layers]
        self.num_attn = self.kinds.count("attention")
        self.num_mamba = self.kinds.count("mamba")

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        last_token_ids: torch.Tensor,
        *states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        na, nm = self.num_attn, self.num_mamba
        kvs = list(states[:na])
        convs = list(states[na : na + nm])
        ssms = list(states[na + nm : na + 2 * nm])
        kv_i = conv_i = 0
        hidden = inputs_embeds
        present_kv, present_conv, present_ssm = [], [], []
        for block, kind in zip(self.layers, self.kinds):
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
        hidden = self.norm(hidden)
        last = gather_last_token_hidden(hidden, last_token_ids)
        logits = self.lm_head(last).float()
        return (logits, *present_kv, *present_conv, *present_ssm)
