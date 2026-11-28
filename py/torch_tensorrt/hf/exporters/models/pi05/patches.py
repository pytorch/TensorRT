from __future__ import annotations

import torch.nn.functional as F
from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding
from torch_tensorrt.hf.exporters.models.common.patches import ActionStepEncoderPatch
from torch_tensorrt.hf.exporters.prefix_cache import PrefixKVCache


class PI05PrefixKVStepEncoderPatch(ActionStepEncoderPatch):
    """PI05 suffix embed + AdaRMS cond, consumed by Gemma action expert."""

    def __init__(self, core):
        super().__init__()
        self.action_in_proj = core.action_in_proj
        self.time_mlp_in = core.time_mlp_in
        self.time_mlp_out = core.time_mlp_out
        self.config = core.config
        self.hidden_size = core.action_in_proj.out_features

    def forward(
        self,
        x_t,
        timestep,
        prefix_k,
        prefix_v,
        position_ids,
        attention_mask,
    ):
        suffix_embs = self.action_in_proj(x_t)

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.hidden_size,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).to(dtype=suffix_embs.dtype)

        adarms_cond = self.time_mlp_in(time_emb)
        adarms_cond = F.silu(adarms_cond)
        adarms_cond = self.time_mlp_out(adarms_cond)
        adarms_cond = F.silu(adarms_cond)

        expert_kwargs = {
            "inputs_embeds": suffix_embs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": PrefixKVCache(prefix_k, prefix_v),
            "use_cache": False,
            "adarms_cond": adarms_cond,
        }
        return (), expert_kwargs, (), {}
