"""PI05 setattr replacements. Installed for the whole export via ``Pi05Spec.apply_patches``."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch_tensorrt.hf.exporters.models.common.patches import causal_lm_plugin_forward
from torch_tensorrt.hf.exporters.plugin.attn_patches import (
    _patch_language_attention,
    _patch_vision_attention,
    register_patch,
)

PI05 = "pi05"

register_patch(
    PI05,
    "transformers.models.siglip.modeling_siglip.SiglipAttention.forward",
    "transformers.models.siglip.modeling_siglip.SiglipSdpaAttention.forward",
    "transformers.models.siglip.modeling_siglip.SiglipFlashAttention2.forward",
)(_patch_vision_attention)

register_patch(
    PI05,
    "transformers.models.gemma.modeling_gemma.GemmaAttention.forward",
    "transformers.models.gemma2.modeling_gemma2.Gemma2Attention.forward",
)(_patch_language_attention)


@register_patch(
    PI05,
    "lerobot.policies.pi_gemma.PiGemmaModel.forward",
    "transformers.models.gemma.modeling_gemma.GemmaModel.forward",
    "transformers.models.gemma2.modeling_gemma2.Gemma2Model.forward",
)
def _patch_pi05_language_model(original: Callable) -> Callable:
    """Edge prefill when rope is present; otherwise HF / action-expert forward."""

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
        return causal_lm_plugin_forward(
            self,
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
    PI05,
    "lerobot.policies.pi05.modeling_pi05.PI05Pytorch.forward",
)
def _patch_pi05_action_step_forward(original: Callable) -> Callable:
    """One diffusion step when prefix KV is present; otherwise training forward."""

    def forward(
        self,
        x_t,
        timestep,
        prefix_k,
        prefix_v,
        position_ids,
        attention_mask,
        *args,
        **kwargs: Any,
    ):
        if getattr(prefix_k, "ndim", 0) != 5:
            return original(
                self,
                x_t,
                timestep,
                prefix_k,
                prefix_v,
                position_ids,
                attention_mask,
                *args,
                **kwargs,
            )
        from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding
        from torch_tensorrt.hf.exporters.prefix_cache import PrefixKVCache

        suffix_embs = self.action_in_proj(x_t)
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).to(dtype=suffix_embs.dtype)
        adarms_cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))
        expert_out = self.paligemma_with_expert.gemma_expert.model(
            inputs_embeds=suffix_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=PrefixKVCache(prefix_k, prefix_v),
            use_cache=False,
            adarms_cond=adarms_cond,
        )
        hidden = (
            expert_out.last_hidden_state
            if hasattr(expert_out, "last_hidden_state")
            else expert_out
        )
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[0]
        return self.action_out_proj(hidden[:, -int(self.config.chunk_size) :])

    return forward


@register_patch(
    PI05,
    "transformers.models.paligemma.modeling_paligemma.PaliGemmaModel.forward",
)
def _patch_paligemma_image_features(_original: Callable) -> Callable:
    """Match LeRobot ``embed_image`` / HF ``get_image_features``, return a tensor."""

    def forward(self, pixel_values: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        out_dtype = pixel_values.dtype
        if pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.to(torch.float32)
        image_outputs = self.vision_tower(pixel_values, **kwargs)
        hidden = (
            image_outputs.last_hidden_state
            if hasattr(image_outputs, "last_hidden_state")
            else image_outputs[0]
        )
        features = self.multi_modal_projector(hidden)
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    return forward
