"""GR00T setattr replacements. Installed for the whole export via ``GrootSpec.apply_patches``."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

import torch

from ...plugin.attn_patches import (
    _patch_language_attention,
    _patch_vision_attention,
    register_patch,
)
from ..common.patches import causal_lm_plugin_forward

GROOT = "groot"

register_patch(
    GROOT,
    "transformers.models.internvl.modeling_internvl.InternVLVisionAttention.forward",
    "transformers.models.siglip.modeling_siglip.SiglipAttention.forward",
    "transformers.models.siglip.modeling_siglip.SiglipSdpaAttention.forward",
    "transformers.models.siglip.modeling_siglip.SiglipFlashAttention2.forward",
)(_patch_vision_attention)

register_patch(
    GROOT,
    "transformers.models.llama.modeling_llama.LlamaAttention.forward",
    "transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward",
    "transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward",
)(_patch_language_attention)


@register_patch(
    GROOT,
    "lerobot.policies.groot.eagle2_hg_model.modeling_eagle2_5_vl.Eagle25VLForConditionalGeneration.forward",
)
def _patch_eagle_image_features(original: Callable) -> Callable:
    """Vision-only compile calls ``eagle(pixel_values)``; otherwise the full VLM forward."""

    def forward(self, pixel_values, input_ids=None, **kwargs: Any):
        if input_ids is None:
            return self.extract_feature(pixel_values)
        return original(self, pixel_values, input_ids, **kwargs)

    return forward


@contextmanager
def apply_groot_patches(model: Any | None = None) -> Iterator[None]:
    """Family setattr, plus the live Eagle class.

    LeRobot builds Eagle with ``AutoModel.from_config(..., trust_remote_code=True)``,
    so the running class is HuggingFace ``transformers_modules`` code, not
    ``lerobot.policies.groot.eagle2_hg_model``. The dotted path still covers the
    in-tree copy; this patches ``type(eagle_model)`` so vision
    ``eagle(pixel_values)`` hits ``extract_feature``.
    """
    from ...plugin.attn_patches import apply_patches, patch_attribute
    from .helpers import _groot

    with apply_patches(GROOT):
        if model is None:
            yield
            return
        eagle_cls = type(_groot(model).backbone.eagle_model)
        with patch_attribute(eagle_cls, "forward", _patch_eagle_image_features):
            yield


@register_patch(
    GROOT,
    "transformers.models.llama.modeling_llama.LlamaForCausalLM.forward",
    "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.forward",
    "transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.forward",
    "transformers.models.llama.modeling_llama.LlamaModel.forward",
    "transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward",
    "transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward",
)
def _patch_groot_language_model(original: Callable) -> Callable:
    """Edge prefill when rope is present; otherwise HF causal-LM forward."""

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
        decoder = self if hasattr(self, "layers") else self.model
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
    GROOT,
    "lerobot.policies.groot.groot_n1.GR00TN15.forward",
)
def _patch_groot_context_projection(original: Callable) -> Callable:
    """One linear + VLLN + VL attention when a single hidden tensor is present."""

    def forward(self, hidden_states, *args, **kwargs: Any):
        if args:
            return original(self, hidden_states, *args, **kwargs)
        context_embs = self.backbone.eagle_linear(hidden_states)
        vlln = self.action_head.vlln
        weight = getattr(vlln, "weight", None)
        if weight is not None:
            context_embs = context_embs.to(dtype=weight.dtype)
        context_embs = vlln(context_embs)
        return self.action_head.vl_self_attention(context_embs)

    return forward


@register_patch(
    GROOT,
    "lerobot.policies.groot.action_head.flow_matching_action_head.FlowmatchingActionHead.forward",
)
def _patch_groot_action_step_forward(original: Callable) -> Callable:
    """One DiT velocity step when Edge action I/O is present; otherwise training."""

    def forward(
        self,
        actions,
        timestep=None,
        context_embs=None,
        state=None,
        embodiment_id=None,
        *args,
        **kwargs: Any,
    ):
        if context_embs is None:
            return original(self, actions, timestep, *args, **kwargs)
        state_features = self.state_encoder(state, embodiment_id)
        action_features = self.action_encoder(actions, timestep, embodiment_id)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1],
                dtype=torch.long,
                device=action_features.device,
            )
            action_features = action_features + self.position_embedding(
                pos_ids
            ).unsqueeze(0)
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            context_embs.shape[0],
            -1,
            -1,
        )
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
        expert_out = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=context_embs,
            timestep=timestep,
        )
        hidden = (
            expert_out.last_hidden_state
            if hasattr(expert_out, "last_hidden_state")
            else expert_out
        )
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[0]
        action_hidden = hidden[:, -int(self.config.action_horizon) :]
        return self.action_decoder(action_hidden, embodiment_id)

    return forward


@register_patch(
    GROOT,
    "lerobot.policies.groot.action_head.flow_matching_action_head.CategorySpecificLinear.forward",
)
def _patch_category_specific_linear(_original: Callable) -> Callable:
    """``index_select`` + ``bmm`` is the TensorRT-friendly form of ``W[cat_ids]``."""

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        cat_ids = cat_ids.to(dtype=torch.long)
        selected_w = torch.index_select(self.W, dim=0, index=cat_ids).to(dtype=x.dtype)
        selected_b = torch.index_select(self.b, dim=0, index=cat_ids).to(dtype=x.dtype)
        return torch.bmm(x, selected_w) + selected_b.unsqueeze(1)

    return forward
