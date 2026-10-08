"""GR00T N1.7 setattr replacements. Installed for the whole export via ``GrootSpec.apply_patches``."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from ...plugin.attn_patches import _patch_language_attention, register_patch
from ..common.patches import causal_lm_plugin_forward

GROOT = "groot"

register_patch(
    GROOT,
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward",
)(_patch_language_attention)


@register_patch(
    GROOT,
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextModel.forward",
)
def _patch_groot_language_model(original: Callable) -> Callable:
    """Edge prefill when rope is present; otherwise the HF text-model forward.

    ``lm_hidden_states`` is the last kept decoder layer *before* the final norm,
    which is what GR00T N1.7 feeds the action head (``select_layer`` truncation).
    """

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
            select_layer=len(decoder.layers),
        )

    return forward


def action_velocity(
    head: nn.Module,
    actions: torch.Tensor,
    timestep: torch.Tensor,
    context_embs: torch.Tensor,
    state: torch.Tensor,
    embodiment_id: torch.Tensor,
    image_mask: torch.Tensor,
    backbone_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """One denoising step of ``GR00TN17ActionHead.get_action_with_features``."""
    state_features = head.state_encoder(
        state.reshape(state.shape[0], 1, -1), embodiment_id
    )
    action_features = head.action_encoder(actions, timestep, embodiment_id)
    if head.config.add_pos_embed:
        pos_ids = torch.arange(
            action_features.shape[1], dtype=torch.long, device=action_features.device
        )
        action_features = action_features + head.position_embedding(pos_ids).unsqueeze(0)
    sa_embs = torch.cat((state_features, action_features), dim=1)
    if head.config.use_alternate_vl_dit:
        model_output = head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=context_embs,
            timestep=timestep,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
        )
    else:
        model_output = head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=context_embs,
            timestep=timestep,
        )
    pred = head.action_decoder(model_output, embodiment_id)
    return pred[:, -int(head.action_horizon) :]


@register_patch(
    GROOT,
    "lerobot.policies.groot.groot_n1_7.GR00TN17ActionHead.forward",
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
        image_mask=None,
        backbone_attention_mask=None,
        *args,
        **kwargs: Any,
    ):
        if context_embs is None:
            return original(self, actions, timestep, *args, **kwargs)
        return action_velocity(
            self,
            actions,
            timestep,
            context_embs,
            state,
            embodiment_id,
            image_mask,
            backbone_attention_mask,
        )

    return forward


@register_patch(
    GROOT,
    "lerobot.policies.groot.groot_n1_7.CategorySpecificLinear.forward",
)
def _patch_category_specific_linear(_original: Callable) -> Callable:
    """``index_select`` + ``bmm`` is the TensorRT-friendly form of ``W[cat_ids]``."""

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        cat_ids = cat_ids.to(dtype=torch.long)
        selected_w = torch.index_select(self.W, dim=0, index=cat_ids).to(dtype=x.dtype)
        selected_b = torch.index_select(self.b, dim=0, index=cat_ids).to(dtype=x.dtype)
        return torch.bmm(x, selected_w) + selected_b.unsqueeze(1)

    return forward
