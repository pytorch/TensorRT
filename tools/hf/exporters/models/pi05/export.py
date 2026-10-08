from __future__ import annotations

import argparse

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.pi05 import PI05Policy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ...config import EdgeConfig
from ...utils import force_hf_attention
from ..common.lerobot import apply_lerobot_compat, skip_weight_init

DEFAULT_CHECKPOINT = "lerobot/pi05_libero_base"


def _patch_create_causal_mask() -> None:
    """lerobot pins transformers<5.6 and passes `cache_position` to
    `create_causal_mask`; newer transformers (needed for
    `transformers.exporters`) dropped that argument. Drop it when unsupported."""
    import inspect

    from lerobot.policies import pi_gemma

    orig = pi_gemma.create_causal_mask
    if orig is None or "cache_position" in inspect.signature(orig).parameters:
        return

    def create_causal_mask(*args, cache_position=None, **kwargs):
        return orig(*args, **kwargs)

    pi_gemma.create_causal_mask = create_causal_mask


def _patch_siglip_vision_keys() -> None:
    """Newer transformers dropped the `vision_model` level from SigLIP's
    `vision_tower`, but the lerobot checkpoints (and lerobot's key fixup) still
    use `vision_tower.vision_model.*`. Rename checkpoint keys to whichever
    layout the instantiated model uses, so both old and new transformers load."""
    old, new = "vision_tower.vision_model.", "vision_tower."
    orig = PI05Policy._fix_pytorch_state_dict_keys
    if getattr(orig, "_siglip_patched", False):
        return

    def fix_keys(self, state_dict, model_config):
        fixed = orig(self, state_dict, model_config)
        # Normalize to the flat layout, then re-nest if the model expects it.
        fixed = {k.replace(old, new): v for k, v in fixed.items()}
        if any(old in k for k in self.state_dict()):
            fixed = {k.replace(new, old): v for k, v in fixed.items()}
        return fixed

    fix_keys._siglip_patched = True
    PI05Policy._fix_pytorch_state_dict_keys = fix_keys


apply_lerobot_compat()
_patch_create_causal_mask()
_patch_siglip_vision_keys()


def prepare_export(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
):
    with skip_weight_init():
        policy = PI05Policy.from_pretrained(
            args.checkpoint or DEFAULT_CHECKPOINT
        ).eval()
    config = policy.config
    config.device = str(device)
    config.chunk_size = 50
    config.n_action_steps = 50
    config.max_state_dim = 32
    config.max_action_dim = 32
    config.input_features = {
        f"{OBS_IMAGES}.image": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 224, 224)
        ),
        f"{OBS_IMAGES}.image2": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 224, 224)
        ),
        f"{OBS_IMAGES}.image3": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 224, 224)
        ),
        f"{OBS_IMAGES}.image4": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 224, 224)
        ),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(32,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(32,))
    }
    config.empty_cameras = 0
    config.validate_features()

    policy.model.to(device=device, dtype=dtype).eval()
    paligemma = policy.model.paligemma_with_expert.paligemma.model
    force_hf_attention(paligemma.vision_tower, "eager")
    force_hf_attention(paligemma.language_model, "eager")
    force_hf_attention(policy.model.paligemma_with_expert.gemma_expert.model, "eager")

    export_config = EdgeConfig(
        model_type="pi05",
        engine_dir=args.engine_dir or "/tmp/pi05",
        max_seq_len=args.max_seq_len or 968,
    )
    return (
        policy,
        {"device": device, "dtype": dtype},
        export_config,
        "velocity",
    )
