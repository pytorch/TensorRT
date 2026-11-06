from __future__ import annotations

import argparse

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.pi05 import PI05Policy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ...config import EdgeConfig
from ...utils import force_hf_attention

DEFAULT_CHECKPOINT = "lerobot/pi05_libero_base"


def prepare_export(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
):
    policy = PI05Policy.from_pretrained(args.checkpoint or DEFAULT_CHECKPOINT).eval()
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
