from __future__ import annotations

import argparse

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.groot import GrootPolicy
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.utils.constants import ACTION, OBS_STATE

from ...config import EdgeConfig
from ...utils import force_hf_attention

DEFAULT_CHECKPOINT = "nvidia/GR00T-N1.5-3B"


def prepare_export(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
):
    policy_config = GrootConfig(
        base_model_path=args.checkpoint or DEFAULT_CHECKPOINT,
        device=str(device),
        embodiment_tag="new_embodiment",
        chunk_size=50,
        n_action_steps=50,
        max_state_dim=64,
        max_action_dim=32,
        image_size=(224, 224),
        tokenizer_assets_repo="lerobot/eagle2hg-processor-groot-n1p5",
        input_features={
            "observation.images.image": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            "observation.images.image2": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(32,))},
    )
    policy = GrootPolicy(policy_config).to(device).eval()
    model = policy._groot_model.to(device=device, dtype=dtype).eval()
    eagle = model.backbone.eagle_model
    force_hf_attention(eagle.vision_model, "eager")
    force_hf_attention(eagle.language_model, "eager")

    export_config = EdgeConfig(
        model_type="groot",
        engine_dir=args.engine_dir or "/tmp/groot_edge_exporter",
        max_seq_len=args.max_seq_len or 968,
        runtime_export=False,
    )
    return (
        policy,
        {"device": device, "dtype": dtype},
        export_config,
        "velocity",
    )
