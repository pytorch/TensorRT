from __future__ import annotations

import argparse

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.groot import GrootPolicy
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.utils.constants import ACTION, OBS_STATE

from ...config import EdgeConfig
from ..common.lerobot import apply_lerobot_compat, skip_weight_init

DEFAULT_CHECKPOINT = "nvidia/GR00T-N1.7-3B"

apply_lerobot_compat()


def prepare_export(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
):
    # chunk_size / max_*_dim / image_size keep the GrootConfig N1.7 defaults.
    policy_config = GrootConfig(
        base_model_path=args.checkpoint or DEFAULT_CHECKPOINT,
        device=str(device),
        embodiment_tag="new_embodiment",
        input_features={
            "observation.images.image": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 256, 256)
            ),
            "observation.images.image2": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 256, 256)
            ),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    with skip_weight_init():
        policy = GrootPolicy(policy_config)
    policy = policy.to(device).eval()
    policy._groot_model.to(device=device, dtype=dtype).eval()

    export_config = EdgeConfig(
        model_type="groot",
        engine_dir=args.engine_dir or "/tmp/groot_edge_exporter",
        max_seq_len=args.max_seq_len or 968,
    )
    return (
        policy,
        {"device": device, "dtype": dtype},
        export_config,
        "velocity",
    )
