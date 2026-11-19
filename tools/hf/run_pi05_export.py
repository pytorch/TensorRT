#!/usr/bin/env python3
"""Smoke EdgeExporter on pi05 model.

Pass the LeRobot PI05Policy, not policy.model — prepare_sample_inputs
needs the preprocessor on the policy wrapper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch_tensorrt
from exporters import EdgeConfig, EdgeExporter
from exporters.measure import print_bench
from exporters.plugin.plugin_utils import load_plugins_for_trt
from exporters.utils import force_hf_attention
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.pi05 import PI05Policy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def load_pi05(device: torch.device) -> PI05Policy:
    policy = PI05Policy.from_pretrained("lerobot/pi05_libero_base").eval()
    cfg = policy.config
    cfg.device = str(device)
    cfg.chunk_size = 50
    cfg.n_action_steps = 50
    cfg.max_state_dim = 32
    cfg.max_action_dim = 32
    cfg.input_features = {
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
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(32,))}
    cfg.empty_cameras = 0
    cfg.validate_features()
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile", action="store_true", help="Build TRT engines (default: dryrun)"
    )
    parser.add_argument("--engine-dir", default="/tmp/pi05_edge_exporter")
    args = parser.parse_args()

    load_plugins_for_trt()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    policy = load_pi05(device)
    # Weights on GPU; spec still needs the policy object for the preprocessor.
    policy.model.to(device=device, dtype=dtype).eval()
    paligemma = policy.model.paligemma_with_expert.paligemma.model
    force_hf_attention(paligemma.vision_tower, "eager")
    force_hf_attention(paligemma.language_model, "eager")
    force_hf_attention(policy.model.paligemma_with_expert.gemma_expert.model, "eager")

    exporter = EdgeExporter()
    config = EdgeConfig(
        model_type="pi05",  # optional; inferred from paligemma_with_expert
        engine_dir=args.engine_dir,
        max_seq_len=968,
    )

    # Spec loads libero + preprocessor because we pass the policy, not a tensor dict.
    sample_inputs = {"device": device, "dtype": dtype}

    program = exporter.export(policy, sample_inputs, config=config)

    print("engines:", exporter.engines)

    # Runtime kwargs are tensors only (pixel_values, lang_embeds, rope, KVs, …).
    runtime_kwargs = exporter.sample
    print("runtime keys:", sorted(runtime_kwargs))

    with torch.no_grad():
        if hasattr(program, "module"):
            velocity = program.module()(**runtime_kwargs)
        else:
            velocity = program(**runtime_kwargs)

    out = velocity[0] if isinstance(velocity, (tuple, list)) else velocity
    print("velocity", tuple(out.shape), "mean", float(out.float().mean()))
    print_bench(exporter.bench)


if __name__ == "__main__":
    main()
