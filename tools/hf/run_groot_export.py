#!/usr/bin/env python3
"""Smoke EdgeExporter on GR00T (4 engines: vision, language, context_projection, action).

Pass the LeRobot GrootPolicy, not policy._groot_model — prepare_sample_inputs
needs GrootEagleEncodeStep / embodiment_id from the policy wrapper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch  # noqa: E402
import torch_tensorrt  # noqa: E402
from exporters import EdgeConfig, EdgeExporter
from exporters.measure import print_bench
from exporters.plugin.plugin_utils import load_plugins_for_trt
from exporters.utils import force_hf_attention
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.groot import GrootPolicy
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.utils.constants import ACTION, OBS_STATE


def load_groot(device: torch.device) -> GrootPolicy:
    config = GrootConfig(
        base_model_path="nvidia/GR00T-N1.5-3B",
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
    return GrootPolicy(config).to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile", action="store_true", help="Build TRT engines (default: dryrun)"
    )
    parser.add_argument("--engine-dir", default="/tmp/groot_edge_exporter")
    args = parser.parse_args()

    load_plugins_for_trt()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    policy = load_groot(device)
    model = policy._groot_model.to(device=device, dtype=dtype).eval()
    eagle = model.backbone.eagle_model
    force_hf_attention(eagle.vision_model, "eager")
    force_hf_attention(eagle.language_model, "eager")

    exporter = EdgeExporter()
    config = EdgeConfig(model_type="groot", engine_dir=args.engine_dir, max_seq_len=968)

    # Spec tokenizes libero via Eagle chat template because we pass the policy.
    sample_inputs = {"device": device, "dtype": dtype}
    program = exporter.export(policy, sample_inputs, config=config)

    print("engines:", exporter.engines)
    print("runtime keys:", sorted(exporter.sample))

    with torch.no_grad():
        if hasattr(program, "module"):
            velocity = program.module()(**exporter.sample)
        else:
            velocity = program(**exporter.sample)

    out = velocity[0] if isinstance(velocity, (tuple, list)) else velocity
    print("velocity", tuple(out.shape), "mean", float(out.float().mean()))
    print_bench(exporter.bench)


if __name__ == "__main__":
    main()
