from __future__ import annotations

import argparse
from pathlib import Path

import torch
from exporters.compile import compile_component
from exporters.executorch import build_vision_artifact, save_vision_pte
from exporters.models.pi05.patches import PI05
from exporters.models.pi05.vision import Pi05HwcVision
from exporters.plugin.attn_patches import apply_patches
from exporters.plugin.plugin_utils import load_plugins_for_trt
from exporters.spec import ComponentBundle
from exporters.utils import force_hf_attention
from lerobot.policies.pi05 import PI05Policy

DEFAULT_CHECKPOINT = "lerobot/pi05_libero_base"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freshly compile PI0.5 vision and package EdgeLLMBackend .pte"
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--engine-dir", type=Path, default=Path("/tmp/pi05_executorch_fresh")
    )
    parser.add_argument(
        "--pte-path", type=Path, default=Path("/tmp/pi05_vision_edge.pte")
    )
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda", args.device_id)
    dtype = torch.float16
    load_plugins_for_trt()

    policy = PI05Policy.from_pretrained(args.checkpoint).eval()
    policy.model.to(device=device, dtype=dtype).eval()
    paligemma = policy.model.paligemma_with_expert.paligemma.model
    force_hf_attention(paligemma.vision_tower, "eager")

    pixel_values_hwc = torch.randn(4, 224, 224, 3, device=device, dtype=dtype)
    bundle = ComponentBundle(
        module=Pi05HwcVision(paligemma).eval(),
        trace_args=(pixel_values_hwc,),
        save_args=(pixel_values_hwc,),
        input_names=["pixel_values"],
        output_names=["visual_embeds"],
        model_type="vit",
        engine_file="visual.engine",
        extra_config={
            "input_layout": "hwc",
            "input_dtype": "float16",
        },
        trt_settings={
            "disable_tf32": False,
            "use_fp32_acc": False,
            "use_explicit_typing": False,
            "decompose_attention": True,
        },
    )

    with apply_patches(PI05):
        engine_path, _, _ = compile_component(
            bundle,
            name="vision",
            engine_dir=args.engine_dir,
        )

    artifact = build_vision_artifact(engine_path, device_id=args.device_id)
    save_vision_pte(artifact, pixel_values_hwc, args.pte_path)
    print(f"Saved fresh vision engine under {engine_path}")
    print(f"Saved EdgeLLMBackend program to {args.pte_path}")


if __name__ == "__main__":
    main()
