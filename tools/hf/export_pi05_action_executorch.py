from __future__ import annotations

import argparse
from pathlib import Path

import torch
from exporters.compile import compile_component
from exporters.executorch import build_action_artifact, save_action_pte
from exporters.models.common.patches import language_decoder
from exporters.models.pi05.helpers import make_pi05_suffix_position_and_mask
from exporters.models.pi05.patches import PI05
from exporters.plugin.attn_patches import apply_patches
from exporters.plugin.plugin_utils import load_plugins_for_trt
from exporters.spec import ComponentBundle
from lerobot.policies.pi05 import PI05Policy

DEFAULT_CHECKPOINT = "lerobot/pi05_libero_base"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freshly compile one PI0.5 action step and package an EdgeLLMBackend .pte"
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=Path("/tmp/pi05_action_executorch_fresh"),
    )
    parser.add_argument(
        "--pte-path",
        type=Path,
        default=Path("/tmp/pi05_action_edge.pte"),
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--prefix-len", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda", args.device_id)
    dtype = torch.float16
    load_plugins_for_trt()

    policy = PI05Policy.from_pretrained(args.checkpoint).eval()
    policy.config.chunk_size = 50
    policy.config.max_action_dim = 32
    policy.model.to(device=device, dtype=dtype).eval()
    core = policy.model
    language = core.paligemma_with_expert.paligemma.model.language_model
    decoder = language_decoder(language)

    batch_size = 1
    chunk_size = int(core.config.chunk_size)
    action_dim = int(core.config.max_action_dim)
    language_config = language.config
    num_kv_heads = int(language_config.num_key_value_heads)
    head_dim = int(
        getattr(
            language_config,
            "head_dim",
            language_config.hidden_size // language_config.num_attention_heads,
        )
    )

    x_t = torch.randn(
        batch_size,
        chunk_size,
        action_dim,
        device=device,
        dtype=dtype,
    )
    timestep = torch.ones(batch_size, device=device, dtype=torch.float32)
    prefix_k = torch.zeros(
        len(decoder.layers),
        batch_size,
        num_kv_heads,
        args.prefix_len,
        head_dim,
        device=device,
        dtype=dtype,
    )
    prefix_v = torch.zeros_like(prefix_k)
    prefix_pad_mask = torch.ones(
        batch_size,
        args.prefix_len,
        device=device,
        dtype=torch.bool,
    )
    position_ids, attention_mask = make_pi05_suffix_position_and_mask(
        core,
        prefix_pad_mask,
        x_t,
        device,
    )
    action_inputs = (
        x_t,
        timestep,
        prefix_k,
        prefix_v,
        position_ids,
        attention_mask,
    )
    bundle = ComponentBundle(
        module=core.eval(),
        trace_args=action_inputs,
        save_args=action_inputs,
        input_names=[
            "x_t",
            "timestep",
            "prefix_k",
            "prefix_v",
            "position_ids",
            "attention_mask",
        ],
        output_names=["velocity"],
        model_type="action",
        engine_file="action.engine",
        extra_config={
            "chunk_size": chunk_size,
            "max_action_dim": action_dim,
            "prefix_len": args.prefix_len,
        },
        trt_settings={
            "disable_tf32": True,
            "use_fp32_acc": True,
            "use_explicit_typing": True,
            "decompose_attention": True,
        },
    )

    with apply_patches(PI05):
        engine_path, outputs, _ = compile_component(
            bundle,
            name="action",
            engine_dir=args.engine_dir,
        )

    artifact = build_action_artifact(engine_path, device_id=args.device_id)
    save_action_pte(artifact, action_inputs, args.pte_path)
    print(f"Saved fresh action engine under {engine_path}")
    print(f"Saved action-step EdgeLLMBackend program to {args.pte_path}")
    print("Output shapes:", [tuple(output.shape) for output in outputs])


if __name__ == "__main__":
    main()
