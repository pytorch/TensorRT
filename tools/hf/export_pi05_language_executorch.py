from __future__ import annotations

import argparse
from pathlib import Path

import torch
from exporters.compile import compile_component
from exporters.executorch import build_language_artifact, save_language_prefill_pte
from exporters.models.common.helpers import causal_lm_flat
from exporters.models.common.patches import language_decoder
from exporters.models.pi05.patches import PI05
from exporters.models.pi05.spec import Pi05Spec
from exporters.plugin.attention import ContextAttentionMaskType
from exporters.plugin.attn_patches import apply_patches
from exporters.plugin.plugin_utils import load_plugins_for_trt
from exporters.spec import ComponentBundle
from exporters.utils import force_hf_attention
from lerobot.policies.pi05 import PI05Policy

DEFAULT_CHECKPOINT = "lerobot/pi05_libero_base"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freshly compile PI0.5 language prefill and package an EdgeLLMBackend .pte"
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=Path("/tmp/pi05_language_executorch_fresh"),
    )
    parser.add_argument(
        "--pte-path",
        type=Path,
        default=Path("/tmp/pi05_language_edge.pte"),
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=968)
    parser.add_argument("--prefill-len", type=int, default=64)
    args = parser.parse_args()

    if not 1 <= args.prefill_len <= args.max_seq_len:
        parser.error("--prefill-len must be between 1 and --max-seq-len")

    device = torch.device("cuda", args.device_id)
    dtype = torch.float16
    load_plugins_for_trt()

    policy = PI05Policy.from_pretrained(args.checkpoint).eval()
    policy.model.to(device=device, dtype=dtype).eval()
    language = policy.model.paligemma_with_expert.paligemma.model.language_model
    force_hf_attention(language, "eager")

    hidden_size = int(language.config.hidden_size)
    inputs_embeds = torch.randn(
        1,
        args.prefill_len,
        hidden_size,
        device=device,
        dtype=dtype,
    )
    flat_inputs, metadata = causal_lm_flat(
        language,
        inputs_embeds,
        max_seq_len=args.max_seq_len,
        device=device,
        dtype=dtype,
        seq_len=args.prefill_len,
    )
    input_names = list(metadata["input_names"])
    input_specs = Pi05Spec().create_dynamic_shapes(
        input_names,
        flat_inputs,
        max_seq_len=args.max_seq_len,
    )
    bundle = ComponentBundle(
        module=language_decoder(language).eval(),
        trace_args=flat_inputs,
        save_args=flat_inputs,
        input_specs=input_specs,
        input_names=input_names,
        output_names=["logits", "lm_hidden_states", "prefix_k", "prefix_v"],
        context_attention_mask_type=int(ContextAttentionMaskType.PADDING),
        extra_config={"prefix_pad_mask_len": args.prefill_len},
        model_type="language",
        engine_file="language.engine",
        trt_settings={
            "disable_tf32": True,
            "use_fp32_acc": True,
            "use_explicit_typing": True,
            "decompose_attention": True,
            "assume_dynamic_shape_support": True,
        },
    )

    with apply_patches(PI05):
        engine_path, outputs, _ = compile_component(
            bundle,
            name="language",
            engine_dir=args.engine_dir,
        )

    artifact = build_language_artifact(engine_path, device_id=args.device_id)
    save_language_prefill_pte(artifact, flat_inputs, args.pte_path)
    print(f"Saved fresh language engine under {engine_path}")
    print(f"Saved EdgeLLMBackend program to {args.pte_path}")
    print("Output shapes:", [tuple(output.shape) for output in outputs])


if __name__ == "__main__":
    main()
