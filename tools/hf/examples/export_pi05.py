"""Export a pretrained PI0.5 checkpoint using preprocessed LeRobot inputs.

python examples/export_pi05.py --checkpoint /path/to/checkpoint \
    --inputs observations.pt --output /tmp/pi05 --steps 10

observations.pt is a tensor-only dictionary with images (NCHW camera list),
image_masks, tokens, token_mask, and noise. Resize/normalization/tokenization
must use the checkpoint's LeRobot processors before writing this file.
"""

import argparse
from pathlib import Path

import torch


def load_policy(checkpoint: str):
    from huggingface_hub import snapshot_download
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05 import PI05Policy
    from safetensors.torch import load_file

    directory = Path(checkpoint)
    if not directory.is_dir():
        directory = Path(
            snapshot_download(
                checkpoint, allow_patterns=["config.json", "model.safetensors"]
            )
        )
    weights = directory / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint has no pretrained weights: {weights}")
    config = PreTrainedConfig.from_pretrained(directory)
    if config.type != "pi05":
        raise ValueError("Expected a PI0.5 checkpoint")
    if (
        getattr(config, "use_proprioceptive_memory", False)
        or getattr(config, "rtc_config", None) is not None
    ):
        raise ValueError("This exporter currently supports base PI0.5 without MEM/RTC")
    policy = PI05Policy(config)
    state = policy._fix_pytorch_state_dict_keys(load_file(weights), config)
    state = {
        name if name.startswith("model.") else f"model.{name}": tensor
        for name, tensor in state.items()
    }
    # The two language heads are unused by vision, prefix embedding and action
    # execution. Safetensors may omit them because weights are tied.
    missing, unexpected = policy.load_state_dict(state, strict=False)
    required_missing = [
        name for name in missing if not name.endswith(".lm_head.weight")
    ]
    if required_missing or unexpected:
        raise ValueError(
            f"Checkpoint does not match PI0.5: missing={required_missing}, unexpected={unexpected}"
        )
    return policy.eval()


def main():
    from torch_tensorrt_edge_llm.pi05 import (
        export_pi05,
        prepare_pi05_sample,
        write_native_inputs,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    policy = load_policy(args.checkpoint)
    # Float32 establishes the initial native application contract. Optimized
    # precision profiles can be added after full-checkpoint numerical validation.
    core = policy.model.to(device=args.device, dtype=torch.float32).eval()
    core.paligemma_with_expert.precision = "float32"
    raw = torch.load(args.inputs, weights_only=True, map_location=args.device)
    sample = prepare_pi05_sample(
        core,
        raw["images"],
        raw["image_masks"],
        raw["tokens"],
        raw["token_mask"],
        raw["noise"],
    )
    output = Path(args.output)
    exported = export_pi05(
        core,
        sample,
        engine_dir=output / "engines",
        cameras=len(raw["images"]),
        num_steps=args.steps,
    )
    with torch.no_grad():
        reference = core.sample_actions(
            raw["images"],
            raw["image_masks"],
            raw["tokens"],
            raw["token_mask"],
            noise=raw["noise"].clone(),
            num_steps=args.steps,
        )
        result = exported.program.module()(*sample)
        torch.testing.assert_close(result, reference, rtol=2e-2, atol=2e-2)
    torch.save(result.cpu(), output / "reference.pt")
    print(exported.save(output))
    print(exported.save(output, output_format="executorch"))
    write_native_inputs(sample, output / "inputs")


if __name__ == "__main__":
    main()
