# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate one SDXL image with FP16 PyTorch or a Torch-TensorRT UNet.

Run from the repository root:
    python examples/apps/sdxl_demo.py --pytorch --output sdxl_pytorch.png
    python examples/apps/sdxl_demo.py --output sdxl_trt.png

Windows ARM64 requires CUDA-enabled ARM64 PyTorch and, for the TensorRT path,
a matching Torch-TensorRT-RTX build. ModelOpt and xformers are not required.
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--prompt", default="a majestic castle in the clouds")
    parser.add_argument("--negative_prompt", default="blurry, low quality")
    parser.add_argument("--output", type=Path, default=Path("sdxl.png"))
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pytorch", action="store_true", help="Run without importing Torch-TensorRT"
    )
    args = parser.parse_args()
    if any(size <= 0 or size % 8 for size in (args.height, args.width)):
        parser.error("--height and --width must be positive multiples of 8")
    if args.num_inference_steps <= 0:
        parser.error("--num_inference_steps must be positive")
    return args


def main(args):
    import torch
    from diffusers import StableDiffusionXLPipeline

    if not torch.cuda.is_available():
        raise RuntimeError(
            "SDXL requires a CUDA-enabled PyTorch installation and NVIDIA GPU. "
            "On Windows ARM64, use native ARM64 CUDA PyTorch."
        )

    print(f"Loading {args.model} in FP16")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        add_watermarker=False,
    ).to("cuda:0")
    pipe.unet.eval()
    pipe.vae.enable_tiling()

    if not args.pytorch:
        import torch_tensorrt

        pipe.unet = torch_tensorrt.MutableTorchTensorRTModule(
            pipe.unet,
            immutable_weights=False,
            min_block_size=1,
            use_fp32_acc=True,
            truncate_double=True,
            offload_module_to_cpu=True,
        )
        print("The first generation will compile the UNet with TensorRT")

    with torch.no_grad():
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device="cuda:0").manual_seed(args.seed),
        ).images[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved {image.width}x{image.height} image to {args.output.resolve()}")


if __name__ == "__main__":
    main(parse_args())
