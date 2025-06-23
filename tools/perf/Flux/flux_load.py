import argparse
import os
import sys
from time import time

import torch
from diffusers import FluxPipeline

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../examples/apps"))
from flux_perf import benchmark


def main(args):
    # Load the TRT module
    if args.load_full_path:
        trt_ep_path = args.load_full_path
    else:
        trt_ep_path = os.path.join(os.path.dirname(__file__), "flux_trt.ep")

    if not os.path.exists(trt_ep_path):
        raise FileNotFoundError(f"TRT module not found at {trt_ep_path}")

    try:
        loaded_trt_module = torch.export.load(trt_ep_path)
        print(f"Model loaded from {trt_ep_path}")
    except Exception as e:
        print(f"Failed to load TRT module: {e}")
        return

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    DEVICE = "cuda:0"

    # Load the Flux pipeline
    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16,
        ).to(DEVICE)
    except Exception as e:
        print(f"Failed to load Flux pipeline: {e}")
        return

    # Replace transformer with TRT module
    config = pipe.transformer.config
    pipe.transformer = loaded_trt_module
    pipe.transformer.config = config

    # Generate images
    prompt = "Beach and kids"
    print(f"Generating images with prompt: '{prompt}'")

    start_time = time()
    images = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=args.batch_size,
    )
    end_time = time()

    print(
        f"Generated {len(images.images)} images in {end_time - start_time:.2f} seconds"
    )

    images[0].save("flux_trt.png")

    if args.benchmark:
        benchmark(pipe, prompt, args.num_inference_steps, batch_size, iterations=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and run Flux model with TensorRT acceleration"
    )
    parser.add_argument(
        "--load_full_path",
        "-l",
        help="Load the model from a full path",
    )
    parser.add_argument(
        "--num_inference_steps",
        "-n",
        type=int,
        default=20,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark the model",
    )
    args = parser.parse_args()
    main(args)
