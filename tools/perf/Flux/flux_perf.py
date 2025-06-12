import argparse
import os
import sys
from time import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../examples/apps"))
from flux_demo import compile_model


def benchmark(pipe, prompt, inference_step, batch_size=1, iterations=1):

    start = time()
    for i in range(iterations):
        image = pipe(
            prompt,
            output_type="pil",
            num_inference_steps=inference_step,
            num_images_per_prompt=batch_size,
        ).images
    end = time()

    print(f"Batch Size: {batch_size}")
    print("Time Elapse for", iterations, "iterations:", end - start)
    print(
        "Average Latency Per Step:",
        (end - start) / inference_step / iterations / batch_size,
    )
    return image


def main(args):
    pipe, backbone, trt_gm = compile_model(args)
    for batch_size in range(1, args.max_batch_size + 1):
        benchmark(pipe, ["Test"], 20, batch_size=batch_size, iterations=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Flux quantization with different dtypes"
    )

    parser.add_argument(
        "--dtype",
        choices=["fp8", "int8", "fp16"],
        default="fp16",
        help="Select the data type to use (fp8 or int8 or fp16)",
    )
    parser.add_argument(
        "--low_vram_mode",
        action="store_true",
        help="Use low VRAM mode when you have a small GPU (<=32GB)",
    )
    parser.add_argument(
        "--dynamic_shapes",
        "-d",
        action="store_true",
        help="Use dynamic shapes",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="Maximum batch size to use",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug mode",
    )
    args = parser.parse_args()
    main(args)
