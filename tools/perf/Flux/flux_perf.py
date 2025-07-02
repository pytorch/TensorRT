import argparse
import os
import sys
from time import time

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../examples/apps"))
from flux_demo import compile_model


def benchmark(pipe, prompt, inference_step, batch_size=1, iterations=1):
    # warmup
    for i in range(10):
        start = time()
        images = pipe(
            prompt,
            output_type="pil",
            num_inference_steps=inference_step,
            num_images_per_prompt=batch_size,
        ).images
        print(
            f"Warmup {i} done in {time() - start} seconds, with {batch_size=} {inference_step=}, generated {len(images)} images"
        )

    # actual benchmark
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

    # run the perf tool
    from cuda import cudart

    cudart.cudaProfilerStart()
    image = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=inference_step,
        num_images_per_prompt=batch_size,
    ).images
    cudart.cudaProfilerStop()
    return


def main(args):
    print(f"Running flux_perfwith args: {args}")
    pipe, backbone, trt_gm = compile_model(args)

    for batch_size in range(1, args.max_batch_size + 1):
        benchmark(pipe, ["Test"], 20, batch_size=batch_size, iterations=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Flux quantization with different dtypes"
    )
    parser.add_argument(
        "--use_sdpa",
        action="store_true",
        help="Use sdpa",
        default=False,
    )
    parser.add_argument(
        "--dtype",
        choices=["fp4", "fp8", "int8", "fp16"],
        default="fp16",
        help="Select the data type to use (fp4 or fp8 or int8 or fp16)",
    )
    parser.add_argument(
        "--fp4_mha",
        action="store_true",
        help="Use NVFP4_FP8_MHA_CONFIG config instead of NVFP4_FP8_MHA_CONFIG",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug mode",
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
        "--use_dynamo",
        action="store_true",
        help="Use dynamo compile",
        default=False,
    )
    parser.add_argument("--max_batch_size", type=int, default=1)
    args = parser.parse_args()
    main(args)
