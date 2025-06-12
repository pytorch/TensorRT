# %%
# Import the following libraries
# -----------------------------
# Load the ModelOpt-modified model architecture and weights using Huggingface APIs
# Add argument parsing for dtype selection
import argparse
import gc
import re

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import torch
import torch_tensorrt
from diffusers import FluxPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from modelopt.core.torch.quantization.config import NVFP4_FP8_MHA_CONFIG
from modelopt.torch.quantization.utils import export_torch_mode
from torch.export._trace import _export
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser(
    description="Run Flux quantization with different dtypes"
)
parser.add_argument(
    "--path",
    type="string",
    required=True,
    help="ep path",
)

args = parser.parse_args()


def generate_image(pipe, prompt, image_name):
    seed = 42
    image = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=20,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    image.save(f"{image_name}.png")
    print(f"Image generated using {image_name} model saved as {image_name}.png")


def benchmark(prompt, inference_step, batch_size=1, iterations=1):
    from time import time

    print(f"Benchmark TRT Module Latency started with {batch_size=} {iterations=}")
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


loaded_trt_module = torch.export.load(path)

DEVICE = "cuda:0"
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
)

config = pipe.transformer.config
pipe.transformer = loaded_trt_module
pipe.transformer.config = config

generate_image(pipe, ["beach and kids"], "beach_kids")


print(f"Benchmark TRT Module Latency at ({args.dtype=}) started")
for batch_size in range(1, 9):
    benchmark(["Test"], 20, batch_size=batch_size, iterations=3)
print(f"Benchmark TRT Module Latency at ({args.dtype=}) ended")
