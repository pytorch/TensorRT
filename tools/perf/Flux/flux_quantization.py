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
from modelopt.torch.quantization.utils import export_torch_mode
from torch.export._trace import _export
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser(
    description="Run Flux quantization with different dtypes"
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="debug mode",
)
parser.add_argument(
    "--dtype",
    choices=["fp8", "int8", "fp4", "fp16", "bf16", "fp32"],
    default="fp8",
    help="Quantization data type to use (fp8 or int8 or fp4 or fp16 or bf16 or fp32)",
)

parser.add_argument(
    "--sdpa",
    action="store_true",
    default=False,
    help="Register SDPA operator",
)

parser.add_argument(
    "--strong-typing",
    action="store_true",
    help="string type flag",
)

args = parser.parse_args()
if args.sdpa:
    import register_sdpa

dtype = torch.float16
ptq_config = None
use_explicit_typing = args.strong_typing
enabled_precisions = [
    torch.float32,
]

# Update enabled precisions based on dtype argument
if args.dtype == "fp8":
    (
        enabled_precisions.extend([torch.float8_e4m3fn, torch.float16])
        if not use_explicit_typing
        else None
    )
    ptq_config = mtq.FP8_DEFAULT_CFG
elif args.dtype == "int8":  # int8
    (
        enabled_precisions.extend([torch.int8, torch.float16])
        if not use_explicit_typing
        else None
    )
    ptq_config = mtq.INT8_DEFAULT_CFG
elif args.dtype == "fp4":
    ptq_config = mtq.NVFP4_DEFAULT_CFG
    use_explicit_typing = True
elif args.dtype == "fp16":
    enabled_precisions.append(torch.float16) if not use_explicit_typing else None
elif args.dtype == "bf16":
    dtype = torch.bfloat16
    (
        enabled_precisions.extend([torch.bfloat16, torch.float16])
        if not use_explicit_typing
        else None
    )
elif args.dtype == "fp32":
    dtype = torch.float32
else:
    raise ValueError(f"Invalid dtype: {args.dtype}")
print(f"\nUsing {args.dtype} quantization with {args=}")
# %%
DEVICE = "cuda:0"
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=dtype,
)

total_params = sum(p.numel() for p in pipe.transformer.parameters())
print(f"\n Total number of parameters: {total_params/1000/1000/1000}B")
if dtype in (torch.float16, torch.bfloat16):
    total_size = total_params * 2 / 1024 / 1024 / 1024
    print(f"\n Total size: {total_size}GB")
elif dtype == torch.float32:
    total_size = total_params * 4 / 1024 / 1024 / 1024
    print(f"\n Total size: {total_size}GB")

if args.debug:
    pipe.transformer = FluxTransformer2DModel(
        num_layers=1, num_single_layers=1, guidance_embeds=True
    )

pipe.to(DEVICE).to(dtype)
# Store the config and transformer backbone
config = pipe.transformer.config
# global backbone
backbone = pipe.transformer
backbone.eval()


def filter_func(name):
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|pos_embed|time_text_embed|context_embedder|norm_out|x_embedder).*"
    )
    return pattern.match(name) is not None


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


# %%
# Quantization


def do_calibrate(
    pipe,
    prompt: str,
) -> None:
    """
    Run calibration steps on the pipeline using the given prompts.
    """
    image = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=20,
        generator=torch.Generator("cuda").manual_seed(0),
    ).images[0]


def forward_loop(mod):
    # Switch the pipeline's backbone, run calibration
    pipe.transformer = mod
    do_calibrate(
        pipe=pipe,
        prompt="test",
    )


if ptq_config is not None:
    backbone = mtq.quantize(backbone, ptq_config, forward_loop)
    mtq.disable_quantizer(backbone, filter_func)
else:
    print("No quantization config provided, skipping quantization")

batch_size = 2
BATCH = torch.export.Dim("batch", min=1, max=8)
SEQ_LEN = torch.export.Dim("seq_len", min=1, max=512)
# This particular min, max values for img_id input are recommended by torch dynamo during the export of the model.
# To see this recommendation, you can try exporting using min=1, max=4096
IMG_ID = torch.export.Dim("img_id", min=3586, max=4096)
dynamic_shapes = {
    "hidden_states": {0: BATCH},
    "encoder_hidden_states": {0: BATCH, 1: SEQ_LEN},
    "pooled_projections": {0: BATCH},
    "timestep": {0: BATCH},
    "txt_ids": {0: SEQ_LEN},
    "img_ids": {0: IMG_ID},
    "guidance": {0: BATCH},
    "joint_attention_kwargs": {},
    "return_dict": None,
}
# The guidance factor is of type torch.float32
dummy_inputs = {
    "hidden_states": torch.randn((batch_size, 4096, 64), dtype=dtype).to(DEVICE),
    "encoder_hidden_states": torch.randn((batch_size, 512, 4096), dtype=dtype).to(
        DEVICE
    ),
    "pooled_projections": torch.randn((batch_size, 768), dtype=dtype).to(DEVICE),
    "timestep": torch.tensor([1.0] * batch_size, dtype=dtype).to(DEVICE),
    "txt_ids": torch.randn((512, 3), dtype=dtype).to(DEVICE),
    "img_ids": torch.randn((4096, 3), dtype=dtype).to(DEVICE),
    "guidance": torch.tensor([1.0] * batch_size, dtype=torch.float32).to(DEVICE),
    "joint_attention_kwargs": {},
    "return_dict": False,
}


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
gc.collect()
# This will create an exported program which is going to be compiled with Torch-TensorRT
with export_torch_mode():
    ep = _export(
        backbone,
        args=(),
        kwargs=dummy_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
        allow_complex_guards_as_runtime_asserts=True,
    )

peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
print(f"Peak memory allocated during torch-export: {peak_memory=}GB {peak_reserved=}GB")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
gc.collect()

with torch_tensorrt.logging.debug():
    trt_gm = torch_tensorrt.dynamo.compile(
        ep,
        inputs=dummy_inputs,
        enabled_precisions=enabled_precisions,
        use_explicit_typing=use_explicit_typing,
        truncate_double=True,
        min_block_size=1,
        debug=args.debug,
        immutable_weights=True,
        offload_module_to_cpu=True,
    )

peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
print(
    f"Peak memory allocated during torch dynamo compilation: {peak_memory=}GB {peak_reserved=}GB"
)

del ep
pipe.transformer = trt_gm
pipe.transformer.config = config

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
gc.collect()

# %%

trt_gm.device = torch.device(DEVICE)
# Function which generates images from the flux pipeline
generate_image(pipe, ["A golden retriever"], "dog_code2")

peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
print(f"Peak memory allocated during inference: {peak_memory=}GB {peak_reserved=}GB")

if not args.debug:
    print(f"Benchmark TRT Module Latency at ({args.dtype}) started")
    for batch_size in range(1, 3):
        benchmark(["Test"], 20, batch_size=batch_size, iterations=3)
    print(f"Benchmark TRT Module Latency at ({args.dtype}) ended")

# For this dummy model, the fp16 engine size is around 1GB, fp32 engine size is around 2GB
