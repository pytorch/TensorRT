"""
Single-GPU Llama inference with Torch-TensorRT (C++ runtime).

Usage
-----
  uv run python tools/llm/llama_single_gpu.py

Optional args:
  --model   meta-llama/Llama-3.2-1B-Instruct  (default)
  --prompt  "Your prompt here"
  --precision FP16|BF16|FP32
  --num_tokens 64
  --debug
"""

import argparse
import logging
from contextlib import nullcontext

import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate

DEVICE = torch.device("cuda:0")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_model(args):
    logger.info(f"Loading {args.model} ...")
    with torch.no_grad():
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                use_cache=False,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
            )
            .eval()
            .to(DEVICE)
        )

    logger.info("Model loaded.")
    return model


def compile_torchtrt(model, args):
    with torch_tensorrt.logging.debug() if args.debug else nullcontext():
        trt_model = torch.compile(
            model,
            backend="torch_tensorrt",
            dynamic=True,
            options={
                # "use_fp32_acc": True,
                "device": DEVICE,
                "disable_tf32": True,
                "use_python_runtime": False,
                "debug": args.debug,
                "min_block_size": 1,
                "assume_dynamic_shape_support": True,
                "use_distributed_trace": True,
            },
        )
    return trt_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-GPU Llama inference with Torch-TensorRT (C++ runtime)"
    )
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.2-1B-Instruct", help="HF model name"
    )
    parser.add_argument(
        "--prompt", default="What is tensor parallelism?", help="Input prompt"
    )
    parser.add_argument(
        "--precision",
        default="FP16",
        choices=["FP16", "BF16", "FP32"],
        help="Model precision",
    )
    parser.add_argument("--num_tokens", type=int, default=64)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with torch.inference_mode():
        model = get_model(args)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(DEVICE)
        max_len = input_ids.shape[1] + args.num_tokens

        logger.info("Running uncompiled PyTorch baseline ...")
        torch_tokens = generate(
            model, input_ids.clone(), max_len, tokenizer.eos_token_id
        )
        print("\n===== PyTorch (uncompiled) =====")
        print(tokenizer.decode(torch_tokens[0], skip_special_tokens=True))

        logger.info("Compiling with Torch-TensorRT (C++ runtime)...")
        trt_model = compile_torchtrt(model, args)

        logger.info("Warming up TRT model (triggering engine build)...")
        _position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = trt_model(input_ids.clone(), position_ids=_position_ids)
        logger.info("Compilation done. Starting TRT inference...")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            trt_tokens = generate(
                trt_model,
                input_ids.clone(),
                max_len,
                tokenizer.eos_token_id,
                dynamic_seqlen_range=(1, max_len),
            )
        print("\n===== TensorRT (C++ runtime) =====")
        print(tokenizer.decode(trt_tokens[0], skip_special_tokens=True))

    del trt_model
    del model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    logger.info("Done.")
