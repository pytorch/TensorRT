"""
.. _run_vlm:

Benchmarking VLM Inference with Torch-TensorRT
==========================================================

This script provides a framework for benchmarking the performance of Visual-Language
Models (VLMs). It optimizes the two most computationally intensive components of a
VLM—the language model and the vision model (image feature extraction)—using
the Torch-TensorRT dynamo backend.

Key Features:
- **Component-wise Optimization**: Compiles both the language and vision models
  separately with Torch-TensorRT to accelerate inference.
- **Performance Benchmarking**: Runs the model for multiple iterations to
  measure and compare inference latency against the PyTorch baseline.
- **Output Verification**: Checks for token-level consistency between the optimized
  TensorRT model and the original PyTorch model to ensure correctness.
- **KV Cache Testing**: Includes options to test inference with and without
  KV caching to evaluate its impact on performance.

This tool mirrors the style and structure of `run_llm.py`, providing a clear
workflow for VLM optimization and analysis.

Dependencies:
- For Qwen VLM models: pip install qwen-vl-utils
- For Eagle2 models: pip install flash-attn --no-build-isolation -v
"""

import argparse
import copy
import os
import sys
from contextlib import nullcontext
from typing import Tuple

import requests
import torch
import torch_tensorrt

# we "monkey-patch" the global attention function map for Qwen2.
# This ensures that any part of the code (including torch.export) requesting
# "flash_attention_2" will receive the "sdpa" implementation instead.
# This patch is global for the script's execution context.
import transformers.models.qwen2.modeling_qwen2 as mq
from PIL import Image
from torchtrt_ext import register_sdpa
from transformers import AutoConfig, AutoModel, AutoProcessor
from utils import (
    export_llm,
    generate_mm,
    generate_mm_qwen2_5_vl,
    generate_mm_qwen2_5_vl_with_static_cache,
    generate_mm_with_static_cache,
    record_stats,
)

# --- WORKAROUND FOR EAGLE2 SDPA COMPILATION ---
# Eagle2's language model (Qwen2) implicitly defaults to "flash_attention_2"
# due to settings in its remote code and config.json. This prevents direct
# compilation with SDPA. To work around this without modifying the library,


mq.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = mq.ALL_ATTENTION_FUNCTIONS["sdpa"]
# --- END WORKAROUND ---

# --- Model-specific constants for benchmark and compilation ---
# Centralizing these values improves readability and maintainability.
MODEL_CONSTANTS = {
    "nvidia/Eagle2-2B": {
        "EXAMPLE_SEQLEN": 2560,  # A fixed sequence length for creating the example tensor for TRT compilation.
        "IMAGE_TOKENS": 1792,  # Number of special tokens used to represent the image patch embeddings in the input sequence for Eagle2-2B VLM.
        "PROMPT_WRAPPER_TOKENS": 26,  # The number of special/processing tokens added by the processor's chat template in benchmark mode.
    },
    "Qwen/Qwen2.5-VL-3B-Instruct": {
        "EXAMPLE_SEQLEN": 2560,
        "IMAGE_TOKENS": 1426,
        "PROMPT_WRAPPER_TOKENS": 21,
    },
}
# --- END Model-specific constants ---

# -----------------------------------------------------------------------------#
# Model loading helpers
# -----------------------------------------------------------------------------#


def _load_eagle2(device: torch.device, torch_dtype: torch.dtype):
    """
    Load nvidia/Eagle2-2B model and processor, ensuring the language model uses SDPA.

    Returns
    -------
    tuple[torch.nn.Module, transformers.AutoProcessor, torch.nn.Embedding]
        The model, its processor and the language-model input embedding layer.
    """
    model_id = "nvidia/Eagle2-2B"
    try:
        with torch.no_grad():
            model = (
                AutoModel.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    # attn_implementation="sdpa" is ignored due to the model's remote code.
                )
                .eval()
                .to(device)
            )
    except ImportError as e:
        if "flash_attn" in str(e):
            raise ImportError(
                "FlashAttention2 is required for Eagle2 models but not installed. "
                "Please install it using: pip install flash-attn --no-build-isolation -v"
            ) from e
        raise

    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    emb_layer = model.language_model.get_input_embeddings().to(torch_dtype).to(device)
    return model, processor, emb_layer


def _load_qwen2_5_vl(device, torch_dtype: torch.dtype):
    """
    Load Qwen2.5-VL model and processor.
    """
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map=device
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    emb_layer = model.model.get_input_embeddings().to(torch_dtype).to(device)
    return model, processor, emb_layer


def load_model(
    model_name: str, device: torch.device, torch_dtype: torch.dtype
) -> Tuple[torch.nn.Module, AutoProcessor, torch.nn.Embedding]:
    """Dispatch helper for supported VLMs."""
    if model_name == "nvidia/Eagle2-2B":
        return _load_eagle2(device, torch_dtype)
    elif model_name == "Qwen/Qwen2.5-VL-3B-Instruct":
        return _load_qwen2_5_vl(device, torch_dtype)
    msg = f"Unsupported model: '{model_name}'. Supported models are: ['nvidia/Eagle2-2B', 'Qwen/Qwen2.5-VL-3B-Instruct']"
    raise ValueError(msg)


# -----------------------------------------------------------------------------#
# Input loading helpers
# -----------------------------------------------------------------------------#


def load_inputs(args: argparse.Namespace, processor, device: torch.device):
    """
    Loads and constructs the input dictionary for the specified VLM model.
    """
    # Load image from local path if provided, otherwise use default URL
    if args.image_path is not None:
        # Use local image file
        image = Image.open(args.image_path)
    else:
        # Use default URL image
        url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

    if args.benchmark:
        model_constants = MODEL_CONSTANTS[args.model]
        image_tokens = model_constants["IMAGE_TOKENS"]
        wrapper_tokens = model_constants["PROMPT_WRAPPER_TOKENS"]

        prompt_len = args.isl - image_tokens - wrapper_tokens
        prompt_txt = " ".join(["token"] * max(prompt_len, 0))
    else:
        prompt_txt = args.prompt or "Describe this image."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_txt},
            ],
        }
    ]

    text = [
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    ]

    # --- Model-specific vision processing ---
    if args.model == "Qwen/Qwen2.5-VL-3B-Instruct":
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError(
                "The 'qwen-vl-utils' package is required for Qwen VLM models. "
                "Please install it using: pip install qwen-vl-utils"
            )

        image_inputs, video_inputs = process_vision_info(messages)
    else:  # eagle2
        image_inputs, video_inputs = processor.process_vision_info(messages)

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    return inputs


# -----------------------------------------------------------------------------#
# Torch-TensorRT compilation helpers
# -----------------------------------------------------------------------------#


class _LMNoCache(torch.nn.Module):
    """
    Thin wrapper that exposes a language model via ``inputs_embeds`` without KV-cache.
    """

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, inputs_embeds, position_ids):
        out = self.lm(inputs_embeds=inputs_embeds, position_ids=position_ids)
        return (
            out.logits
            if hasattr(out, "logits")
            else out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        )


def _compile_lm(
    language_model: torch.nn.Module,
    input_embeds: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    """
    Compile the language model component of a VLM with Torch-TensorRT
    """
    lm_wrap = _LMNoCache(language_model).to(device).eval()
    max_seq_len = input_embeds.shape[1] + args.num_tokens

    seq_len = torch.export.Dim("seq", min=1, max=max_seq_len)
    position_ids = torch.arange(input_embeds.shape[1]).unsqueeze(0).to(device)

    dyn_shapes = {"inputs_embeds": {1: seq_len}, "position_ids": {1: seq_len}}

    use_fp32_acc = False
    use_explicit_typing = False
    if args.precision == "FP16":
        enabled_precisions = {torch.float32}
        use_fp32_acc = True
        use_explicit_typing = True
    else:  # FP32
        enabled_precisions = {torch.float32}

    exported_program = export_llm(
        lm_wrap, input_embeds, min_seq_len=1, max_seq_len=2560
    )

    with torch_tensorrt.logging.debug() if args.debug else nullcontext():
        trt_mod = torch_tensorrt.dynamo.compile(
            exported_program,
            inputs=[input_embeds, position_ids],
            enabled_precisions=enabled_precisions,
            use_explicit_typing=use_explicit_typing,
            use_fp32_acc=use_fp32_acc,
            device=device,
            disable_tf32=args.disable_tf32,
            use_python_runtime=args.use_python_runtime,
            offload_module_to_cpu=args.offload_module_to_cpu,
            min_block_size=args.min_block_size,
        )
    return trt_mod


def compile_lm_torchtrt(
    model: torch.nn.Module, args: argparse.Namespace, device: torch.device
) -> torch.nn.Module:
    """
    Compiles the Language Model (LLM) component of the VLM using Torch-TensorRT.
    """
    torch_dtype = {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }.get(args.precision, torch.float32)

    lm_model = (
        model.model
        if args.model == "Qwen/Qwen2.5-VL-3B-Instruct"
        else model.language_model
    )

    model_constants = MODEL_CONSTANTS.get(
        args.model, {"EXAMPLE_SEQLEN": args.num_tokens}
    )
    example_seq_len = model_constants["EXAMPLE_SEQLEN"]

    example_embeds = torch.randn(
        args.batch_size,
        example_seq_len,
        lm_model.config.hidden_size,
        dtype=torch_dtype,
        device=device,
    )

    # All supported models use the same compilation helper.
    if args.model in ["nvidia/Eagle2-2B", "Qwen/Qwen2.5-VL-3B-Instruct"]:
        return _compile_lm(lm_model, example_embeds, args, device)
    else:
        msg = f"Unsupported model: '{args.model}'. Supported models are: ['nvidia/Eagle2-2B', 'Qwen/Qwen2.5-VL-3B-Instruct']"
        raise ValueError(msg)


def _compile_eagle2_vision(
    vision_model: torch.nn.Module,
    example_pixel_values: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    """
    Compile Eagle2 vision model with Torch-TensorRT.
    """
    # Set precision-specific flags
    use_fp32_acc = False
    use_explicit_typing = False
    if args.precision == "FP16":
        enabled_precisions = {torch.float32}
        use_fp32_acc = True
        use_explicit_typing = True
    elif args.precision == "BF16":
        enabled_precisions = {torch.bfloat16}
    else:  # FP32
        enabled_precisions = {torch.float32}

    with torch.inference_mode():
        exported_program = torch.export.export(
            vision_model,
            (example_pixel_values,),
            strict=False,
        )

    with torch_tensorrt.logging.debug() if args.debug else nullcontext():
        trt_mod = torch_tensorrt.dynamo.compile(
            exported_program,
            inputs=[example_pixel_values],
            enabled_precisions=enabled_precisions,
            use_explicit_typing=use_explicit_typing,
            use_fp32_acc=use_fp32_acc,
            device=device,
            disable_tf32=args.disable_tf32,
            use_python_runtime=args.use_python_runtime,
            offload_module_to_cpu=args.offload_module_to_cpu,
            min_block_size=args.min_block_size,
        )
    return trt_mod


def compile_vision_torchtrt(
    model: torch.nn.Module,
    args: argparse.Namespace,
    example_pixel_values: torch.Tensor,
    device: torch.device,
) -> torch.nn.Module:
    """
    Dispatcher function for vision model compilation.
    """
    if args.model == "nvidia/Eagle2-2B":
        return _compile_eagle2_vision(
            model.vision_model, example_pixel_values, args, device
        )
    elif args.model == "Qwen/Qwen2.5-VL-3B-Instruct":
        # TODO: Vision model compilation for Qwen2.5-VL is currently skipped.
        # The model's `get_window_index` method uses dynamic Python list operations
        # (e.g., .tolist(), .extend()) to process variable-sized image grids for
        # windowed attention. These operations are incompatible with torch.export's
        # static graph tracing, preventing successful compilation.
        return model.visual
    else:
        raise ValueError(f"Unsupported model: {args.model}")


# -----------------------------------------------------------------------------#
# Utility helpers
# -----------------------------------------------------------------------------#


def print_outputs(backend_name: str, gen_tokens: torch.Tensor, tokenizer):
    """Pretty-print generated text for comparison."""
    print(f"========= {backend_name} =========")
    print(
        f"{backend_name} model generated text: ",
        tokenizer.decode(gen_tokens[0], skip_special_tokens=True),
    )
    print("===================================")


# -----------------------------------------------------------------------------#
# Main driver
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VLM inference (PyTorch & TensorRT back-ends)"
    )
    parser.add_argument(
        "--model",
        default="nvidia/Eagle2-2B",
        choices=["nvidia/Eagle2-2B", "Qwen/Qwen2.5-VL-3B-Instruct"],
        help="VLM model name",
    )
    parser.add_argument("--prompt", default="Describe this image.", help="Prompt text")
    parser.add_argument(
        "--precision",
        default="FP16",
        choices=["FP16", "FP32"],
        help="Computation precision",
    )
    parser.add_argument("--iterations", type=int, default=5, help="# iterations")
    parser.add_argument("--min_block_size", type=int, default=1, help="Min block size")
    parser.add_argument("--num_tokens", type=int, default=128, help="# new tokens")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--isl", type=int, default=2048, help="Input seq length")
    parser.add_argument(
        "--enable_pytorch_run",
        action="store_true",
        help="Run the PyTorch baseline as well",
    )
    parser.add_argument(
        "--cache",
        default="",
        choices=["", "static_v1"],
        help="KV-cache variant to use",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable Torch-TensorRT debug logs"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Enable benchmarking mode"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to local image file. If not provided, uses default URL image.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (e.g., 'cuda:0', 'cuda:1')",
    )
    parser.add_argument(
        "--disable_tf32",
        action="store_false",
        default=True,
        help="Disable TF32 precision for TensorRT compilation (default: True)",
    )
    parser.add_argument(
        "--use_python_runtime",
        action="store_false",
        default=True,
        help="Use Python runtime for TensorRT compilation (default: True)",
    )
    parser.add_argument(
        "--offload_module_to_cpu",
        action="store_false",
        default=True,
        help="Offload module to CPU for TensorRT compilation (default: True)",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # -------------------------------------------------------------------------#
    # 1. Model / processor / embeddings
    # -------------------------------------------------------------------------#
    dtype = {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }.get(args.precision, torch.float32)

    model, processor, emb_layer = load_model(args.model, device, dtype)

    # -------------------------------------------------------------------------#
    # 2. Input construction (image + text prompt)
    # -------------------------------------------------------------------------#
    inputs = load_inputs(args, processor, device)

    max_output_len = inputs["input_ids"].shape[1] + args.num_tokens

    # -------------------------------------------------------------------------#
    # 3. Optional: PyTorch baseline
    # -------------------------------------------------------------------------#
    pyt_gen_tokens = pyt_timings = pyt_stats = None
    if args.enable_pytorch_run:
        # For benchmarking, we run the generation with timing enabled.
        # For regular runs, we run without timing for a single output.
        if args.benchmark:
            if args.model == "Qwen/Qwen2.5-VL-3B-Instruct":
                (
                    pyt_gen_tokens,
                    _,
                    overall_time,
                    _,
                    _,
                ) = generate_mm_qwen2_5_vl(
                    model,
                    inputs["pixel_values"],
                    inputs["input_ids"],
                    inputs["image_grid_thw"],
                    processor.tokenizer.eos_token_id,
                    emb_layer,
                    max_new_tokens=args.num_tokens,
                    with_timing=True,
                )
            else:  # eagle2
                (
                    pyt_gen_tokens,
                    _,
                    overall_time,
                    _,
                    _,
                ) = generate_mm(
                    model,
                    inputs["pixel_values"],
                    inputs["input_ids"],
                    processor.tokenizer.eos_token_id,
                    emb_layer,
                    max_new_tokens=args.num_tokens,
                    with_timing=True,
                )
            pyt_stats = record_stats(
                "PyTorch",
                [overall_time / 1000],  # time_generate returns seconds
                args.precision,
                batch_size=args.batch_size,
            )
        else:
            if args.model == "Qwen/Qwen2.5-VL-3B-Instruct":
                pyt_gen_tokens = generate_mm_qwen2_5_vl(
                    model,
                    inputs["pixel_values"],
                    inputs["input_ids"],
                    inputs["image_grid_thw"],
                    processor.tokenizer.eos_token_id,
                    emb_layer,
                    max_new_tokens=args.num_tokens,
                )
            else:  # eagle2
                pyt_gen_tokens = generate_mm(
                    model,
                    inputs["pixel_values"],
                    inputs["input_ids"],
                    processor.tokenizer.eos_token_id,
                    emb_layer,
                    max_new_tokens=args.num_tokens,
                )

    # -------------------------------------------------------------------------#
    # 4. Torch-TensorRT compile & run
    # -------------------------------------------------------------------------#

    trt_model = copy.deepcopy(model)
    # 4.1. Vision model compilation
    # --- Add vision model compilation --- #
    example_pixel_values = inputs["pixel_values"]
    trt_vision = compile_vision_torchtrt(model, args, example_pixel_values, device)
    if args.model == "Qwen/Qwen2.5-VL-3B-Instruct":
        trt_model.visual = trt_vision
    else:
        trt_model.vision_model = trt_vision

    # -------------------------------------------------------------------------#
    # 4.2. Language model compilation
    # -------------------------------------------------------------------------#
    # Register static cache lowering passes if requested
    # Cache is not applied to vision model.
    if args.cache == "static_v1":
        import static_cache_v1  # noqa: F401
    elif args.cache not in ("", None):
        raise ValueError(
            f"Cache mode '{args.cache}' is not supported. Only 'static_v1' is supported."
        )

    trt_lm = compile_lm_torchtrt(model, args, device)
    if args.model == "Qwen/Qwen2.5-VL-3B-Instruct":
        trt_model.model = trt_lm
    else:
        trt_model.language_model = trt_lm

    emb_layer = emb_layer.to(device)
    if args.model == "Qwen/Qwen2.5-VL-3B-Instruct":
        trt_model.lm_head = trt_model.lm_head.to(device)

    if args.cache == "static_v1":
        if args.model == "Qwen/Qwen2.5-VL-3B-Instruct":
            trt_generate = generate_mm_qwen2_5_vl_with_static_cache
        else:  # eagle2
            trt_generate = generate_mm_with_static_cache
    else:
        if args.model == "Qwen/Qwen2.5-VL-3B-Instruct":
            trt_generate = generate_mm_qwen2_5_vl
        else:  # eagle2
            trt_generate = generate_mm

    # Prepare args for generate function
    generate_args = {
        "model": trt_model,
        "pixel_values": inputs["pixel_values"],
        "input_ids": inputs["input_ids"],
        "eos_token_id": processor.tokenizer.eos_token_id,
        "emb_layer": emb_layer,
        "max_new_tokens": args.num_tokens,
    }
    if args.model == "Qwen/Qwen2.5-VL-3B-Instruct":
        generate_args["image_grid_thw"] = inputs["image_grid_thw"]

    if args.cache == "static_v1" or args.benchmark:
        generate_args["with_timing"] = True

    if args.cache == "static_v1":
        generate_args["device"] = device

    # Run TRT generation
    trt_output = trt_generate(**generate_args)

    # Unpack results
    if args.benchmark or args.cache == "static_v1":
        trt_gen_tokens, _, overall_time, _, _ = trt_output
        trt_stats = record_stats(
            "TensorRT",
            [overall_time / 1000],  # time is in ms, convert to s
            args.precision,
            batch_size=args.batch_size,
        )
    else:
        trt_gen_tokens = trt_output

    # -------------------------------------------------------------------------#
    # 5. Reporting
    # -------------------------------------------------------------------------#
    if not args.benchmark:
        if args.enable_pytorch_run:
            print_outputs("PyTorch", pyt_gen_tokens, processor.tokenizer)
        print_outputs("TensorRT", trt_gen_tokens, processor.tokenizer)

        if args.enable_pytorch_run:
            print(
                f"PyTorch and TensorRT outputs match: "
                f"{torch.equal(pyt_gen_tokens, trt_gen_tokens)}"
            )

    if args.benchmark:
        if args.enable_pytorch_run:
            print("========= PyTorch PERFORMANCE =========\n")
            print(pyt_stats)
        print("=====================\n")
        print("========= TensorRT PERFORMANCE =========\n")
        print(trt_stats)
