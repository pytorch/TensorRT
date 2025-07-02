import argparse
import copy
import os
import sys
from contextlib import nullcontext

import requests

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import torch
import torch_tensorrt
from PIL import Image
from transformers import AutoModel, AutoProcessor
from utils import (
    export_llm,
    generate_mm,
    generate_mm_with_static_cache,
    recordStats,
    time_generate_mm,
)

# Register SDPA as a standalone operator. Converter and lowering pass are defined in register_sdpa.py
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# SDPA patch
import transformers.models.qwen2.modeling_qwen2 as mq
from register_sdpa import *

mq.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = mq.ALL_ATTENTION_FUNCTIONS["sdpa"]

DEVICE = torch.device("cuda:0")


def load_model(model_name, device, torch_dtype):
    """Load the specified VLM model and its processor."""
    if model_name == "eagle2":
        return load_eagle2(device, torch_dtype)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def load_eagle2(device, torch_dtype):
    """Load Eagle2 model and processor."""
    model_id = "nvidia/Eagle2-2B"
    with torch.no_grad():
        model = (
            AutoModel.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch_dtype
            )
            .eval()
            .to(device)
        )
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    emb_layer = model.language_model.get_input_embeddings().to(torch_dtype).to(device)
    return model, processor, emb_layer


def get_model(args):
    if args.precision == "FP16":
        torch_dtype = torch.float16
    elif args.precision == "BF16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    model, processor, emb_layer = load_model(args.model, args.device, torch_dtype)

    return model, processor, emb_layer


def compile_lm_torchtrt(model, args):
    """Compile the language model with Torch-TensorRT."""

    if args.precision == "FP16":
        torch_dtype = torch.float16
    elif args.precision == "BF16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    example_embeds = torch.randn(
        1,
        2560,
        model.language_model.config.hidden_size,
        dtype=torch_dtype,
        device=args.device,
    )

    if args.model == "eagle2":
        return compile_eagle2_lm_with_trt_embed(
            model.language_model, example_embeds, args
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")


class LMNoCache(torch.nn.Module):
    """Wrapper for language model to use inputs_embeds without KV-cache."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, inputs_embeds, position_ids):
        out = self.lm(inputs_embeds=inputs_embeds, position_ids=position_ids)
        return out.logits if hasattr(out, "logits") else out


def compile_eagle2_lm_with_trt_embed(language_model, input_embeds, args):
    """Compile Eagle2 language model with Torch-TensorRT."""
    lm_wrap = LMNoCache(language_model).to(args.device).eval()
    max_seq_len = input_embeds.shape[1] + args.num_tokens

    S = torch.export.Dim("seq", min=1, max=max_seq_len)
    position_ids = torch.arange(input_embeds.shape[1]).unsqueeze(0).to(args.device)
    dyn_shapes = {"inputs_embeds": {1: S}, "position_ids": {1: S}}

    # Set precision specific flags
    use_fp32_acc = False
    use_explicit_typing = False

    if args.precision == "FP16":
        enabled_precisions = {torch.float32}
        use_fp32_acc = True
        use_explicit_typing = True
    elif args.precision == "BF16":
        enabled_precisions = {torch.bfloat16}
        use_fp32_acc = False
    else:
        enabled_precisions = {torch.float32}

    with torch.inference_mode():
        exported = torch.export.export(
            lm_wrap,
            (input_embeds, position_ids),
            dynamic_shapes=dyn_shapes,
            strict=False,
        )

    with torch_tensorrt.logging.debug() if args.debug else nullcontext():
        trt_mod = torch_tensorrt.dynamo.compile(
            exported,
            inputs=[input_embeds, position_ids],
            enabled_precisions=enabled_precisions,
            use_explicit_typing=use_explicit_typing,
            use_fp32_acc=use_fp32_acc,
            device=args.device,
            disable_tf32=True,
            use_python_runtime=True,
            debug=args.debug,
            offload_module_to_cpu=True,
            min_block_size=args.min_block_size,
        )

    return trt_mod


def print_outputs(backend_name, gen_tokens, tokenizer):
    print(f"========= {backend_name} =========")
    print(
        f"{backend_name} model generated text: ",
        tokenizer.decode(gen_tokens[0], skip_special_tokens=True),
    )
    print("===================================")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run inference on a model with random input values"
    )
    arg_parser.add_argument(
        "--model", type=str, default="eagle2", help="Name of VLM model"
    )
    arg_parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="Name of LLM model tokenizer",
    )
    arg_parser.add_argument(
        "--prompt", type=str, default="Describe this image.", help="Prompt"
    )
    arg_parser.add_argument(
        "--precision",
        type=str,
        default="FP16",
        help="Precision to use in the model. Options: FP16, BF16, FP32",
    )
    arg_parser.add_argument(
        "--iterations", type=int, default=5, help="no. of iterations to run"
    )
    arg_parser.add_argument(
        "--min_block_size", type=int, default=1, help="no. of iterations to run"
    )
    arg_parser.add_argument(
        "--num_tokens",
        type=int,
        default=128,
        help="no. of output tokens to be generated",
    )
    arg_parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size used for benchmarking"
    )
    arg_parser.add_argument(
        "--isl",
        type=int,
        default=2048,
        help="Input sequence length used for benchmarking",
    )
    arg_parser.add_argument(
        "--enable_pytorch_run",
        action="store_true",
        help="Enable pytorch run (default: False)",
    )
    arg_parser.add_argument(
        "--cache",
        type=str,
        default="",
        help="Type of KV cache to use. Options: static_v1",
    )
    arg_parser.add_argument(
        "--debug", action="store_true", help="Enable debug (default: False)"
    )
    arg_parser.add_argument(
        "--benchmark", action="store_true", help="Enable benchmark (default: False)"
    )
    arg_parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run on"
    )

    args = arg_parser.parse_args()
    with torch.inference_mode():
        model, processor, emb_layer = get_model(args)

        url = "https://cdn.pixabay.com/photo/2019/08/08/23/33/car-4393990_1280.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        # Prepare input for benchmarking or evaluation
        if args.benchmark:
            prompt_len = args.isl - 1792 - 26
            prompt = " ".join(["token"] * max(prompt_len, 0))
        else:
            prompt = args.prompt

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # Preprocess vision information

        txt = [
            processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        ]
        img_in, vid_in = processor.process_vision_info(messages)
        inputs = processor(
            text=txt, images=img_in, videos=vid_in, return_tensors="pt", padding=True
        ).to(args.device)

        MAX_OUTPUT_SEQ_LENGTH = inputs["input_ids"].shape[1] + args.num_tokens
        # Pyt
        pyt_gen_tokens = None
        pyt_timings = None
        pyt_stats = None

        if args.enable_pytorch_run:
            pyt_gen_tokens = generate_mm(
                model,
                inputs["pixel_values"],
                inputs["input_ids"],
                MAX_OUTPUT_SEQ_LENGTH,
                processor.tokenizer.eos_token_id,
                emb_layer,
            )
            if args.benchmark:
                pyt_timings = time_generate_mm(
                    generate_mm,
                    model,
                    inputs["pixel_values"].clone(),
                    inputs["input_ids"].clone(),
                    MAX_OUTPUT_SEQ_LENGTH,
                    processor.tokenizer.eos_token_id,
                    emb_layer,
                    iterations=args.iterations,
                )
                pyt_stats = recordStats(
                    "PyTorch",
                    pyt_timings,
                    args.precision,
                    batch_size=args.batch_size,
                    compile_time_s=None,
                )

        if args.cache == "static_v1":
            # This import is required to register static v1 KV cache transformations as lowering passes
            import static_cache_v1

        # Compile the language model with Torch-TensorRT
        trt_lm = compile_lm_torchtrt(model, args)
        trt_model = copy.deepcopy(model)
        trt_model.language_model = trt_lm

        emb_layer = emb_layer.to(args.device)

        if args.cache == "static_v1":
            trt_gen_tokens = generate_mm_with_static_cache(
                trt_model,
                inputs["pixel_values"],
                inputs["input_ids"],
                MAX_OUTPUT_SEQ_LENGTH,
                processor.tokenizer.eos_token_id,
                emb_layer,
                args.device,
            )

            if args.benchmark:
                trt_timings = time_generate_mm(
                    generate_mm_with_static_cache,
                    trt_model,
                    inputs["pixel_values"].clone(),
                    inputs["input_ids"].clone(),
                    MAX_OUTPUT_SEQ_LENGTH,
                    processor.tokenizer.eos_token_id,
                    emb_layer,
                    iterations=args.iterations,
                    device=args.device,
                )
        else:
            trt_gen_tokens = generate_mm(
                trt_model,
                inputs["pixel_values"],
                inputs["input_ids"],
                MAX_OUTPUT_SEQ_LENGTH,
                processor.tokenizer.eos_token_id,
                emb_layer,
            )
            if args.benchmark:
                trt_timings = time_generate_mm(
                    generate_mm,
                    trt_model,
                    inputs["pixel_values"].clone(),
                    inputs["input_ids"].clone(),
                    MAX_OUTPUT_SEQ_LENGTH,
                    processor.tokenizer.eos_token_id,
                    emb_layer,
                    iterations=args.iterations,
                )

        if args.benchmark:
            trt_stats = recordStats(
                "TensorRT",
                trt_timings,
                args.precision,
                batch_size=args.batch_size,
                compile_time_s=None,
            )

        if not args.benchmark:
            if args.enable_pytorch_run:
                print_outputs("PyTorch", pyt_gen_tokens, processor.tokenizer)

            print_outputs("TensorRT", trt_gen_tokens, processor.tokenizer)

            if args.enable_pytorch_run:
                print(
                    f"PyTorch and TensorRT outputs match: {torch.equal(pyt_gen_tokens, trt_gen_tokens)}"
                )

        if args.benchmark:
            if args.enable_pytorch_run:
                print("=========PyTorch PERFORMANCE============ \n")
                print(pyt_stats)
            print("===================== \n")
            print("=========TensorRT PERFORMANCE============ \n")
            print(trt_stats)
