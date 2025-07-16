"""
.. _run_vlm:

Running VLM inference with Torch-TensorRT
==========================================================

This script mirrors the style and structure of *run_llm.py*, illustrating a
Torch-TensorRT (dynamo backend) workflow for Visual-Language Models (VLMs).
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
from PIL import Image
from torchtrt_ext import register_sdpa
from transformers import AutoModel, AutoProcessor
from utils import (
    generate_mm,
    generate_mm_with_static_cache,
    generate_mm_paligemma,
    generate_mm_paligemma_with_static_cache,
    record_stats,
    time_generate_mm,
    generate_mm_qwen2_5_vl,
)

# -----------------------------------------------------------------------------#
# Global configuration
# -----------------------------------------------------------------------------#
DEVICE = torch.device("cuda:1")

# Register SDPA as a standalone operator.  Converter & lowering pass are defined
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import transformers.models.qwen2.modeling_qwen2 as mq  # noqa: E402

mq.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = mq.ALL_ATTENTION_FUNCTIONS["sdpa"]

# -----------------------------------------------------------------------------#
# Model loading helpers
# -----------------------------------------------------------------------------#


def _load_eagle2(device: torch.device, torch_dtype: torch.dtype):
    """
    Load Eagle2 model and processor.

    Returns
    -------
    tuple[torch.nn.Module, transformers.AutoProcessor, torch.nn.Embedding]
        The model, its processor and the language-model input embedding layer.
    """
    model_id = "nvidia/Eagle2-2B"
    with torch.no_grad():
        model = (
            AutoModel.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch_dtype, attn_implementation="sdpa"
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


def _load_paligemma(device, torch_dtype: torch.dtype):
    """
    Load Paligemma model and processor.

    Returns
    -------
    tuple[torch.nn.Module, transformers.AutoProcessor, torch.nn.Embedding]
        The model, its processor and the language-model input embedding layer.
    """
    from transformers.models.paligemma import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

    model_id = "google/paligemma2-3b-pt-224"  # or other Paligemma variant
    with torch.no_grad():
        model = (
            PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch_dtype # trust_remote_code=True, torch_dtype=torch_dtype
            )
            .eval()
            .to(device)
        )
    processor = PaliGemmaProcessor.from_pretrained("google/paligemma2-3b-mix-224")
    # processor = PaliGemmaProcessor.from_pretrained(
    #     model_id, trust_remote_code=True, use_fast=True
    # )
    # if hasattr(processor, "tokenizer"):
    #     processor.tokenizer.padding_side = "left"

    emb_layer = model.language_model.get_input_embeddings().to(torch_dtype).to(device)
    return model, processor, emb_layer


def _load_qwen2_5_vl(device, torch_dtype: torch.dtype):
    """
    Load Qwen2_5_VL model and processor.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    # Use the same loading approach as qwen2_5_vl.py
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map=device
        # model_id, torch_dtype=torch_dtype, device_map="cuda:0"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)
    
    emb_layer = model.model.get_input_embeddings().to(torch_dtype).to(device)
    return model, processor, emb_layer

def _load_model(
    model_name, device, torch_dtype: torch.dtype
) -> Tuple[torch.nn.Module, AutoProcessor, torch.nn.Embedding]:
    """Dispatch helper for supported VLMs."""
    if model_name.lower() == "eagle2":
        return _load_eagle2(device, torch_dtype)
    elif model_name.lower() == "paligemma":
        return _load_paligemma(device, torch_dtype)
    elif model_name.lower() == "qwen2_5_vl":
        return _load_qwen2_5_vl(device, torch_dtype)
    msg = f"Unsupported model: {model_name}"
    raise ValueError(msg)


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
        return out.logits if hasattr(out, "logits") else out


def _compile_eagle2_lm(
    language_model: torch.nn.Module,
    input_embeds: torch.Tensor,
    args: argparse.Namespace,
) -> torch.nn.Module:
    """
    Compile Eagle2 language model with Torch-TensorRT.

    The function follows the same precision-specific flag logic used in
    *run_llm.py* for consistency.
    """
    lm_wrap = _LMNoCache(language_model).to(DEVICE).eval()
    max_seq_len = input_embeds.shape[1] + args.num_tokens

    S = torch.export.Dim("seq", min=1, max=max_seq_len)
    position_ids = torch.arange(input_embeds.shape[1]).unsqueeze(0).to(DEVICE)
    dyn_shapes = {"inputs_embeds": {1: S}, "position_ids": {1: S}}

    # Precision-specific flags --------------------------------------------------#
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
            device=DEVICE,
            disable_tf32=True,
            use_python_runtime=True,
            debug=args.debug,
            offload_module_to_cpu=True,
            min_block_size=args.min_block_size,
        )
    return trt_mod


def _compile_paligemma_lm(
    language_model: torch.nn.Module,
    input_embeds: torch.Tensor,
    args: argparse.Namespace,
) -> torch.nn.Module:
    """
    Compile Paligemma language model with Torch-TensorRT.

    The function follows the same precision-specific flag logic used in
    *run_llm.py* for consistency.
    """
    lm_wrap = _LMNoCache(language_model).to(DEVICE).eval()
    max_seq_len = input_embeds.shape[1] + args.num_tokens

    S = torch.export.Dim("seq", min=1, max=max_seq_len)
    position_ids = torch.arange(input_embeds.shape[1]).unsqueeze(0).to(DEVICE)
    dyn_shapes = {"inputs_embeds": {1: S}, "position_ids": {1: S}}

    # Precision-specific flags --------------------------------------------------#
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
            device=DEVICE,
            disable_tf32=True,
            use_python_runtime=True,
            debug=args.debug,
            offload_module_to_cpu=True,
            min_block_size=args.min_block_size,
        )
    return trt_mod


def compile_torchtrt(
    model: torch.nn.Module, args: argparse.Namespace
) -> torch.nn.Module:
    """
    Front-end dispatcher mirroring *run_llm.py*'s `compile_torchtrt`.

    Depending on the target VLM, delegates to the appropriate compile routine.
    """
    torch_dtype = {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }.get(args.precision, torch.float32)

    example_embeds = torch.randn(
        1,
        2560,
        model.language_model.config.hidden_size,
        dtype=torch_dtype,
        device=DEVICE,
    )

    if args.model.lower() == "eagle2":
        return _compile_eagle2_lm(model.language_model, example_embeds, args)
    elif args.model.lower() == "paligemma":
        return _compile_paligemma_lm(model.language_model, example_embeds, args)

    msg = f"Unsupported model for compilation: {args.model}"
    raise ValueError(msg)


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
    parser.add_argument("--model", default="qwen2_5_vl", choices=["eagle2", "paligemma", "qwen2_5_vl"], help="VLM model name")
    parser.add_argument("--prompt", default="", help="Prompt text")
    parser.add_argument(
        "--precision",
        default="BF16",
        choices=["FP16", "BF16", "FP32"],
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

    args = parser.parse_args()

    # -------------------------------------------------------------------------#
    # 1. Model / processor / embeddings
    # -------------------------------------------------------------------------#
    dtype = {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }.get(args.precision, torch.float32)

    model, processor, emb_layer = _load_model(args.model, DEVICE, dtype)

    # -------------------------------------------------------------------------#
    # 2. Input construction (image + text prompt)
    # -------------------------------------------------------------------------#
    # url = "https://cdn.pixabay.com/photo/2019/08/08/23/33/car-4393990_1280.jpg"
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    # url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)

    if args.benchmark:
        prompt_len = args.isl - 1792 - 26
        prompt_txt = " ".join(["token"] * max(prompt_len, 0))
    else:
        prompt_txt = args.prompt

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_txt},
            ],
        }
    ]

    txt = [
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    ]
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=txt,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)
    # Inference: Generation of the output
    max_output_len = inputs["input_ids"].shape[1] + args.num_tokens

    # comment out for CUDA out of memory error
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("=============================")
    print("Original Torch Model generated text:")
    print(output_text)      
    print("=============================")


    # # img_in, vid_in = processor.process_vision_info(messages)
    # # inputs = processor(
    # #     text=txt, images=img_in, videos=vid_in, return_tensors="pt", padding=True
    # # ).to(DEVICE)

    # # inputs = processor(text=prompt_txt, images=image, return_tensors="pt").to(DEVICE)
    # input_len = inputs["input_ids"].shape[-1]

    # max_output_len = inputs["input_ids"].shape[1] + args.num_tokens
    # pyt_generation = model.generate(
    #     **inputs, max_new_tokens=100, do_sample=False
    # ) 
    # Original_pyt_gen_tokens = pyt_generation[0][input_len:]
    # Original_pyt_decoded = processor.decode(Original_pyt_gen_tokens, skip_special_tokens=True)
    # print("=============================")
    # print("Original Torch Model generated text:")
    # print(Original_pyt_decoded)
    # print("=============================")

    # -------------------------------------------------------------------------#
    # 3. Optional: PyTorch baseline
    # -------------------------------------------------------------------------#
    pyt_gen_tokens = pyt_timings = pyt_stats = None
    if args.enable_pytorch_run:
        pyt_gen_tokens = generate_mm_qwen2_5_vl(
            model,
            inputs["pixel_values"],
            inputs["input_ids"],
            inputs["image_grid_thw"],
            max_output_len,
            processor.tokenizer.eos_token_id,
            emb_layer,
        )
        # pyt_gen_tokens = generate_mm_paligemma(
        #     model,
        #     inputs["pixel_values"],
        #     inputs["input_ids"],
        #     max_output_len,
        #     processor.tokenizer.eos_token_id,
        #     emb_layer,
        # )
        print_outputs("Ouput tokens form custom generated function", pyt_gen_tokens, processor.tokenizer)
        if args.benchmark:
            pyt_timings = time_generate_mm(
                generate_mm_paligemma,
                model,
                inputs["pixel_values"].clone(),
                inputs["input_ids"].clone(),
                max_output_len,
                processor.tokenizer.eos_token_id,
                emb_layer,
                iterations=args.iterations,
            )
            pyt_stats = record_stats(
                "PyTorch",
                pyt_timings,
                args.precision,
                batch_size=args.batch_size,
                compile_time_s=None,
            )

    # Register static cache lowering passes if requested
    if args.cache == "static_v1":
        import static_cache_v1  # noqa: F401

    # -------------------------------------------------------------------------#
    # 4. Torch-TensorRT compile & run
    # -------------------------------------------------------------------------#
    trt_lm = compile_torchtrt(model, args)
    trt_model = copy.deepcopy(model)
    trt_model.language_model = trt_lm

    emb_layer = emb_layer.to(DEVICE)

    if args.cache == "static_v1":
        if args.model.lower() == "eagle2":
            trt_generate = generate_mm_with_static_cache
        elif args.model.lower() == "paligemma":
            trt_generate = generate_mm_paligemma_with_static_cache
    else:
        if args.model.lower() == "eagle2":
            trt_generate = generate_mm
        elif args.model.lower() == "paligemma":
            trt_generate = generate_mm_paligemma

    trt_gen_tokens = trt_generate(
        trt_model,
        inputs["pixel_values"],
        inputs["input_ids"],
        max_output_len,
        processor.tokenizer.eos_token_id,
        emb_layer,
        DEVICE if args.cache == "static_v1" else None,  # device arg only for static_v1
    )

    if args.benchmark:
        trt_timings = time_generate_mm(
            trt_generate,
            trt_model,
            inputs["pixel_values"].clone(),
            inputs["input_ids"].clone(),
            max_output_len,
            processor.tokenizer.eos_token_id,
            emb_layer,
            iterations=args.iterations,
            device=DEVICE if args.cache == "static_v1" else None,
        )
        trt_stats = record_stats(
            "TensorRT",
            trt_timings,
            args.precision,
            batch_size=args.batch_size,
            compile_time_s=None,
        )

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
