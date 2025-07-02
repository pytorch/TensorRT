import copy
import os
import sys

import requests
import torch
import torch.nn as nn
import torch_tensorrt
from PIL import Image
from transformers import AutoModel, AutoProcessor, GenerationMixin
from utils import (
    generate_mm,
    generate_mm_with_static_cache,
    generate_mm_with_static_cache_timing,
    generate_mm_with_timing,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# ───────────────────────────── SDPA patch ──────────────────────────────
import transformers.models.qwen2.modeling_qwen2 as mq
from register_sdpa import *

mq.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = mq.ALL_ATTENTION_FUNCTIONS["sdpa"]

CACHE = "static_v1"
# CACHE = None


# 1) Load base model & processor
def load_base(device="cuda:1"):
    model_id = "nvidia/Eagle2-2B"
    model = (
        AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float16
        )
        .to(device)
        .eval()
    )
    # model = model.to(device)

    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    return model, processor


class LMNoCache(torch.nn.Module):
    """Wrapper exposing inputs_embeds and position_ids forward (no KV-cache)"""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, inputs_embeds, position_ids):
        out = self.lm(
            inputs_embeds=inputs_embeds, position_ids=position_ids
        )  # , use_cache=False)
        # Ensure a CausalLMOutput/loss-style object with .logits attribute is returned
        if hasattr(out, "logits"):
            return out.logits
        else:
            # When using compile path we'll want a simple tensor
            return out


def compile_eagle2_lm_with_trt_embed(language_model, example_embeds, device="cuda:0"):
    """Compile language model that expects inputs_embeds using Torch-TensorRT."""

    lm_wrap = LMNoCache(language_model).to(device).eval()

    S = torch.export.Dim("seq", min=1, max=2560)
    example_position_ids = (
        torch.arange(example_embeds.shape[1]).unsqueeze(0).to(example_embeds.device)
    )
    dyn_shapes = {"inputs_embeds": {1: S}, "position_ids": {1: S}}

    with torch.inference_mode():
        exported = torch.export.export(
            lm_wrap,
            (example_embeds, example_position_ids),
            dynamic_shapes=dyn_shapes,
            strict=False,
        )

    trt_mod = torch_tensorrt.dynamo.compile(
        exported,
        inputs=[example_embeds, example_position_ids],
        enabled_precisions={torch.float32},
        device=device,
        use_explicit_typing=True,
        use_fp32_acc=True,
        disable_tf32=True,
        use_python_runtime=True,
        debug=False,
        offload_module_to_cpu=True,
        min_block_size=1,
    )
    return trt_mod


def compile_submodules(base_model, device="cuda:1"):
    vision_model = base_model.vision_model
    mlp1 = base_model.mlp1
    language_model = base_model.language_model

    hidden_size = language_model.config.hidden_size
    dummy_seq = 2050
    dummy_embeds = torch.randn(
        1, dummy_seq, hidden_size, dtype=torch.float16, device=device
    )
    trt_lm = compile_eagle2_lm_with_trt_embed(language_model, dummy_embeds, device)

    return vision_model, mlp1, trt_lm


def build_trt_model(device="cuda:1"):
    base_model, processor = load_base(device)

    # if args.cache == "static_v1":
    # This import is required to register static v1 KV cache transformations as lowering passes

    trt_vis, trt_mlp1, trt_lm = compile_submodules(
        base_model, device
    )  # None, None, None #compile_submodules(base_model, device)

    trt_model = copy.deepcopy(base_model)

    # trt_model = trt_model.to(device)
    # base_model = base_model.to(device)

    trt_model.vision_model, trt_model.mlp1, trt_model.language_model = (
        trt_vis,
        trt_mlp1,
        trt_lm,
    )

    return base_model, trt_model, processor


@torch.no_grad()
def run_benchmark(isl=2048, osl=128, device="cuda:1"):
    torch.cuda.set_device(device)

    if CACHE == "static_v1":
        import static_cache_v1

    # Build models (Torch reference & TensorRT-optimised)
    base_model, trt_model, processor = build_trt_model(device)

    url = "https://cdn.pixabay.com/photo/2019/08/08/23/33/car-4393990_1280.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Eagle2 template overhead 26 tokens + 1792 image tokens
    prompt_len = isl - 1792 - 26
    prompt = " ".join(["token"] * max(prompt_len, 0))

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
    ).to(device)

    # Validate ISL (optional)
    assert (
        inputs["input_ids"].shape[1] == isl
    ), f"Actual ISL={inputs['input_ids'].shape[1]}, Requested ISL={isl}"

    # ----- Generation & Timing -----
    emb_layer = base_model.language_model.get_input_embeddings()
    emb_layer = emb_layer.to(device)

    if CACHE == "static_v1":
        seq_tokens, step_times, overall_time, vision_time, mlp_time = (
            generate_mm_with_static_cache_timing(
                trt_model,
                inputs["pixel_values"],
                inputs["input_ids"],
                processor.tokenizer.eos_token_id,
                emb_layer,
                max_new_tokens=osl,
            )
        )
        lm_time = sum(step_times)
    else:
        seq_tokens, step_times, overall_time, vision_time, mlp_time = (
            generate_mm_with_timing(
                trt_model,
                inputs["pixel_values"],
                inputs["input_ids"],
                processor.tokenizer.eos_token_id,
                emb_layer,
                max_new_tokens=osl,
            )
        )
        lm_time = sum(step_times)

    input_len = inputs["input_ids"].shape[1]
    gen_ids = seq_tokens[:, input_len:]  # Part after prompt
    gen_text = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # ----- Print Results -----
    print(f"\n[torchtrt-CustomGenerate-KV_False_SDPA]  ISL={isl}  OSL={osl}")
    print(f"  Vision : {vision_time:.2f} ms")
    print(f"  MLP    : {mlp_time:.2f} ms")
    print(f"  Language  : {lm_time:.2f} ms")
    print(f"  Overall   : {overall_time:.2f} ms")
    print("  └─ Generated text:", gen_text or "<empty>")


if __name__ == "__main__":
    run_benchmark(isl=2048, osl=128, device="cuda:1")
