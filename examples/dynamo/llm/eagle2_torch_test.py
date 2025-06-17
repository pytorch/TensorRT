import torch
import torch.nn as nn
import torch_tensorrt
import copy, requests
from PIL import Image
from transformers import AutoModel, AutoProcessor, GenerationMixin
from utils import generate_mm

import transformers.models.qwen2.modeling_qwen2 as mq
mq.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = mq.ALL_ATTENTION_FUNCTIONS["sdpa"]
# Load the base model and processor
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from register_sdpa import *

# 1) Load base model & processor
def load_base(device="cuda:1"):
    model_id = "nvidia/Eagle2-2B"
    model = (
        AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    return model, processor


if __name__ == "__main__":
    device = "cuda:1"
    torch.cuda.set_device(device)
    base_model, processor = load_base(device)

    # Build models (Torch reference & TensorRT-optimised
    url = "https://cdn.pixabay.com/photo/2019/08/08/23/33/car-4393990_1280.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Build a 230-word placeholder prompt ("token token …") so that ISL totals 256 tokens after template overhead.
    prompt_tokens = ["token"] * 230
    prompt_text = " ".join(prompt_tokens)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]

    # Apply chat template & process vision info
    text_list = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
    image_inputs, video_inputs = processor.process_vision_info(messages)

    # Tokenise → dict of tensors
    model_inputs = processor(
        text=text_list,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # seq_tokens, step_times = generate_mm(base_model, model_inputs["pixel_values"], model_inputs["input_ids"], processor.tokenizer.eos_token_id)
    emb_layer = base_model.language_model.get_input_embeddings()
    seq_tokens_torch, step_times_torch = generate_mm(base_model, model_inputs["pixel_values"], model_inputs["input_ids"], processor.tokenizer.eos_token_id, emb_layer)

    # Results
    print("\n================  RESULTS  ================")
    print("PyTorch generated text   :", processor.tokenizer.decode(seq_tokens_torch[0][model_inputs["input_ids"].shape[1]:], skip_special_tokens=True))

    # Per-token breakdown (ms)
    def _fmt(ts):
        return [f"{t*1000:.2f}" for t in ts]

    print("PyTorch per-token times (ms):", _fmt(step_times_torch))
    print("===========================================") 
