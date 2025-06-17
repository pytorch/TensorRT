import torch
import torch.nn as nn
import torch_tensorrt
import copy, requests, time
from PIL import Image
from transformers import AutoModel, AutoProcessor, GenerationMixin
from utils import generate_mm, generate_mm_with_timing 

import transformers.models.qwen2.modeling_qwen2 as mq

# Store SDPA activation status as a global variable
SDPA_ENABLED = False
ORIGINAL_ATTENTION_FUNCTION = mq.ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

# Function to load model and processor
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

# Function to enable/disable SDPA
def set_attention_function(enable_sdpa=True):
    global SDPA_ENABLED
    SDPA_ENABLED = enable_sdpa
    if enable_sdpa:
        mq.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = mq.ALL_ATTENTION_FUNCTIONS["sdpa"]
    else:
        # Revert to default attention function (e.g., eager, adjust based on actual implementation)
        mq.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = ORIGINAL_ATTENTION_FUNCTION


# Function to measure time for TransformersGenerate
def transformers_generate_with_timing(model, processor, messages, max_new_tokens=64, use_cache=False):
    overall_start = torch.cuda.Event(enable_timing=True)
    overall_end = torch.cuda.Event(enable_timing=True)
    overall_start.record()

    text_list = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
    image_inputs, video_inputs = processor.process_vision_info(messages)
    model_inputs = processor(
        text=text_list,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    # Convert attention_mask to bool if SDPA is enabled
    if SDPA_ENABLED and "attention_mask" in model_inputs:
        model_inputs["attention_mask"] = model_inputs["attention_mask"].bool()
        print("Converted attention_mask to bool for TransformersGenerate")

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, use_cache=use_cache)

    overall_end.record()
    torch.cuda.synchronize()
    overall_time = overall_start.elapsed_time(overall_end)

    # Detailed component timing not available in Transformers generate
    return generated_ids, overall_time

def run_performance_analysis(device="cuda:1", isl=2048, osl=128):
    torch.cuda.set_device(device)
    base_model, processor = load_base(device)

    # Prepare input data
    url = "https://cdn.pixabay.com/photo/2019/08/08/23/33/car-4393990_1280.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    prompt_tokens = ["token"] * (isl - 1792 - 26)
    prompt_text = " ".join(prompt_tokens)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]

    results = {}

    for case in ["CustomGenerate-KV_False_SDPA",
                 "CustomGenerate-KV_False_FI",
                 "TransformersGenerate-KV_True_SDPA",
                 "TransformersGenerate-KV_True_FI"]:
        enable_sdpa = "SDPA" in case
        use_custom_generate = "CustomGenerate" in case
        set_attention_function(enable_sdpa)

        # Prepare input for CustomGenerate
        text_list = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
        image_inputs, video_inputs = processor.process_vision_info(messages)
        model_inputs = processor(
            text=text_list,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(device)

        # Remember input length (number of tokens)
        input_len = model_inputs["input_ids"].shape[1]
            
        if use_custom_generate:
            if SDPA_ENABLED and "attention_mask" in model_inputs:
                model_inputs["attention_mask"] = model_inputs["attention_mask"].bool()
                print("Converted attention_mask to bool for CustomGenerate")

            emb_layer = base_model.language_model.get_input_embeddings()
            # generation & timing measurement
            seq_tokens, step_times, overall_time, vision_time, mlp_time = generate_mm_with_timing(
                base_model,
                model_inputs["pixel_values"],
                model_inputs["input_ids"],
                processor.tokenizer.eos_token_id,
                emb_layer,
                max_new_tokens=osl
            )
            lm_time = sum(step_times)


            # Decode only the generated part
            gen_ids = seq_tokens[:, input_len:]
            gen_text = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            # ───────────────────────────────────────────────────────────────

            results[case] = {
                "overall": overall_time,
                "VISION": vision_time,
                "MLP": mlp_time,
                "Language Model": lm_time,
                "Output Text": gen_text
            }

        else:
            # transformers generate that takes input embeds defaults to use_cache=True
            use_cache = True
            # Execute TransformersGenerate
            generated_ids, overall_time = transformers_generate_with_timing(
                base_model, processor, messages, max_new_tokens=osl, use_cache=use_cache
            )

            gen_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # ───────────────────────────────────────────────────────────────

            results[case] = {
                "overall": overall_time,
                "VISION": "N/A",
                "MLP": "N/A",
                "Language Model": "N/A",
                "Output Text": gen_text
            }

    # Print results
    for case, times in results.items():
        print(f"\n{case}:")
        for component, value in times.items():
            print(f"  {component}: {value}")

if __name__ == "__main__":
    run_performance_analysis()