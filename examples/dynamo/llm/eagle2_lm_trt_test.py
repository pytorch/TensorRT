import time
import torch
import torch_tensorrt
from transformers import AutoProcessor, AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# Re-use helper utilities shipped with the other dynamo examples
from utils import export_llm, generate
import transformers.models.qwen2.modeling_qwen2 as mq

mq.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = mq.ALL_ATTENTION_FUNCTIONS["sdpa"]
# Load the base model and processor
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from register_sdpa import *


def load_eagle2_model(device="cuda:0"):
    model_id = "nvidia/Eagle2-2B"
    model = (
        AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    return model, processor

# Compile the language model with Torch-TensorRT

def compile_eagle2_lm_with_trt(language_model, example_input_ids, device="cuda:0"):
    """Compile the language model with Torch-TensorRT using dynamo backend."""
    exported_program = export_llm(language_model, example_input_ids, min_seq_len=2, max_seq_len=1023)

    trt_model = torch_tensorrt.dynamo.compile(
        exported_program,
        inputs=[example_input_ids],
        enabled_precisions={torch.float32},  # FP16 inference
        device=device,
        use_fp32_acc=True,  # keep matmul accumulations in FP32 (matches PyTorch)
        use_explicit_typing=True,
    )
    return trt_model

# Helper function to measure generation time

def timed_generate(model, input_ids, eos_id, max_new_tokens):
    """Helper that returns runtime (seconds) + generated ids using greedy decode."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        out = generate(model, input_ids, max_new_tokens, eos_id)
    torch.cuda.synchronize()
    return out, time.perf_counter() - start

if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)

    # Load the model and processor
    base_model, processor = load_eagle2_model(device)
    language_model = base_model.language_model

    # Prepare input
    prompt = "I enjoy walking with my cute dog"
    input_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Baseline (Pure PyTorch)
    torch_out, torch_time = timed_generate(
        language_model, input_ids.clone(), processor.tokenizer.eos_token_id, 128
    )
    print("PyTorch generated text   :", processor.tokenizer.decode(torch_out[0], skip_special_tokens=True))

    # Compile with TensorRT
    print("\nCompiling Eagle2 language model with Torch-TensorRT …")
    trt_model = compile_eagle2_lm_with_trt(language_model, input_ids, device)

    trt_out, trt_time = timed_generate(
        trt_model, input_ids.clone(), processor.tokenizer.eos_token_id, 128
    )

    # Results
    print("\n================  RESULTS  ================")
    print("PyTorch generated text   :", processor.tokenizer.decode(torch_out[0], skip_special_tokens=True))
    print("TensorRT generated text  :", processor.tokenizer.decode(trt_out[0], skip_special_tokens=True))
    print("Tokens identical         :", torch.equal(torch_out, trt_out))
    print()
    print(f"PyTorch time   : {torch_time:.3f} s")
    print(f"TensorRT time  : {trt_time:.3f} s")
    if trt_time > 0:
        print(f"Speed-up (×)   : {torch_time / trt_time:.2f}")
    print("===========================================") 