"""
Test torch.compile(backend="tensorrt") with Llama 3.2 1B using dynamic shapes.
"""

import logging
import os
import sys

import torch
import torch_tensorrt

# Register SDPA converter and lowering pass
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from torchtrt_ext.register_sdpa import enable_sdpa_converter
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


print(f"Loading model: {MODEL_NAME}")
model = (
    AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_cache=False,
        attn_implementation="sdpa",
        torch_dtype=torch.float32,
    )
    .eval()
    .cuda()
)

# Register the SDPA converter so the SDPA pass gets added to pre-AOT lowering
enable_sdpa_converter(MODEL_NAME, model.config)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]  # shape: (1, seq_len)

print(f"Input shape: {input_ids.shape}")

# Baseline PyTorch output
with torch.inference_mode():
    pyt_out = model(input_ids=input_ids)
pyt_logits = pyt_out.logits
pyt_next_token = pyt_logits[0, -1].argmax().item()
print(
    f"PyTorch next token: {pyt_next_token!r} => {tokenizer.decode([pyt_next_token])!r}"
)

# Compile with torch.compile(backend="tensorrt")
print("\nCompiling with torch.compile(backend='tensorrt', dynamic=True)...")
compiled_model = torch.compile(
    model,
    backend="tensorrt",
    dynamic=True,
    options={
        "debug": False,
        "min_block_size": 1,
        "use_fp32_acc": False,
    },
)

# Warm up / compile
print("Running compiled model (first call triggers compilation)...")
with torch.inference_mode():
    try:
        trt_out = compiled_model(input_ids=input_ids)
        trt_logits = trt_out.logits
        trt_next_token = trt_logits[0, -1].argmax().item()
        print(
            f"TRT next token:     {trt_next_token!r} => {tokenizer.decode([trt_next_token])!r}"
        )
        token_match = pyt_next_token == trt_next_token
        print(f"Token match: {token_match}")

        import torch.nn.functional as F

        cos_sim = F.cosine_similarity(
            pyt_logits[0, -1].unsqueeze(0), trt_logits[0, -1].unsqueeze(0)
        ).item()
        max_diff = (pyt_logits[0, -1] - trt_logits[0, -1]).abs().max().item()
        print(f"Cosine similarity: {cos_sim:.6f}")
        print(f"Max logit diff:    {max_diff:.6f}")
    except Exception as e:
        print(f"ERROR during TRT compilation/execution: {e}")
        import traceback

        traceback.print_exc()

# Test with a different sequence length (dynamic shape test)
print("\nTesting with different sequence length (dynamic shapes)...")
with torch.inference_mode():
    try:
        longer_prompt = "The capital of France is Paris. The capital of Germany is"
        longer_inputs = tokenizer(longer_prompt, return_tensors="pt").to("cuda")
        longer_input_ids = longer_inputs["input_ids"]
        print(f"Longer input shape: {longer_input_ids.shape}")

        pyt_out2 = model(input_ids=longer_input_ids)
        trt_out2 = compiled_model(input_ids=longer_input_ids)

        match2 = (
            pyt_out2.logits[0, -1].argmax().item()
            == trt_out2.logits[0, -1].argmax().item()
        )
        print(f"Dynamic shape token match: {match2}")
    except Exception as e:
        print(f"ERROR during dynamic shape test: {e}")
        import traceback

        traceback.print_exc()
