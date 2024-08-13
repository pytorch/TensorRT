import copy
import time

import numpy as np
import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import export_llm, generate

# Define the parameters
MAX_TOKENS = 32
DEVICE = torch.device("cuda:0")

# Define the GPT2 model from hugging face
# kv_cache is not supported in Torch-TRT currently.
# CPU is used here so that GPU memory is reserved for TRT compilation.
with torch.no_grad():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
        attn_implementation="eager",
    ).eval()

# Input prompt
prompt = "I enjoy walking with my cute dog"
model_inputs = tokenizer(prompt, return_tensors="pt")
input_ids = model_inputs["input_ids"]

# Auto-regressive generation loop for greedy decoding using PyTorch model
pyt_gen_tokens = generate(model, input_ids, MAX_TOKENS, tokenizer.eos_token_id)

# Export the GPT2 model into an ExportedProgram which is input of TRT compilation
gpt2_ep = export_llm(model, input_ids, max_seq_len=1024)
trt_model = torch_tensorrt.dynamo.compile(
    gpt2_ep,
    inputs=[input_ids],
    enabled_precisions={torch.float32},
    truncate_double=True,
    device=DEVICE,
    disable_tf32=True,
)

# Auto-regressive generation loop for greedy decoding using TensorRT model
# Move inputs to GPU
input_ids = input_ids.to(DEVICE)
trt_gen_tokens = generate(trt_model, input_ids, MAX_TOKENS, tokenizer.eos_token_id)

# Decode the sentence
print("=============================")
print(
    "Pytorch model generated text: ",
    tokenizer.decode(pyt_gen_tokens[0], skip_special_tokens=True),
)
print("=============================")
print(
    "TensorRT model generated text: ",
    tokenizer.decode(trt_gen_tokens[0], skip_special_tokens=True),
)
