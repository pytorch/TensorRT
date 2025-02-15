"""
.. _torch_export_gpt2:

Compiling GPT2 using the dynamo backend
==========================================================

This script illustrates Torch-TensorRT workflow with dynamo backend on popular GPT2 model.
"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import export_llm, generate

# %%

# Define the parameters and initialize the model
MAX_TOKENS = 6
DEVICE = torch.device("cuda:0")

# Define the GPT2 model from hugging face
# kv_cache is not supported in Torch-TRT currently.
# CPU is used here so that GPU memory is reserved for TRT compilation.
with torch.no_grad():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = (
        AutoModelForCausalLM.from_pretrained(
            "gpt2",
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            attn_implementation="sdpa",
        )
        .eval()
        .half()
        .to(DEVICE)
    )

# %%
# Tokenize a sample input prompt and get pytorch model outputs
prompt = "What is parallel programming ?"
model_inputs = tokenizer(prompt, return_tensors="pt")
input_ids = model_inputs["input_ids"].to(DEVICE)

# Auto-regressive generation loop for greedy decoding using PyTorch model
# We use a custom generate function which is very similar to the huggingface one.
pyt_gen_tokens = generate(model, input_ids, MAX_TOKENS, tokenizer.eos_token_id)


# %%
# Compilation with `Torch-TensorRT` using dynamo backend and generate TensorRT outputs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Export the GPT2 model into an ExportedProgram which is input of TRT compilation
# To compile the model in FP16, we do the following
# 1) Cast the model to FP16 via model.half()
# 2) Enable use_explicit_typing=True. Certain layers are explicitly casted to FP32 within the pytorch model and this flag respects this behavior during TRT compilation
# 3) Enable use_fp32_acc=True. This ensures all the matmuls are accumulated in FP32 precision (similar to PyTorch)
gpt2_ep = export_llm(model, input_ids, max_seq_len=1024)
with torch_tensorrt.logging.debug():
    trt_model = torch_tensorrt.dynamo.compile(
        gpt2_ep,
        inputs=[input_ids],
        enabled_precisions={torch.float32},
        truncate_double=True,
        device=DEVICE,
        disable_tf32=True,
        use_explicit_typing=True,
        use_fp32_acc=True,
        debug=True,
        torch_executed_ops={"torch.ops.tensorrt.flashinfer_forward"},
    )

# Auto-regressive generation loop for greedy decoding using TensorRT model
# We use a custom generate function which is very similar to the huggingface one.
# Move inputs to GPU
input_ids = input_ids.to(DEVICE)
trt_gen_tokens = generate(trt_model, input_ids, MAX_TOKENS, tokenizer.eos_token_id)

# %%
# Decode the output sentences of PyTorch and TensorRT
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

# Prompt : What is parallel programming ?

# =============================
# Pytorch model generated text: The parallel programming paradigm is a set of programming languages that are designed to be used in parallel. The main difference between parallel programming and parallel programming is that

# =============================
# TensorRT model generated text: The parallel programming paradigm is a set of programming languages that are designed to be used in parallel. The main difference between parallel programming and parallel programming is that
