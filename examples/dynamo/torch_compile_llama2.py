"""
.. _torch_compile_gpt2:

Compiling GPT2 using the Torch-TensorRT `torch.compile` Backend
==========================================================

This interactive script is intended as a sample of the Torch-TensorRT workflow with `torch.compile` on a GPT2 model."""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate

# %%

# Define the parameters
MAX_TOKENS = 32
DEVICE = torch.device("cuda:0")

# Define the GPT2 model from hugging face
# kv_cache is not supported in Torch-TRT currently.
# CPU is used here so that GPU memory is reserved for TRT compilation.
llama_path = "meta-llama/Llama-2-7b-chat-hf"
with torch.no_grad():
    model = AutoModelForCausalLM.from_pretrained(
        llama_path, use_cache=False, attn_implementation="eager"
    ).eval()

tokenizer = AutoTokenizer.from_pretrained(llama_path)

# %%
# Tokenize a sample input prompt and get pytorch model outputs
prompt = "I enjoy walking with my cute dog"
model_inputs = tokenizer(prompt, return_tensors="pt")
input_ids = model_inputs["input_ids"].cuda()

# Auto-regressive generation loop for greedy search using PyTorch model.
# We use a custom generate function which is very similar to the huggingface one.
# pyt_gen_tokens = generate(model, input_ids, MAX_TOKENS, tokenizer.eos_token_id)

# %%
# Compilation with `torch.compile` using tensorrt backend and generate TensorRT outputs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Compile the model and mark the input sequence length to be dynamic
with torch_tensorrt.logging.debug():
    torch._dynamo.mark_dynamic(input_ids, 1, min=7, max=1023)
    model.forward = torch.compile(
        model.forward,
        backend="tensorrt",
        dynamic=None,
        options={
            "enabled_precisions": {torch.float32},
            "disable_tf32": True,
            "debug": True,
            # "use_python_runtime": True
        },
    )
model(input_ids)
breakpoint()
model(input_ids)
# Auto-regressive generation loop for greedy decoding using TensorRT model
# We use a custom generate function which is very similar to the huggingface one.
# Move inputs to GPU
input_ids = input_ids.to(DEVICE)
trt_gen_tokens = generate(model, input_ids, MAX_TOKENS, tokenizer.eos_token_id)

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

# %%
# The output sentences should look like
#
#
