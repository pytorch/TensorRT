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

# %%

# Define the parameters
MAX_TOKENS = 32
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
            use_cache=False,
            attn_implementation="eager",
        )
        .eval()
        .cuda()
    )

# %%
# Tokenize a sample input prompt and get pytorch model outputs
prompt = "I enjoy walking with my cute dog"
model_inputs = tokenizer(prompt, return_tensors="pt")
input_ids = model_inputs["input_ids"].cuda()

# Auto-regressive generation loop for greedy search using PyTorch model.
pyt_gen_tokens = model.generate(
    input_ids,
    max_length=MAX_TOKENS,
    use_cache=False,
    pad_token_id=tokenizer.eos_token_id,
)

# %%
# Compilation with `torch.compile` using tensorrt backend and generate TensorRT outputs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Compile the model and mark the input sequence length to be dynamic
torch._dynamo.mark_dynamic(input_ids, 1, min=2, max=1023)
model.forward = torch.compile(
    model.forward,
    backend="tensorrt",
    dynamic=None,
    options={
        "enabled_precisions": {torch.float32},
        "disable_tf32": True,
        "min_block_size": 1,
        "debug": True,
    },
)

# Auto-regressive generation loop for greedy decoding using TensorRT model
# The first token generation compiles the model using TensorRT and the second token
# encounters recompilation
trt_gen_tokens = model.generate(
    inputs=input_ids,
    max_length=MAX_TOKENS,
    use_cache=False,
    pad_token_id=tokenizer.eos_token_id,
)

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

# Pytorch model generated text:  I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with my dog. I'm not sure if I'll
# =============================
# TensorRT model generated text:  I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with my dog. I'm not sure if I'll
