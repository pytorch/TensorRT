"""
.. _torch_compile_phi3_vision:

Compiling Phi 3 vision model from Hugging Face using the Torch-TensorRT `torch.compile` Backend
======================================================

This script is intended as a sample of the Torch-TensorRT workflow with `torch.compile` on a Phi 3 vision model from Hugging Face.
"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import requests
import torch
import torch_tensorrt
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# %%
# Load the pre-trained model weights from Hugging Face
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

model_id = "microsoft/Phi-3-vision-128k-instruct"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype="auto"
).cuda()

# %%
# Compile the model with torch.compile, using Torch-TensorRT backend
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

model = torch.compile(model, backend="tensorrt")

# %%
# Write prompt and load image
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

user_prompt = "<|user|>\n"
assistant_prompt = "<|assistant|>\n"
prompt_suffix = "<|end|>\n"

# single-image prompt
prompt = f"{user_prompt}<|image_1|>\nWhat is shown in this image?{prompt_suffix}{assistant_prompt}"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
print(f">>> Prompt\n{prompt}")

image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

# %%
# Inference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    eos_token_id=processor.tokenizer.eos_token_id,
)
generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f">>> Response\n{response}")
