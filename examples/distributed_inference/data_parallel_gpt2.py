"""
.. _data_parallel_gpt2:

Torch-TensorRT Distributed Inference
======================================================

This interactive script is intended as a sample of distributed inference using data
parallelism using Accelerate
library with the Torch-TensorRT workflow on GPT2 model.

"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import torch
import torch_tensorrt
from accelerate import PartialState
from transformers import AutoTokenizer, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Set input prompts for different devices
prompt1 = "GPT2 is a model developed by."
prompt2 = "Llama is a model developed by "

input_id1 = tokenizer(prompt1, return_tensors="pt").input_ids
input_id2 = tokenizer(prompt2, return_tensors="pt").input_ids

distributed_state = PartialState()

# Import GPT2 model and load to distributed devices
model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(distributed_state.device)


# Instantiate model with Torch-TensorRT backend
model.forward = torch.compile(
    model.forward,
    backend="torch_tensorrt",
    options={
        "truncate_long_and_double": True,
        "enabled_precisions": {torch.float16},
    },
    dynamic=False,
)

# %%
# Inference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Assume there are 2 processes (2 devices)
with distributed_state.split_between_processes([input_id1, input_id2]) as prompt:
    cur_input = torch.clone(prompt[0]).to(distributed_state.device)

    gen_tokens = model.generate(
        cur_input,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
