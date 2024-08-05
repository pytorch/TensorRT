import copy
import time

import numpy as np
import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import export_llm, generate


def time_generate(model, inputs, max_tokens, eos_token_id, iterations=10):
    timings = []
    for _ in range(iterations):
        start_time = time.time()
        inputs_copy = copy.copy(inputs)
        generate(model, inputs_copy, max_tokens, eos_token_id)
        timings.append(time.time() - start_time)

    time_mean = np.mean(timings) * 1000  # convert to ms
    time_med = np.median(timings) * 1000  # convert to ms

    return time_mean, time_med


# Define tokenizer and model
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = (
    AutoModelForCausalLM.from_pretrained(
        "gpt2",
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
        attn_implementation="eager",
    )
    .eval()
    .to(torch_device)
    .half()
)

# Input prompt
model_inputs = tokenizer("I enjoy walking with my cute dog", return_tensors="pt").to(
    torch_device
)
input_ids = model_inputs["input_ids"]
max_tokens = 20

# Auto-regressive generation loop for greedy search using PyTorch model
pyt_gen_tokens, num_tokens_gen = generate(
    model, input_ids, max_tokens, tokenizer.eos_token_id
)
pyt_mean_time, pyt_med_time = time_generate(
    model, input_ids, max_tokens, tokenizer.eos_token_id
)

# Compile Torch-TRT model
gpt2_ep = export_llm(model, input_ids, max_seq_len=1024)
trt_model = torch_tensorrt.dynamo.compile(
    gpt2_ep,
    inputs=[input_ids],
    enabled_precisions={torch.float16},
    truncate_double=True,
    debug=False,
)

# Auto-regressive generation loop for greedy search using Torch-TensorRT model
generated_token_ids, num_tokens_gen = generate(
    trt_model, input_ids, max_tokens, tokenizer.eos_token_id
)
trt_mean_time, trt_med_time = time_generate(
    trt_model, input_ids, max_tokens, tokenizer.eos_token_id
)

# Decode the sentence
print("=============================")
print(
    "Pytorch model generated text: ",
    tokenizer.decode(pyt_gen_tokens[0], skip_special_tokens=True),
)
print(f"Pytorch total tokens generated: {num_tokens_gen}")
print(f"Pytorch total mean time in ms: {pyt_mean_time} median time: {pyt_med_time}")
print("=============================")
print(
    "TensorRT model generated text: ",
    tokenizer.decode(generated_token_ids[0], skip_special_tokens=True),
)
print(f"TensorRT total tokens generated: {num_tokens_gen}")
print(f"TensorRT total mean time in ms: {trt_mean_time} median time: {trt_med_time}")
