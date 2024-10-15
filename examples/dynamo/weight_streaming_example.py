"""
.. _weight_streaming_example:

Weight Streaming
=======================

Weight streaming in TensorRT is a powerful feature designed to overcome GPU memory limitations
when working with large models. It enables running models larger than available GPU memory
by streaming weight data from host (CPU) memory to GPU memory during inference.

Streaming larger amounts of memory will likely result in lower performance. But if
streaming weights allows the user to run larger batch sizes and it can lead to higher throughput.
This increased throughput can sometimes outweigh the slowdown caused by streaming weights.
The optimal amount of memory to stream varies depending on the specific model and hardware.
Experimenting with different memory limits can help find the best balance between streaming
overhead and batch size benefits.

This example uses a pre-trained Llama-2 model and show how to use weight streaming feature with
Torch-TensorRT.
    1. compile option - build trt engine with weight streaming feature
    2. runtime api - weight streaming budget control by context manager
"""

# %%
# Imports and Model Definition
# ----------------------------------

import copy
import timeit

import numpy as np
import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM


def time_generate(model, inputs, output_seq_length, iterations=10):
    """
    Measure the time for generating a sentence over certain number of iterations
    """
    # We only support single input (B x seq_len) for LLMs now
    input_seq = inputs[0]
    with torch.no_grad():
        timings = []
        for _ in range(iterations):
            start_time = timeit.default_timer()
            inputs_copy = copy.copy(input_seq)
            # Greedy decoding of the model. This generates up to max_tokens.
            while inputs_copy.shape[1] <= output_seq_length:
                outputs = model(inputs_copy)
                logits = outputs.logits
                next_token_logits = logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                inputs_copy = torch.cat([inputs_copy, next_tokens[:, None]], dim=-1)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            timings.append(end_time - start_time)

    times = np.array(timings)
    time_mean_ms = np.mean(times) * 1000

    return time_mean_ms


DEVICE = torch.device("cuda:0")
llama_path = "meta-llama/Llama-2-7b-chat-hf"
with torch.no_grad():
    model = AutoModelForCausalLM.from_pretrained(
        llama_path, use_cache=False, attn_implementation="eager"
    ).eval()

isl = 128
osl = 256

input_tensors = [torch.randint(0, 5, (1, isl), dtype=torch.int64).cuda()]
model = model.half()
with torch.no_grad():
    seq_len = torch.export.Dim("seq_len", min=1, max=osl)
    # strict=False only enables aotautograd tracing and excludes dynamo.
    llama2_ep = torch.export.export(
        model, tuple(input_tensors), dynamic_shapes=({1: seq_len},), strict=False
    )

# %%
# Compiler option
# ----------------------------------
#
# enable_weight_streaming=True option and use_explicit_typing=True are required to build
# the engine with weight streaming feature. use_explicit_typing=True option creates a
# `strongly typed network <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#strongly-typed-networks>`_ and only float32 precision is allowed in enabled_precisions option
#
trt_model = torch_tensorrt.dynamo.compile(
    llama2_ep,
    inputs=input_tensors,
    enabled_precisions={torch.float32},
    truncate_double=True,
    device=DEVICE,
    use_explicit_typing=True,
    enable_weight_streaming=True,
)

# Warm up for 3 iterations
_ = time_generate(trt_model, input_tensors, osl, 3)

# %%
# Running with automatic budget size
# ----------------------------------
#
# Once you specify the enable_weight_streaming option, automatic budget size is configured.
# This automatic size may not always provide the optimal solution because the automatically determined
# budget lacks insight into the user's specific memory constraints and usage patterns
weight_streaming_ctx = torch_tensorrt.runtime.weight_streaming(trt_model)
mean_latency = time_generate(trt_model, input_tensors, osl, 1)
weight_budget_pct = (
    weight_streaming_ctx.device_budget / weight_streaming_ctx.total_device_budget * 100
)
print(
    f"Set weight streaming budget as {weight_budget_pct}%. {weight_streaming_ctx.device_budget} bytes out of {weight_streaming_ctx.total_device_budget}. mean latency = {mean_latency} ms"
)

# %%
# Running with weight streaming context manager
# ----------------------------------
#
# Weight streaming budget can be limited by using weight streaming context manager.
# The permissible range for the budget size is from 0 to ctx.total_device_budget.
# 0 means maximum memory savings occur by using minimum amounts of memory. Value
# equal to ctx.total_device_budget will disable weight streaming.
# If multiple trt engines are created, budgets are distributed proportionally

with torch_tensorrt.runtime.weight_streaming(trt_model) as weight_streaming_ctx:
    # The size of the streamable weights in the engine
    streamable_budget = weight_streaming_ctx.total_device_budget

    # get automatic weight streaming budget size by using get_automatic_weight_streaming_budget
    requested_budget = weight_streaming_ctx.get_automatic_weight_streaming_budget()
    # Set and get the current weight streaming budget for inference
    weight_streaming_ctx.device_budget = requested_budget
    mean_latency = time_generate(trt_model, input_tensors, osl, 1)
    weight_budget_pct = (
        weight_streaming_ctx.device_budget
        / weight_streaming_ctx.total_device_budget
        * 100
    )
    print(
        f"Set auto weight streaming budget as {weight_budget_pct}%. {weight_streaming_ctx.device_budget} bytes out of {weight_streaming_ctx.total_device_budget}. mean latency = {mean_latency} ms"
    )

    # Set 10% of weight streaming budget
    requested_budget = int(streamable_budget * 0.1)
    weight_streaming_ctx.device_budget = requested_budget
    mean_latency = time_generate(trt_model, input_tensors, osl, 1)
    weight_budget_pct = (
        weight_streaming_ctx.device_budget
        / weight_streaming_ctx.total_device_budget
        * 100
    )
    print(
        f"Set weight streaming budget as {weight_budget_pct}%. {weight_streaming_ctx.device_budget} bytes out of {weight_streaming_ctx.total_device_budget}. mean latency = {mean_latency} ms"
    )
