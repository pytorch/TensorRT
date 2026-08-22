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


def export_llm(model, inputs, min_seq_len=1, max_seq_len=16):
    """
    Exports the LLM model into an ExportedProgram with dynamic shapes.
    In the case of guard failures due to some PyTorch kernel implements, we also
    try to re-export the graph by expressing them as runtime assert nodes
    """
    with torch.no_grad():
        # max=1024 has contraint violation error. https://github.com/pytorch/pytorch/issues/125604
        seq_len = torch.export.Dim("seq_len", min=min_seq_len, max=max_seq_len)
        position_ids = torch.arange(inputs.shape[1]).unsqueeze(0).to(inputs.device)
        try:
            print("Trying to export the model using torch.export.export()..")
            # strict=False only enables aotautograd tracing and excludes dynamo.
            ep = torch.export.export(
                model,
                args=(inputs,),
                kwargs={"position_ids": position_ids},
                dynamic_shapes=({1: seq_len}, {1: seq_len}),
                strict=False,
            )
        except:
            print(
                "Trying torch.export._trace._export to trace the graph since torch.export.export() failed"
            )
            # This API is used to express the constraint violation guards as asserts in the graph.
            ep = torch.export._trace._export(
                model,
                args=(inputs,),
                kwargs={"position_ids": position_ids},
                dynamic_shapes=({1: seq_len}, {1: seq_len}),
                strict=False,
                prefer_deferred_runtime_asserts_over_guards=True,
            )

    return ep


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


# Load the LLaMA-2 model
DEVICE = torch.device("cuda:0")
llama_path = "meta-llama/Llama-2-7b-chat-hf"
with torch.no_grad():
    model = AutoModelForCausalLM.from_pretrained(
        llama_path, use_cache=False, attn_implementation="eager"
    ).eval()

# Set input and output sequence lengths
isl = 128
osl = 256

# Create random input tensors
input_tensors = [torch.randint(0, 5, (1, isl), dtype=torch.int64).cuda()]
# Convert the model to half precision (FP16)
model = model.half()
# Exports the LLM model into an ExportedProgram with dynamic shapes.
llama2_ep = export_llm(model, input_tensors[0], max_seq_len=osl)

# %%
# Compiler option
# ----------------------------------
#
# enable_weight_streaming=True option and use_explicit_typing=True are required to build
# the engine with weight streaming feature. use_explicit_typing=True option creates a
# `strongly typed network <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#strongly-typed-networks>`_ and only float32 precision is allowed in enabled_precisions option
#

# Create a TensorRT-compiled model
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
# Once you specify the enable_weight_streaming compile option, automatic budget size is configured.
# This automatic size may not always provide the optimal solution because the automatically determined
# budget lacks insight into the user's specific memory constraints and usage patterns

# Weight streaming context to get current weight budget information
weight_streaming_ctx = torch_tensorrt.runtime.weight_streaming(trt_model)
# Measure the mean latency of the model with weight streaming
mean_latency = time_generate(trt_model, input_tensors, osl, 1)
# Calculate the percentage of current weight budget used
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

# Use a context manager for weight streaming
with torch_tensorrt.runtime.weight_streaming(trt_model) as weight_streaming_ctx:
    # Get the total size of streamable weights in the engine
    streamable_budget = weight_streaming_ctx.total_device_budget

    # Scenario 1: Automatic weight streaming budget
    # Get the automatically determined weight streaming budget
    requested_budget = weight_streaming_ctx.get_automatic_weight_streaming_budget()
    # Set the device budget to the automatically determined value
    weight_streaming_ctx.device_budget = requested_budget
    # Measure the mean latency with automatic budget
    mean_latency = time_generate(trt_model, input_tensors, osl, 1)
    # Calculate the percentage of the weight budget used
    weight_budget_pct = (
        weight_streaming_ctx.device_budget
        / weight_streaming_ctx.total_device_budget
        * 100
    )
    print(
        f"Set auto weight streaming budget as {weight_budget_pct}%. {weight_streaming_ctx.device_budget} bytes out of {weight_streaming_ctx.total_device_budget}. mean latency = {mean_latency} ms"
    )

    # Scenario 2: Manual 10% weight streaming budget
    # Set the budget to 10% of the total streamable weights
    requested_budget = int(streamable_budget * 0.1)
    weight_streaming_ctx.device_budget = requested_budget
    # Measure the mean latency with 10% budget
    mean_latency = time_generate(trt_model, input_tensors, osl, 1)
    # Calculate the percentage of the weight budget used
    weight_budget_pct = (
        weight_streaming_ctx.device_budget
        / weight_streaming_ctx.total_device_budget
        * 100
    )
    print(
        f"Set weight streaming budget as {weight_budget_pct}%. {weight_streaming_ctx.device_budget} bytes out of {weight_streaming_ctx.total_device_budget}. mean latency = {mean_latency} ms"
    )
