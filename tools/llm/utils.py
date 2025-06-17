import copy
import timeit

import numpy as np
import torch
from transformers import StoppingCriteriaList
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
)


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
                allow_complex_guards_as_runtime_asserts=True,
            )

    return ep


def get_zeroed_static_cache_inputs(model: torch.fx.GraphModule):
    """
    Extracts and returns zeroed static KV cache tensors from a torch.fx.GraphModule. This should only be used for static cache_v1 and static cache_v2.

    This function identifies placeholder nodes in the graph that represent KV cache tensors,
    and creates zeroed tensors with the same shape, dtype, and device as the original placeholders.

    Args:
        model (torch.fx.GraphModule): The exported model graph containing KV cache placeholders

    Returns:
        tuple: A tuple of zeroed tensors corresponding to the KV cache placeholders in the graph
    """
    # placeholder nodes are expected to be in the following order:
    # input_ids, kv_cache_key, kv_cache_value, start_idx, end_idx
    placeholder_nodes = [node for node in model.graph.nodes if node.op == "placeholder"]
    # The first two inputs are input_ids, position_ids. The last two inputs are start_idx, end_idx. In between are the KV cache tensors.
    kv_cache_inputs = placeholder_nodes[2:-2]
    zeroed_kv_cache_inputs = []
    for input in kv_cache_inputs:
        zeroed_kv_cache_inputs.append(
            torch.zeros(
                input.meta["val"].shape,
                dtype=input.meta["val"].dtype,
                device=torch.device("cuda:0"),
            )
        )

    return tuple(zeroed_kv_cache_inputs)


def get_zeroed_dynamic_cache_inputs(model: torch.fx.GraphModule):
    """
    Extracts and returns zeroed KV cache tensors from a torch.fx.GraphModule. This should only be used for dynamic cache.

    This function identifies placeholder nodes in the graph that represent KV cache tensors,
    and creates zeroed tensors with the same shape, dtype, and device as the original placeholders.

    Args:
        model (torch.fx.GraphModule): The exported model graph containing KV cache placeholders

    Returns:
        tuple: A tuple of zeroed tensors corresponding to the KV cache placeholders in the graph
    """
    # placeholder nodes are expected to be in the following order:
    # input_ids, kv_cache_key, kv_cache_value, start_idx, end_idx
    placeholder_nodes = [node for node in model.graph.nodes if node.op == "placeholder"]
    # The first two inputs are input_ids, position_ids. The last input is is_generate. In between are the KV cache tensors.
    kv_cache_inputs = placeholder_nodes[2:-1]
    zeroed_kv_cache_inputs = []
    for input in kv_cache_inputs:
        zeroed_kv_cache_inputs.append(
            torch.zeros(
                input.meta["val"].shape,
                dtype=input.meta["val"].dtype,
                device=torch.device("cuda:0"),
            )
        )

    return tuple(zeroed_kv_cache_inputs)


def generate(model, input_seq, max_output_seq_length, eos_token_id, benchmark=True):
    """
    Greedy decoding of the model. This generates up to max_tokens.
    """
    stopping_criteria = StoppingCriteriaList(
        [
            MaxLengthCriteria(max_length=max_output_seq_length),
            EosTokenCriteria(eos_token_id=eos_token_id),
        ]
    )
    isl = input_seq.shape[1]
    osl = max_output_seq_length - isl

    num_tokens_generated = 0
    while num_tokens_generated < osl:
        position_ids = torch.arange(input_seq.shape[1]).unsqueeze(0).cuda()
        outputs = model(input_seq, position_ids=position_ids)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        input_seq = torch.cat([input_seq, next_tokens[:, None]], dim=-1)
        num_tokens_generated += 1
        # TODO: Handle batch in this check
        if not benchmark and stopping_criteria(input_seq, logits).item():
            break

    return input_seq


def generate_with_static_cache(model, input_seq, max_output_seq_length, eos_token_id):
    """
    Greedy decoding of the model with static KV cache.
    """
    start_idx = 0
    end_idx = input_seq.shape[1]
    position_ids = torch.arange(input_seq.shape[1]).unsqueeze(0).cuda()
    output_seq = input_seq.clone()
    # TODO: Confirm this: When end_idx = max_output_seq_length-1, number of tokens generated = OSL
    num_tokens_generated = 0
    kv_cache = get_zeroed_static_cache_inputs(model)
    while end_idx < max_output_seq_length:
        position_ids = (
            torch.tensor([[start_idx]], dtype=torch.int64).cuda()
            if input_seq.shape[1] == 1
            else position_ids
        )
        input_signature = (input_seq, position_ids, *kv_cache, start_idx, end_idx)
        logits_keys_values = model(*input_signature)
        num_tokens_generated += 1
        logits = logits_keys_values[0]
        kv_cache = logits_keys_values[1:]
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        output_seq = torch.cat([output_seq, next_tokens], dim=-1)
        input_seq = next_tokens
        start_idx = end_idx
        end_idx = start_idx + 1
    return output_seq


def generate_with_dynamic_cache(model, input_seq, max_output_seq_length, eos_token_id):
    """
    Greedy decoding of the model with dynamic KV cache.
    """
    position_ids = torch.arange(input_seq.shape[1]).unsqueeze(0).cuda()
    output_seq = input_seq.clone()
    num_output_tokens = max_output_seq_length - input_seq.shape[1]
    num_tokens_generated = 0
    kv_cache = get_zeroed_dynamic_cache_inputs(model)
    last_position_id = position_ids[-1, -1].item()
    breakpoint()
    while num_tokens_generated < num_output_tokens:
        is_generate = False if input_seq.shape[1] > 1 else True
        position_ids = (
            torch.tensor([[last_position_id + 1]], dtype=torch.int64).cuda()
            if input_seq.shape[1] == 1
            else position_ids
        )
        input_signature = (input_seq, position_ids, *kv_cache, is_generate)
        logits_keys_values = model(*input_signature)
        num_tokens_generated += 1
        logits = logits_keys_values[0]
        kv_cache = logits_keys_values[1:]
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        output_seq = torch.cat([output_seq, next_tokens], dim=-1)
        input_seq = next_tokens
        last_position_id += 1
    return output_seq


def time_generate(
    generate_fn, model, inputs, output_seq_length, eos_token_id, iterations=10
):
    """
    Measure the time for generating a sentence over certain number of iterations
    """
    timings = []
    for _ in range(iterations):
        start_time = timeit.default_timer()
        _ = generate_fn(model, inputs, output_seq_length, eos_token_id)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)

    return timings


def record_stats(backend, timings, precision, batch_size=1, compile_time_s=None):
    """
    Records different timing stats and adds it to the result
    """
    times = np.array(timings)
    speeds = batch_size / times
    time_mean = np.mean(times).item()
    time_med = np.median(times).item()
    time_99th = np.percentile(times, 99).item()
    time_std = np.std(times, ddof=0).item()
    speed_mean = np.mean(speeds).item()
    speed_med = np.median(speeds).item()

    stats = {
        "Backend": backend,
        "Precision": precision,
        "Batch size": batch_size,
        "Median(FPS)": speed_med,
        "Mean(FPS)": speed_mean,
        "Median-Latency(ms)": time_med * 1000,
        "Mean-Latency(ms)": time_mean * 1000,
        "Latency-StdDev(ms)": time_std * 1000,
        "Compile Time(s)": compile_time_s,
    }
    return stats
