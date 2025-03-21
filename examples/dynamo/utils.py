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
        try:
            print("Trying to export the model using torch.export.export()..")
            # strict=False only enables aotautograd tracing and excludes dynamo.
            ep = torch.export.export(
                model, (inputs,), dynamic_shapes=({1: seq_len},), strict=False
            )
        except:
            print(
                "Trying torch.export._trace._export to trace the graph since torch.export.export() failed"
            )
            # This API is used to express the constraint violation guards as asserts in the graph.
            ep = torch.export._trace._export(
                model,
                (inputs,),
                dynamic_shapes=({1: seq_len},),
                strict=False,
                allow_complex_guards_as_runtime_asserts=True,
            )

    return ep


def generate(model, input_seq, max_tokens, eos_token_id, csi=None):
    """
    Greedy decoding of the model. This generates up to max_tokens.
    """
    # Max length of output seq = current input_seq length + max_tokens allowed to generate
    max_output_seq_length = input_seq.shape[1] + max_tokens
    stopping_criteria = StoppingCriteriaList(
        [
            MaxLengthCriteria(max_length=max_output_seq_length),
            EosTokenCriteria(eos_token_id=eos_token_id),
        ]
    )
    iter_time = []
    i = 0
    while i < max_tokens:
        start_time = timeit.default_timer()
        # print(f"=== Input shape: {input_seq.shape}" )
        outputs = model(input_seq)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        iter_time.append(end_time - start_time)
        # breakpoint()
        logits = outputs[0]  # .logits

        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        input_seq = torch.cat([input_seq, next_tokens[:, None]], dim=-1)
        # TODO: Handle batch in this check
        i += 1
        # if stopping_criteria(input_seq, logits).item():
        #     break
    # breakpoint()
    return input_seq, iter_time


def time_generate(
    generate_fn, model, inputs, output_seq_length, eos_token_id, csi=None, iterations=10
):
    """
    Measure the time for generating a sentence over certain number of iterations
    """
    timings = []
    for _ in range(iterations):
        start_time = timeit.default_timer()
        inputs_copy = copy.copy(inputs)
        _, iter_time = generate_fn(
            model, inputs_copy, output_seq_length, eos_token_id, csi=csi
        )
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)
    # breakpoint()

    return timings


def recordStats(backend, timings, precision, batch_size=1, compile_time_s=None):
    """
    Records different timing stats and adds it to the result
    """
    times = np.array(timings)
    speeds = batch_size / times
    time_mean = np.mean(times)
    time_med = np.median(times)
    time_99th = np.percentile(times, 99)
    time_std = np.std(times, ddof=0)
    speed_mean = np.mean(speeds)
    speed_med = np.median(speeds)

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
