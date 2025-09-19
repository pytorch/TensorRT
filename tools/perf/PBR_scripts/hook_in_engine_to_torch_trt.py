import argparse
import os
import timeit

import numpy as np
import pandas as pd
import tensorrt as trt
import torch
import torch_tensorrt
from utils import BENCHMARK_MODELS, parse_inputs, precision_to_dtype

logger = trt.Logger(trt.Logger.WARNING)

WARMUP_ITER = 20
results = []


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


def record_perf(
    model,
    backend,
    input_tensors,
    precision,
    iterations,
    batch_size,
    compile_time_s=None,
):
    """
    Run the model for certain number of iterations and record the perf.
    Model is warmed up initially
    """
    # Warm up
    with torch.no_grad():
        for _ in range(WARMUP_ITER):
            model(*input_tensors)

    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for i in range(iterations):
            start_time = timeit.default_timer()
            _ = model(*input_tensors)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            timings.append(end_time - start_time)

    return recordStats(
        "Torch-TensorRT " + backend, timings, precision, batch_size, compile_time_s
    )


def hook_in_onnx_engine_to_torch_trt_runtime(
    model_name,
    engine_name,
    inputs,
    batch_size,
    precision,
    use_python_runtime,
    enable_cuda_graph,
    iterations,
    output_folder,
):

    model = BENCHMARK_MODELS[model_name]["model"].cuda().eval()
    if precision == "fp16" or precision == "half":
        model = model.half()
        truncate_double = True
    else:
        truncate_double = False
    dtype = precision_to_dtype(precision)
    input_tensors = parse_inputs(inputs, dtype)
    runtime_type = "python_runtime" if use_python_runtime else "cpp_runtime"

    # mimic compilation
    compilation_options = {
        "enabled_precisions": {dtype},
        "min_block_size": 1,
        "optimization_level": 1,
        "immutable_weights": True,
        "lazy_engine_init": True,
        "truncate_double": truncate_double,
        "use_python_runtime": use_python_runtime,
    }
    exp_program = torch.export.export(model, input_tensors)
    model = torch_tensorrt.dynamo.compile(
        exp_program, inputs=input_tensors, **compilation_options
    )
    # load trt engine from file and hook into torch_tensorrt runtime
    with open(engine_name, "rb") as f:
        saved_serialized_engine = f.read()

    # hook in the engine to the torch_tensorrt runtime
    for name, _ in model.named_children():
        trt_module = getattr(model, name)
        trt_module.engine = None
        trt_module.serialized_engine = saved_serialized_engine
        trt_module.setup_engine()
        setattr(model, name, trt_module)

    if enable_cuda_graph:
        with torch_tensorrt.runtime.enable_cudagraphs(model) as cudagraphs_module:
            result = record_perf(
                cudagraphs_module,
                f"{engine_name}+{runtime_type}",
                input_tensors,
                precision,
                iterations,
                batch_size,
            )
    else:
        result = record_perf(
            model,
            f"{engine_name}+{runtime_type}",
            input_tensors,
            precision,
            iterations,
            batch_size,
        )
    print(result)
    summary = pd.DataFrame([result])
    summary.insert(
        loc=0,
        column="model_name",
        value=model_name,
    )
    print(summary)
    log_name = os.path.join(
        output_folder,
        f"{engine_name}.perf_bs{batch_size}_{precision}_{runtime_type}.csv",
    )
    summary.to_csv(log_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--engine_name", type=str, required=True)
    parser.add_argument("--inputs", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--precision", type=str, required=True)
    parser.add_argument("--use_python_runtime", action="store_true")
    parser.add_argument("--enable_cuda_graph", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()

    hook_in_onnx_engine_to_torch_trt_runtime(
        args.model_name,
        args.engine_name,
        args.inputs,
        args.batch_size,
        args.precision,
        args.use_python_runtime,
        args.enable_cuda_graph,
        args.iterations,
        args.output_folder,
    )
