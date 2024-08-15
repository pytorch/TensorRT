from __future__ import absolute_import, division, print_function

# Config parsers and report generations
import argparse
import logging
import os
import time
import timeit
import warnings
from functools import wraps

import numpy as np
import pandas as pd
import tensorrt as trt

# Importing supported Backends
import torch
import torch_tensorrt as torchtrt
from utils import (
    BENCHMARK_MODELS,
    export_llm,
    parse_backends,
    parse_inputs,
    parse_precisions,
    precision_to_dtype,
    time_generate,
    torch_device_from_trt,
    torch_dtype_from_trt,
)

WARMUP_ITER = 10
results = []


def run_with_try_except(func):
    @wraps(func)
    def wrapper_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except:
            logging.warning(f"Running {func} failed", exc_info=True)

    return wrapper_func


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
    results.append(stats)


def record_llm_perf(
    model,
    backend,
    input_tensors,
    precision,
    output_seq_length,
    batch_size,
    iterations,
    compile_time_s=None,
):
    """
    Measure LLM generation time and record the stats
    """
    # We only support single input (B x seq_len) for LLMs now
    input_seq = input_tensors[0]
    with torch.no_grad():
        # Warm up for 3 iterations
        _ = time_generate(model, input_seq, output_seq_length, iterations=iterations)

        torch.cuda.synchronize()

        # Actual perf measurement
        timings = time_generate(
            model, input_seq, output_seq_length, iterations=iterations
        )

    recordStats(
        "Torch-TensorRT " + backend, timings, precision, batch_size, compile_time_s
    )


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

    recordStats(
        "Torch-TensorRT " + backend, timings, precision, batch_size, compile_time_s
    )


# Runs inference using Torch backend
@run_with_try_except
def run_torch(model, input_tensors, params, precision, batch_size):
    print("Running Torch for precision: ", precision, " batch_size : ", batch_size)
    iters = params.get("iterations", 20)
    model = model.to("cuda:0")
    if params["is_text_llm"]:
        output_seq_length = params["output_sequence_length"]
        return record_llm_perf(
            model,
            "Torch",
            input_tensors,
            precision,
            output_seq_length,
            batch_size,
            iters,
            None,
        )

    record_perf(
        model, "Torch", input_tensors, precision, iters, batch_size, compile_time_s=None
    )


# Runs inference using Torch-TensorRT backend
@run_with_try_except
def run_ts_trt(model, input_tensors, params, precision, batch_size):
    print(
        "Running Torch-TensorRT for precision: ",
        precision,
        " batch_size : ",
        batch_size,
    )
    # Compiling Torch-TensorRT model
    compile_settings = {
        "inputs": input_tensors,
        "enabled_precisions": {precision_to_dtype(precision)},
        "truncate_long_and_double": params.get("truncate", False),
    }

    if precision == "int8":
        compile_settings.update({"calib": params.get("calibration_cache")})

    start_compile = timeit.default_timer()
    model = torchtrt.compile(model, ir="ts", **compile_settings)
    end_compile = timeit.default_timer()
    compile_time_s = end_compile - start_compile

    iters = params.get("iterations", 20)

    record_perf(
        model,
        "Torchscript",
        input_tensors,
        precision,
        iters,
        batch_size,
        compile_time_s,
    )


@run_with_try_except
def run_hf_dynamo(model, input_tensors, params, precision, batch_size):
    """
    Compile the huggingface model using Torch-TensorRT dynamo frontend and record performance stats
    """

    osl = params["output_sequence_length"]
    iters = params.get("iterations", 20)
    # Move the model and inputs to cpu and trace it.
    model = model.to("cpu")
    inputs_cpu = [tensor.clone().cpu() for tensor in input_tensors]
    exp_program = export_llm(model, inputs_cpu, min_seq_len=1, max_seq_len=osl)
    start_compile = timeit.default_timer()

    trt_model = torchtrt.dynamo.compile(
        exp_program,
        inputs=input_tensors,
        enabled_precisions={precision_to_dtype(precision)},
        truncate_double=params.get("truncate", False),
    )
    end_compile = timeit.default_timer()
    compile_time_s = end_compile - start_compile
    record_llm_perf(
        trt_model,
        "Dynamo",
        input_tensors,
        precision,
        osl,
        batch_size,
        iters,
        compile_time_s,
    )


@run_with_try_except
def run_dynamo(model, input_tensors, params, precision, batch_size):
    """
    Compile the given model using Torch-TensorRT dynamo frontend and record performance stats
    """
    print(
        "Running Torch-TensorRT [dynamo] for precision: ",
        precision,
        " batch_size : ",
        batch_size,
    )
    if params["is_text_llm"]:
        return run_hf_dynamo(model, input_tensors, params, precision, batch_size)

    start_compile = timeit.default_timer()
    model = torchtrt.compile(
        model,
        inputs=input_tensors,
        ir="dynamo",
        enabled_precisions={precision_to_dtype(precision)},
        min_block_size=params.get("min_block_size", 1),
        debug=False,
        truncate_long_and_double=params.get("truncate", False),
    )
    end_compile = timeit.default_timer()
    compile_time_s = end_compile - start_compile
    iters = params.get("iterations", 20)

    record_perf(
        model, "Dynamo", input_tensors, precision, iters, batch_size, compile_time_s
    )


@run_with_try_except
def run_torch_compile(model, input_tensors, params, precision, batch_size):
    """
    Compile the given model using Torch-TensorRT torch.compile frontend and record performance stats
    """
    # Move the model to GPU
    model = model.to("cuda:0")
    torch._dynamo.reset()

    print(
        "Running Torch-TensorRT [torch_compile] for precision: ",
        precision,
        " batch_size : ",
        batch_size,
    )
    compile_spec = {
        "inputs": input_tensors,
        "enabled_precisions": {precision_to_dtype(precision)},
        "truncate": params.get("truncate", False),
        "min_block_size": params.get("min_block_size", 1),
    }
    start_compile = timeit.default_timer()
    model = torch.compile(model, backend="tensorrt", dynamic=None, options=compile_spec)
    model(*input_tensors)
    end_compile = timeit.default_timer()
    compile_time_s = end_compile - start_compile
    iters = params.get("iterations", 20)

    record_perf(
        model,
        "torch_compile",
        input_tensors,
        precision,
        iters,
        batch_size,
        compile_time_s,
    )


@run_with_try_except
def run_hf_inductor(model, input_tensors, params, precision, batch_size):
    """
    Compile the huggingface model using torch inductor and record performance stats
    """
    osl = params["output_sequence_length"]
    # Mark dynamic shapes for input sequence
    input_seq = input_tensors[0]
    torch._dynamo.mark_dynamic(input_seq, 1, min=1, max=osl)
    start_compile = timeit.default_timer()
    # Compile the model
    model = torch.compile(model, backend="inductor", dynamic=None, mode="max-autotune")
    model(input_seq)
    end_compile = timeit.default_timer()
    compile_time_s = end_compile - start_compile
    iters = params.get("iterations", 20)

    record_llm_perf(
        model,
        "Inductor",
        input_tensors,
        precision,
        osl,
        batch_size,
        iters,
        compile_time_s,
    )


@run_with_try_except
def run_inductor(model, input_tensors, params, precision, batch_size):
    """
    Compile the given model using torch inductor and record performance stats
    """
    torch._dynamo.reset()
    model = model.to("cuda:0")
    print(
        "Running Torch [inductor] for precision: ",
        precision,
        " batch_size : ",
        batch_size,
    )
    if params["is_text_llm"]:
        return run_hf_inductor(model, input_tensors, params, precision, batch_size)

    start_compile = timeit.default_timer()
    model = torch.compile(model, backend="inductor", dynamic=None, mode="max-autotune")
    model(*input_tensors)
    end_compile = timeit.default_timer()
    compile_time_s = end_compile - start_compile
    iters = params.get("iterations", 20)

    record_perf(
        model, "inductor", input_tensors, precision, iters, batch_size, compile_time_s
    )


@run_with_try_except
def run_tensorrt(
    model,
    input_tensors,
    params,
    precision,
    batch_size=1,
):
    # Export an ONNX model and convert to TRT
    torch.onnx.export(model.eval().cuda(), tuple(input_tensors), "./tmp.onnx")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file("./tmp.onnx")
    if not success:
        raise ValueError("ONNX conversion failed")

    config = builder.create_builder_config()
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    start_compile = timeit.default_timer()
    serialized_engine = builder.build_serialized_network(network, config)
    end_compile = timeit.default_timer()
    compile_time_s = end_compile - start_compile
    # Deserialize the TensorRT engine
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(serialized_engine)

    print("Running TensorRT for precision: ", precision, " batch_size : ", batch_size)
    iters = params.get("iterations", 20)

    # Compiling the bindings
    bindings = engine.num_bindings * [None]
    k = 0
    for idx, _ in enumerate(bindings):
        dtype = torch_dtype_from_trt(engine.get_binding_dtype(idx))
        shape = tuple(engine.get_binding_shape(idx))
        device = torch_device_from_trt(engine.get_location(idx))
        if not engine.binding_is_input(idx):
            # Output bindings
            output = torch.empty(size=shape, dtype=dtype, device=device)
            bindings[idx] = output.data_ptr()
        else:
            # Input bindings
            bindings[idx] = input_tensors[k].data_ptr()
            k += 1

    timings = []
    with engine.create_execution_context() as context:
        for i in range(WARMUP_ITER):
            context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()

        for i in range(iters):
            start_time = timeit.default_timer()
            context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            timings.append(meas_time)

    recordStats("TensorRT", timings, precision, batch_size, compile_time_s)


# Deploys inference run for different backend configurations
def run(
    model,
    backends,
    input_tensors,
    params,
    precision,
    batch_size=1,
    is_trt_engine=False,
    model_torch=None,
):
    for backend in backends:
        if precision == "int8":
            if backend == "all" or backend == "torch":
                print(
                    "int8 precision is not supported for torch runtime in this script yet"
                )
                return False

            if (
                backend == "all"
                or backend == "ts_trt"
                or params.get("calibration_cache", None) == None
            ):
                print("int8 precision expects calibration cache file for inference")
                return False

        if (model is None) and (backend in ("tensorrt", "ts_trt", "all")):
            warnings.warn(
                f"Requested backend {backend} without specifying a TorchScript Model, "
                + "skipping this backend"
            )
            continue

        if (model_torch is None) and (backend in ("all", "fx2trt")):
            warnings.warn(
                f"Requested backend {backend} without specifying a PyTorch Model, "
                + "skipping this backend"
            )
            continue

        if backend == "all":
            run_torch(model, input_tensors, params, precision, batch_size)
            run_ts_trt(
                model,
                input_tensors,
                params,
                precision,
                batch_size,
            )
            run_tensorrt(
                model,
                input_tensors,
                params,
                precision,
                is_trt_engine,
                batch_size,
            )
            run_dynamo(model_torch, input_tensors, params, precision, batch_size)

        elif backend == "torch":
            run_torch(model_torch, input_tensors, params, precision, batch_size)

        elif backend == "ts_trt":
            run_ts_trt(
                model,
                input_tensors,
                params,
                precision,
                batch_size,
            )
        elif backend == "tensorrt":
            run_tensorrt(
                model_torch,
                input_tensors,
                params,
                precision,
                batch_size,
            )
        elif backend == "dynamo":
            run_dynamo(model_torch, input_tensors, params, precision, batch_size)

        elif backend == "torch_compile":
            run_torch_compile(model_torch, input_tensors, params, precision, batch_size)

        elif backend == "inductor":
            run_inductor(model_torch, input_tensors, params, precision, batch_size)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run inference on a model with random input values"
    )
    # The following options are manual user provided settings
    arg_parser.add_argument(
        "--backends",
        type=str,
        help="Comma separated string of backends. Eg: torch, ts_trt, dynamo, torch_compile, inductor, tensorrt",
    )
    arg_parser.add_argument(
        "--model", type=str, default="", help="Name of torchscript model file"
    )
    arg_parser.add_argument(
        "--model_torch",
        type=str,
        default="",
        help="Name of torch model file",
    )
    arg_parser.add_argument(
        "--inputs",
        type=str,
        help="List of input shapes. Eg: (1, 3, 224, 224)@fp32 for Resnet or (1, 128)@int32;(1, 128)@int32 for BERT",
    )
    arg_parser.add_argument(
        "--is_text_llm",
        action="store_true",
        help="Boolean flag to determine if model is a huggingface model",
    )
    arg_parser.add_argument(
        "-osl",
        "--output_sequence_length",
        type=int,
        help="Length of output sequence to HF model",
        default=128,
    )
    arg_parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size to build and run"
    )
    arg_parser.add_argument(
        "--iterations", type=int, default=20, help="Iterations to measure the perf"
    )
    arg_parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        help="Comma separated list of precisions to build TensorRT engine Eg: fp32,fp16",
    )
    arg_parser.add_argument(
        "--calibration_cache", type=str, help="Name of the calibration cache file"
    )
    arg_parser.add_argument("--device", type=int, help="device id")
    arg_parser.add_argument("--min_block_size", type=int, default=1, help="device id")
    arg_parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate long and double weights in the network in Torch-TensorRT",
    )
    arg_parser.add_argument(
        "--is_trt_engine",
        action="store_true",
        help="Boolean flag to determine if the user provided model is a TRT engine or not",
    )
    arg_parser.add_argument(
        "--report",
        type=str,
        help="Path of the output file where performance summary is written.",
    )
    args = arg_parser.parse_args()

    # Create random input tensor of certain size
    torch.manual_seed(12345)
    model_name = "Model"
    params = vars(args)
    model_name = params["model"]
    model = None

    model_name_torch = params["model_torch"]
    model_torch = None

    # Load TorchScript model, if provided
    if os.path.exists(model_name):
        print("Loading user provided torchscript model: ", model_name)
        model = torch.jit.load(model_name).cuda().eval()

    # Load PyTorch Model, if provided
    if len(model_name_torch) > 0 and os.path.exists(model_name_torch):
        print("Loading user provided torch model: ", model_name_torch)
        model_torch = torch.load(model_name_torch).eval()
    elif model_name_torch in BENCHMARK_MODELS:
        model_torch = BENCHMARK_MODELS[model_name_torch]["model"].eval()

    # If neither model type was provided
    if (model is None) and (model_torch is None):
        raise ValueError(
            "No valid models specified. Please provide a torchscript model file or model name (defined in hub.py) or model_hf name in huggingface models "
        )

    backends = parse_backends(params["backends"])
    if ("dynamo" in backends or "torch_compile" in backends) and (model_torch is None):
        raise ValueError(
            "No Pytorch model (nn.Module) is provided for torchdynamo compilation. Please provide a pytorch model using --model_torch argument"
        )

    batch_size = params["batch_size"]
    is_trt_engine = params["is_trt_engine"]
    precisions = parse_precisions(params["precision"])

    for precision in precisions:
        input_tensors = parse_inputs(params["inputs"], precision_to_dtype(precision))

        if not is_trt_engine and (precision == "fp16" or precision == "half"):
            # If model is TensorRT serialized engine then model.half will report failure
            if model is not None:
                model = model.half()
            if model_torch is not None:
                model_torch = model_torch.half()

        with torch.no_grad():
            status = run(
                model,
                backends,
                input_tensors,
                params,
                precision,
                batch_size,
                is_trt_engine,
                model_torch=model_torch,
            )

    # Generate report
    print("Model Summary: ", model_name)
    summary = pd.DataFrame(results)
    summary.insert(
        loc=0,
        column="model_name",
        value=(model_name_torch if model_name_torch is not None else model_name),
    )
    print(summary)
    if args.report:
        summary.to_csv(args.report)
