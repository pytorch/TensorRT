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
    parse_backends,
    parse_inputs,
    parse_precisions,
    precision_to_dtype,
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


# Runs inference using Torch backend
@run_with_try_except
def run_torch(model, input_tensors, params, precision, batch_size):
    print("Running Torch for precision: ", precision, " batch_size : ", batch_size)
    iters = params.get("iterations", 20)

    # Warm up
    with torch.no_grad():
        for _ in range(WARMUP_ITER):
            features = model(*input_tensors)

    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for i in range(iters):
            start_time = timeit.default_timer()
            features = model(*input_tensors)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            timings.append(meas_time)

    recordStats("Torch", timings, precision, batch_size)


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

    start_compile = time.time_ns()
    model = torchtrt.compile(model, ir="ts", **compile_settings)
    end_compile = time.time_ns()
    compile_time_s = (end_compile - start_compile) / 1e9

    iters = params.get("iterations", 20)
    # Warm up
    with torch.no_grad():
        for _ in range(WARMUP_ITER):
            features = model(*input_tensors)

    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for i in range(iters):
            start_time = timeit.default_timer()
            features = model(*input_tensors)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            timings.append(meas_time)

    recordStats(
        "Torch-TensorRT [Torchscript]", timings, precision, batch_size, compile_time_s
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
    start_compile = time.time_ns()
    model = torchtrt.compile(
        model,
        inputs=input_tensors,
        ir="dynamo",
        enabled_precisions={precision_to_dtype(precision)},
        min_block_size=params.get("min_block_size", 1),
        debug=False,
        truncate_long_and_double=params.get("truncate", False),
    )
    end_compile = time.time_ns()
    compile_time_s = (end_compile - start_compile) / 1e9
    iters = params.get("iterations", 20)
    # Warm up
    with torch.no_grad():
        for _ in range(WARMUP_ITER):
            features = model(*input_tensors)

    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for i in range(iters):
            start_time = timeit.default_timer()
            features = model(*input_tensors)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            timings.append(meas_time)

    recordStats(
        "Torch-TensorRT [Dynamo]", timings, precision, batch_size, compile_time_s
    )


@run_with_try_except
def run_torch_compile(model, input_tensors, params, precision, batch_size):
    """
    Compile the given model using Torch-TensorRT torch.compile frontend and record performance stats
    """
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
        "truncate_long_and_double": params.get("truncate", False),
        "min_block_size": params.get("min_block_size", 1),
    }
    start_compile = time.time_ns()
    model = torch.compile(
        model, backend="tensorrt", dynamic=False, options=compile_spec
    )
    model(*input_tensors)
    end_compile = time.time_ns()
    compile_time_s = (end_compile - start_compile) / 1e9
    iters = params.get("iterations", 20)
    # Warm up
    with torch.no_grad():
        for _ in range(WARMUP_ITER):
            features = model(*input_tensors)

    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for i in range(iters):
            start_time = timeit.default_timer()
            features = model(*input_tensors)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            timings.append(meas_time)
    # Reset torch dynamo cache
    torch._dynamo.reset()

    recordStats(
        "Torch-TensorRT [torch_compile]",
        timings,
        precision,
        batch_size,
        compile_time_s,
    )


@run_with_try_except
def run_inductor(model, input_tensors, params, precision, batch_size):
    """
    Compile the given model using torch inductor and record performance stats
    """
    torch._dynamo.reset()

    print(
        "Running Torch [inductor] for precision: ",
        precision,
        " batch_size : ",
        batch_size,
    )

    start_compile = time.time_ns()
    model = torch.compile(model, backend="inductor", dynamic=False, mode="max-autotune")
    model(*input_tensors)
    end_compile = time.time_ns()
    compile_time_s = (end_compile - start_compile) / 1e9
    iters = params.get("iterations", 20)
    # Warm up
    with torch.no_grad():
        for _ in range(WARMUP_ITER):
            features = model(*input_tensors)

    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for i in range(iters):
            start_time = timeit.default_timer()
            features = model(*input_tensors)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            timings.append(meas_time)
    # Reset torch dynamo cache
    torch._dynamo.reset()

    recordStats(
        "Torch [inductor]",
        timings,
        precision,
        batch_size,
        compile_time_s,
    )


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)


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
    serialized_engine = builder.build_serialized_network(network, config)
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

    recordStats("TensorRT", timings, precision, batch_size)


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


# Generate report
def recordStats(backend, timings, precision, batch_size=1, compile_time_s=None):
    times = np.array(timings)
    steps = len(times)
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
        "--batch_size", type=int, default=1, help="Batch size to build and run"
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
        model_torch = torch.load(model_name_torch).eval().cuda()
    elif model_name_torch in BENCHMARK_MODELS:
        model_torch = BENCHMARK_MODELS[model_name_torch]["model"].eval().cuda()

    # If neither model type was provided
    if (model is None) and (model_torch is None):
        raise ValueError(
            "No valid models specified. Please provide a torchscript model file or model name "
            + "(among the following options vgg16|resnet50|efficientnet_b0|vit) "
            + "or provide a torch model file"
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
