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
from transformers import AutoModelForCausalLM, AutoTokenizer
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


def cannot_compile_callable(func):
    @wraps(func)
    def wrapper_func(*args, **kwargs):
        if isinstance(args[0], tuple):
            raise AssertionError(
                "This backend cannot handle callable/non-nn.Module inputs"
            )

        return func(*args, **kwargs)

    return wrapper_func


# Runs inference using Torch backend
@run_with_try_except
def run_torch(
    model_and_callable, params, precision, batch_size, model_name, *inputs, **kw_inputs
):
    print("Running Torch for precision: ", precision, " batch_size : ", batch_size)
    iters = params.get("iterations", 20)

    if isinstance(model_and_callable, tuple):
        _, callable_fn = model_and_callable
    else:
        callable_fn = model_and_callable

    # Warm up
    with torch.no_grad():
        for _ in range(WARMUP_ITER):
            features = callable_fn(*inputs, **kw_inputs)

    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(iters):
            start_time = timeit.default_timer()
            features = callable_fn(*inputs, **kw_inputs)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            timings.append(meas_time)

    recordStats(model_name, "Torch", timings, precision, batch_size)


# Runs inference using Torch-TensorRT backend
@run_with_try_except
@cannot_compile_callable
def run_ts_trt(model, params, precision, batch_size, model_name, *inputs, **kw_inputs):
    if kw_inputs:
        raise ValueError("Keyword inputs not currently supported in ir=ts path")

    print(
        "Running Torch-TensorRT for precision: ",
        precision,
        " batch_size : ",
        batch_size,
    )
    # Compiling Torch-TensorRT model
    compile_settings = {
        "inputs": inputs,
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
            features = model(*inputs)

    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(iters):
            start_time = timeit.default_timer()
            features = model(*inputs)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            timings.append(meas_time)

    recordStats(
        model_name,
        "Torch-TensorRT [Torchscript]",
        timings,
        precision,
        batch_size,
        compile_time_s,
    )


@run_with_try_except
def run_dynamo(
    model_and_callable, params, precision, batch_size, model_name, *inputs, **kw_inputs
):
    """
    Compile the given model using Torch-TensorRT dynamo frontend and record performance stats
    """
    # if kw_inputs:
    #     raise ValueError("Keyword inputs not currently supported in ir=dynamo path")

    if isinstance(model_and_callable, tuple):
        model, callable_fn = model_and_callable
    else:
        model = model_and_callable
        callable_fn = model_and_callable

    print(
        "Running Torch-TensorRT [dynamo] for precision: ",
        precision,
        " batch_size : ",
        batch_size,
    )
    # import pdb; pdb.set_trace()
    start_compile = time.time_ns()
    model = torchtrt.compile(
        model,
        inputs=inputs,
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
            features = model.generate(*inputs)

    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(iters):
            start_time = timeit.default_timer()
            features = model.generate(*inputs)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            timings.append(meas_time)

    recordStats(
        model_name,
        "Torch-TensorRT [Dynamo]",
        timings,
        precision,
        batch_size,
        compile_time_s,
    )


@run_with_try_except
def run_torch_compile(
    model_and_callable, params, precision, batch_size, model_name, *inputs, **kw_inputs
):
    """
    Compile the given model using Torch-TensorRT torch.compile frontend and record performance stats
    """
    torch._dynamo.reset()

    if isinstance(model_and_callable, tuple):
        model, callable_fn = model_and_callable
    else:
        model = model_and_callable
        callable_fn = model_and_callable

    print(
        "Running Torch-TensorRT [torch_compile] for precision: ",
        precision,
        " batch_size : ",
        batch_size,
    )
    compile_spec = {
        "enabled_precisions": {precision_to_dtype(precision)},
        "truncate_long_and_double": params.get("truncate", False),
        "min_block_size": params.get("min_block_size", 1),
        "debug": True,
    }
    start_compile = time.time_ns()
    model.forward = torch.compile(
        model.forward, backend="tensorrt", dynamic=False, options=compile_spec
    )
    callable_fn(*inputs, **kw_inputs)
    end_compile = time.time_ns()
    compile_time_s = (end_compile - start_compile) / 1e9
    iters = params.get("iterations", 20)
    # Warm up
    with torch.no_grad():
        for _ in range(WARMUP_ITER):
            features = callable_fn(*inputs, **kw_inputs)

    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(iters):
            start_time = timeit.default_timer()
            features = callable_fn(*inputs, **kw_inputs)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            timings.append(meas_time)
    # Reset torch dynamo cache
    torch._dynamo.reset()

    recordStats(
        model_name,
        "Torch-TensorRT [torch_compile]",
        timings,
        precision,
        batch_size,
        compile_time_s,
    )


@run_with_try_except
def run_inductor(
    model_and_callable, params, precision, batch_size, model_name, *inputs, **kw_inputs
):
    """
    Compile the given model using torch inductor and record performance stats
    """
    torch._dynamo.reset()

    if isinstance(model_and_callable, tuple):
        model, callable_fn = model_and_callable
    else:
        model = model_and_callable
        callable_fn = model_and_callable

    print(
        "Running Torch [inductor] for precision: ",
        precision,
        " batch_size : ",
        batch_size,
    )

    start_compile = time.time_ns()
    model.forward = torch.compile(model.forward, backend="inductor", dynamic=False)
    callable_fn(*inputs, **kw_inputs)
    end_compile = time.time_ns()
    compile_time_s = (end_compile - start_compile) / 1e9
    iters = params.get("iterations", 20)
    # Warm up
    with torch.no_grad():
        for _ in range(WARMUP_ITER):
            features = callable_fn(*inputs, **kw_inputs)

    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(iters):
            start_time = timeit.default_timer()
            features = callable_fn(*inputs, **kw_inputs)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            timings.append(meas_time)
    # Reset torch dynamo cache
    torch._dynamo.reset()

    recordStats(
        model_name,
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
@cannot_compile_callable
def run_tensorrt(
    model,
    params,
    precision,
    batch_size,
    model_name,
    *inputs,
    **kw_inputs,
):
    if kw_inputs:
        raise ValueError("Keyword inputs not currently supported in TRT-ONNX path")

    # Export an ONNX model and convert to TRT
    torch.onnx.export(model.eval().cuda(), tuple(inputs), "./tmp.onnx")
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
    start_compile = time.time_ns()
    serialized_engine = builder.build_serialized_network(network, config)
    end_compile = time.time_ns()
    compile_time_s = (end_compile - start_compile) / 1e9
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
            bindings[idx] = inputs[k].data_ptr()
            k += 1

    timings = []
    with engine.create_execution_context() as context:
        for _ in range(WARMUP_ITER):
            context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()

        for _ in range(iters):
            start_time = timeit.default_timer()
            context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            timings.append(meas_time)

    recordStats(model_name, "TensorRT", timings, precision, batch_size, compile_time_s)


# Deploys inference run for different backend configurations
def run(
    model_and_callable,
    backends,
    inputs,
    kw_inputs,
    params,
    precision,
    model_name,
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

        if (model_and_callable is None) and (backend in ("tensorrt", "ts_trt", "all")):
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
            run_torch(
                model_and_callable,
                params,
                precision,
                batch_size,
                model_name,
                *inputs,
                **kw_inputs,
            )
            run_ts_trt(
                model_and_callable,
                params,
                precision,
                batch_size,
                model_name,
                *inputs,
                **kw_inputs,
            )
            run_tensorrt(
                model_and_callable,
                inputs,
                params,
                precision,
                is_trt_engine,
                batch_size,
                model_name,
                *inputs,
                **kw_inputs,
            )
            run_dynamo(
                model_and_callable,
                params,
                precision,
                batch_size,
                model_name,
                *inputs,
                **kw_inputs,
            )

        elif backend == "torch":
            run_torch(
                model_torch if model_torch is not None else model_and_callable,
                params,
                precision,
                batch_size,
                model_name,
                *inputs,
                **kw_inputs,
            )

        elif backend == "ts_trt":
            run_ts_trt(
                model_and_callable,
                params,
                precision,
                batch_size,
                model_name,
                *inputs,
                **kw_inputs,
            )
        elif backend == "tensorrt":
            run_tensorrt(
                model_torch,
                params,
                precision,
                batch_size,
                model_name,
                *inputs,
                **kw_inputs,
            )
        elif backend == "dynamo":
            run_dynamo(
                model_torch if model_torch is not None else model_and_callable,
                params,
                precision,
                batch_size,
                model_name,
                *inputs,
                **kw_inputs,
            )

        elif backend == "torch_compile":
            run_torch_compile(
                model_torch if model_torch is not None else model_and_callable,
                params,
                precision,
                batch_size,
                model_name,
                *inputs,
                **kw_inputs,
            )

        elif backend == "inductor":
            run_inductor(
                model_torch if model_torch is not None else model_and_callable,
                params,
                precision,
                batch_size,
                model_name,
                *inputs,
                **kw_inputs,
            )


# Generate report
def recordStats(
    model_name, backend, timings, precision, batch_size=1, compile_time_s=None
):
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
        "Model Name": model_name,
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
        "--model_hf",
        type=str,
        default="",
        help="Name of HuggingFace model",
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
        help="List of input shapes. Eg: (1, 3, 224, 224)@fp32 for Resnet or (1, 128)@int32;(1, 128)@int32 for BERT. "
        "Or text for input to HuggingFace models",
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

    model_name_hf = params["model_hf"]
    model_hf = None

    precisions = parse_precisions(params["precision"])

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

    # Load HF Model, if provided
    if len(model_name_hf) > 0:
        print("Loading user-specified HF model: ", model_name_hf)
        model_hf = (
            AutoModelForCausalLM.from_pretrained(
                model_name_hf,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            .eval()
            .cuda()
        )
        tokenizer_hf = AutoTokenizer.from_pretrained(model_name_hf, padding_side="left")

    # If neither model type was provided
    if (model is None) and (model_torch is None) and (model_hf is None):
        raise ValueError(
            "No valid models specified. Please provide a torchscript model file or model name "
            + "(among the following options vgg16|resnet50|efficientnet_b0|vit) "
            + "or provide a torch model file"
        )

    backends = parse_backends(params["backends"])
    if ("dynamo" in backends or "torch_compile" in backends) and (
        model_torch is None and model_hf is None
    ):
        raise ValueError(
            "No Pytorch model (nn.Module) is provided for torchdynamo compilation. "
            "Please provide a pytorch model using --model_torch or --model_hf arguments"
        )

    batch_size = params["batch_size"]
    is_trt_engine = params["is_trt_engine"]
    is_hf = model_hf is not None

    for precision in precisions:
        inputs = parse_inputs(
            params["inputs"], precision_to_dtype(precision), is_hf=is_hf
        )
        use_fp16 = precision == "fp16" or precision == "half"

        if not is_trt_engine and use_fp16:
            # If model is TensorRT serialized engine then model.half will report failure
            if model is not None:
                model = model.half()
            if model_torch is not None:
                model_torch = model_torch.half()
            if model_hf is not None:
                model_hf = model_hf.half()
        elif model_hf is not None and not use_fp16:
            raise ValueError(
                "Compilation of HF models is currently restricted to FP16 for benchmarking"
            )

        with torch.no_grad():
            if is_hf:
                inputs = tuple([inputs[0]] * batch_size)
                tokenizer_hf.pad_token = tokenizer_hf.eos_token
                tokenizer_outputs = tokenizer_hf(
                    inputs,
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                model_inputs = tokenizer_outputs["input_ids"].cuda()
                attention_mask = tokenizer_outputs["attention_mask"].cuda()

                import pdb

                pdb.set_trace()
                # Run benchmarking for "generate" function separately
                for benchmark_fn in ["generate"]:
                    model_kwargs = {"attention_mask": attention_mask}
                    if benchmark_fn == "generate":
                        model_kwargs["min_new_tokens"] = 1
                        model_kwargs["max_new_tokens"] = 1

                    status = run(
                        (model_hf, getattr(model_hf, benchmark_fn)),
                        backends,
                        [model_inputs],
                        model_kwargs,
                        params,
                        precision,
                        model_name_hf + f"_{benchmark_fn}",
                        batch_size,
                        is_trt_engine,
                        model_torch=model_torch,
                    )

            else:
                status = run(
                    model,
                    backends,
                    inputs,
                    {},
                    params,
                    precision,
                    (model_name_torch if model_name_torch is not None else model_name),
                    batch_size,
                    is_trt_engine,
                    model_torch=model_torch,
                )

    # Generate report
    print("Model Summary: ", model_name)
    summary = pd.DataFrame(results)

    print(summary)
    if args.report:
        summary.to_csv(args.report)
