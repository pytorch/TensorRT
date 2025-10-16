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

# Importing supported Backends
import torch
import torch_tensorrt as torchtrt
from torch_tensorrt.dynamo import convert_exported_program_to_serialized_trt_engine
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
SUPPORTED_BACKENDS = [
    "all",
    "torch",
    "ts_trt",
    "dynamo",
    "torch_compile",
    "inductor",
    "onnx_trt",
]


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
        # Warm up
        _ = time_generate(model, input_seq, output_seq_length, iterations=WARMUP_ITER)

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
    }

    if precision == "int8":
        compile_settings.update({"calib": params.get("calibration_cache")})

    if params.get("enable_cuda_graph", False):
        logging.warning(
            f"Torchscript backend doesn't support CUDA Graphs. `--enable_cuda_graph` will be ignored."
        )

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

    compilation_options = {
        "enabled_precisions": {precision_to_dtype(precision)},
        "min_block_size": params.get("min_block_size", 1),
        "truncate_double": params.get("truncate", False),
        "immutable_weights": params.get("immutable_weights", True),
        "strip_engine_weights": params.get("strip_engine_weights", False),
        "refit_identical_engine_weights": params.get(
            "refit_identical_engine_weights", False
        ),
        "cache_built_engines": params.get("cache_built_engines", False),
        "reuse_cached_engines": params.get("reuse_cached_engines", False),
        "use_python_runtime": params.get("use_python_runtime", False),
        "optimization_level": params.get("optimization_level", 3),
    }
    start_compile = timeit.default_timer()
    exp_program = export_llm(model, input_tensors, min_seq_len=1, max_seq_len=osl)
    trt_model = torchtrt.dynamo.compile(
        exp_program, inputs=input_tensors, **compilation_options
    )
    end_compile = timeit.default_timer()
    compile_time_s = end_compile - start_compile

    if params.get("save_dynamo_trt_engine", False):
        serialized_engine = convert_exported_program_to_serialized_trt_engine(
            exp_program,
            arg_inputs=input_tensors,
            **compilation_options,
        )
        with open(f"{params['model_torch']}-dynamo-trt-engine.trt", "wb") as f:
            f.write(serialized_engine)

    if params.get("enable_cuda_graph", False):
        with torchtrt.runtime.enable_cudagraphs(trt_model) as cudagraphs_module:
            record_llm_perf(
                cudagraphs_module,
                "Dynamo",
                input_tensors,
                precision,
                osl,
                batch_size,
                iters,
                compile_time_s,
            )
    else:
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

    compilation_options = {
        "enabled_precisions": {precision_to_dtype(precision)},
        "min_block_size": params.get("min_block_size", 1),
        "truncate_double": params.get("truncate", False),
        "immutable_weights": params.get("immutable_weights", True),
        "strip_engine_weights": params.get("strip_engine_weights", False),
        "refit_identical_engine_weights": params.get(
            "refit_identical_engine_weights", False
        ),
        "cache_built_engines": params.get("cache_built_engines", False),
        "reuse_cached_engines": params.get("reuse_cached_engines", False),
        "use_python_runtime": params.get("use_python_runtime", False),
        "optimization_level": params.get("optimization_level", None),
    }
    start_compile = timeit.default_timer()
    exp_program = torch.export.export(model, input_tensors)
    model = torchtrt.dynamo.compile(
        exp_program, inputs=input_tensors, **compilation_options
    )
    end_compile = timeit.default_timer()
    compile_time_s = end_compile - start_compile
    iters = params.get("iterations", 20)

    if params.get("save_dynamo_trt_engine", False):
        serialized_engine = convert_exported_program_to_serialized_trt_engine(
            exp_program,
            arg_inputs=input_tensors,
            **compilation_options,
        )
        with open(f"{params['model_torch']}-dynamo-trt-engine.trt", "wb") as f:
            f.write(serialized_engine)

    if params.get("enable_cuda_graph", False):
        with torchtrt.runtime.enable_cudagraphs(model) as cudagraphs_module:
            record_perf(
                cudagraphs_module,
                "Dynamo",
                input_tensors,
                precision,
                iters,
                batch_size,
                compile_time_s,
            )
    else:
        record_perf(
            model, "Dynamo", input_tensors, precision, iters, batch_size, compile_time_s
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
        "truncate": params.get("truncate", False),
        "min_block_size": params.get("min_block_size", 1),
        "use_python_runtime": params.get("use_python_runtime", False),
    }
    start_compile = timeit.default_timer()
    model = torch.compile(model, backend="tensorrt", dynamic=None, options=compile_spec)
    model(*input_tensors)
    end_compile = timeit.default_timer()
    compile_time_s = end_compile - start_compile
    iters = params.get("iterations", 20)

    if params.get("enable_cuda_graph", False):
        with torchtrt.runtime.enable_cudagraphs(model) as cudagraphs_module:
            record_perf(
                cudagraphs_module,
                "torch_compile",
                input_tensors,
                precision,
                iters,
                batch_size,
                compile_time_s,
            )
    else:
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
    torch._dynamo.reset()

    osl = params["output_sequence_length"]
    # Mark dynamic shapes for input sequence
    input_seq = input_tensors[0]
    torch._dynamo.mark_dynamic(input_seq, 1, min=1, max=osl)
    mode = "max-autotune"
    if params.get("enable_cuda_graph", False):
        mode = "reduce-overhead"

    start_compile = timeit.default_timer()
    # Compile the model
    model = torch.compile(model, backend="inductor", dynamic=None, mode=mode)
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

    print(
        "Running Torch [inductor] for precision: ",
        precision,
        " batch_size : ",
        batch_size,
    )
    if params["is_text_llm"]:
        return run_hf_inductor(model, input_tensors, params, precision, batch_size)

    mode = "max-autotune"
    if params.get("enable_cuda_graph", False):
        mode = "reduce-overhead"

    start_compile = timeit.default_timer()
    model = torch.compile(model, backend="inductor", dynamic=None, mode=mode)
    model(*input_tensors)
    end_compile = timeit.default_timer()
    compile_time_s = end_compile - start_compile
    iters = params.get("iterations", 20)

    record_perf(
        model,
        "inductor",
        input_tensors,
        precision,
        iters,
        batch_size,
        compile_time_s,
    )


@run_with_try_except
def run_onnx_trt(
    model,
    input_tensors,
    params,
    precision,
    batch_size=1,
):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    compile_time_s = 0
    if params["is_trt_engine"]:
        serialized_engine = model
    else:
        if params["onnx"]:
            onnx_path = params["onnx"]
        else:
            onnx_path = f"{params['model_torch']}-onnx-trt.onnx"
            len_output = len(model(*input_tensors))
            # to match the output names with Torch-TRT engine's
            torch.onnx.export(
                model,
                tuple(input_tensors),
                onnx_path,
                dynamo=True,
                output_names=[f"output{i}" for i in range(len_output)],
            )
        start_compile = timeit.default_timer()
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        success = parser.parse_from_file(onnx_path)
        if not success:
            raise ValueError("ONNX conversion failed")

        config = builder.create_builder_config()
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        config.builder_optimization_level = params.get("optimization_level", 3)

        if params.get("immutable_weights", True):
            # non-refittable engine
            if params.get("strip_engine_weights", False):
                warnings.warn("strip_engine_weights will be ignored.")
            if params.get("refit_identical_engine_weights", False):
                warnings.warn("refit_identical_engine_weights will be ignored.")
        else:
            # refittable engine
            if params.get("refit_identical_engine_weights", False):
                config.set_flag(trt.BuilderFlag.REFIT_IDENTICAL)
            else:
                config.set_flag(trt.BuilderFlag.REFIT)

            if params.get("strip_engine_weights", False):
                config.set_flag(trt.BuilderFlag.STRIP_PLAN)

        serialized_engine = builder.build_serialized_network(network, config)
        end_compile = timeit.default_timer()
        compile_time_s = end_compile - start_compile

    # Deserialize the TensorRT engine
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(serialized_engine)

    # save the generated TRT engine
    if params.get("save_onnx_trt_engine", False):
        with open(f"{params['model_torch']}-onnx-trt-engine.trt", "wb") as f:
            f.write(serialized_engine)

    print(
        "Running ONNX-TensorRT for precision: ", precision, " batch_size : ", batch_size
    )
    iters = params.get("iterations", 20)

    # Get I/O tensor information using TensorRT 10 API
    input_names = []
    output_names = []
    output_dtypes = []
    output_shapes = []

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_mode = engine.get_tensor_mode(tensor_name)
        tensor_dtype = engine.get_tensor_dtype(tensor_name)
        tensor_shape = engine.get_tensor_shape(tensor_name)

        if tensor_mode == trt.TensorIOMode.INPUT:
            input_names.append(tensor_name)
        else:  # trt.TensorIOMode.OUTPUT
            output_names.append(tensor_name)
            output_dtypes.append(torch_dtype_from_trt(tensor_dtype))
            output_shapes.append(tuple(tensor_shape))

    # Create output tensors
    output_tensors = []
    for i, (shape, dtype) in enumerate(zip(output_shapes, output_dtypes)):
        output = torch.empty(size=shape, dtype=dtype, device="cuda")
        output_tensors.append(output)

    timings = []
    with engine.create_execution_context() as context:
        # Set input tensor addresses
        for i, (input_name, input_tensor) in enumerate(zip(input_names, input_tensors)):
            context.set_tensor_address(input_name, input_tensor.data_ptr())

        # Set output tensor addresses
        for output_name, output_tensor in zip(output_names, output_tensors):
            context.set_tensor_address(output_name, output_tensor.data_ptr())

        # Create a dedicated stream for TensorRT execution
        dedicated_stream = torch.cuda.Stream()
        current_stream = torch.cuda.current_stream()

        # Warm up
        for i in range(WARMUP_ITER):
            # Wait for current stream to finish
            dedicated_stream.wait_stream(current_stream)
            context.execute_async_v3(dedicated_stream.cuda_stream)
            # Wait for TensorRT stream to finish
            current_stream.wait_stream(dedicated_stream)
            torch.cuda.synchronize()

        # Performance measurement
        for i in range(iters):
            infer_start_time = timeit.default_timer()
            # Wait for current stream to finish
            dedicated_stream.wait_stream(current_stream)
            context.execute_async_v3(dedicated_stream.cuda_stream)
            # Wait for TensorRT stream to finish
            current_stream.wait_stream(dedicated_stream)
            torch.cuda.synchronize()
            infer_end_time = timeit.default_timer()
            timings.append(infer_end_time - infer_start_time)

    recordStats("ONNX-TensorRT", timings, precision, batch_size, compile_time_s)


# Deploys inference run for different backend configurations
def run(
    model,
    backends,
    input_tensors,
    params,
    precision,
    batch_size=1,
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

        if (model is None) and (backend in ("ts_trt", "all")):
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
            run_onnx_trt(
                model_torch,
                input_tensors,
                params,
                precision,
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
        elif backend == "onnx_trt":
            run_onnx_trt(
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
        help="Comma separated string of backends. Eg: torch, ts_trt, dynamo, torch_compile, inductor, onnx_trt",
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
        "--onnx",
        type=str,
        default="",
        help="ONNX model file which helps bypass the step of exporting ONNX from torchscript model. If this argument is provided, the ONNX will be directly converted to TRT engine",
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
        "--optimization_level",
        type=int,
        default=3,
        help="Builder optimization level for TensorRT",
    )
    arg_parser.add_argument(
        "--is_trt_engine",
        action="store_true",
        help="Boolean flag to determine if the user provided model is a TRT engine or not",
    )
    arg_parser.add_argument(
        "--use_python_runtime",
        action="store_true",
        help="Whether to use Python runtime or not. Using C++ runtime by default",
    )
    arg_parser.add_argument(
        "--enable_cuda_graph",
        action="store_true",
        help="Whether to enable CUDA Graph. It is not used by default",
    )
    arg_parser.add_argument(
        "--report",
        type=str,
        help="Path of the output file where performance summary is written.",
    )
    arg_parser.add_argument(
        "--immutable_weights",
        action="store_true",
        help="Build non-refittable engines. This is useful for some layers that are not refittable. If this argument is set to true, `strip_engine_weights` and `refit_identical_engine_weights` will be ignored.",
    )
    arg_parser.add_argument(
        "--strip_engine_weights",
        action="store_true",
        help="Strip engine weights from the serialized engine. This is useful when the engine is to be deployed in an environment where the weights are not required.",
    )
    arg_parser.add_argument(
        "--refit_identical_engine_weights",
        action="store_true",
        help="Refit engines with identical weights. This is useful when the same model is compiled multiple times with different inputs and the weights are the same. This will save time by reusing the same engine for different inputs.",
    )
    arg_parser.add_argument(
        "--cache_built_engines",
        action="store_true",
        help="Whether to save the compiled TRT engines to storage.",
    )
    arg_parser.add_argument(
        "--reuse_cached_engines",
        action="store_true",
        help="Whether to load the compiled TRT engines from storage.",
    )
    arg_parser.add_argument(
        "--save_onnx_trt_engine",
        action="store_true",
        help="Whether to save the ONNX-TRT backend generated TRT engine.",
    )
    arg_parser.add_argument(
        "--save_dynamo_trt_engine",
        action="store_true",
        help="Whether to save the Torch-TRT Dynamo backend generated TRT engine.",
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
        if params["is_trt_engine"]:
            with open(model_name, "rb") as f:
                model = f.read()
                print("Loading user provided trt engine: ", model_name)
        else:
            print("Loading user provided torchscript model: ", model_name)
            model = torch.jit.load(model_name).cuda().eval()

    # Load PyTorch Model, if provided
    if len(model_name_torch) > 0 and os.path.exists(model_name_torch):
        print("Loading user provided torch model: ", model_name_torch)
        model_torch = torch.load(model_name_torch).cuda().eval()
    elif model_name_torch in BENCHMARK_MODELS:
        model_torch = BENCHMARK_MODELS[model_name_torch]["model"].cuda().eval()

    # If neither model type was provided
    if (model is None) and (model_torch is None):
        raise ValueError(
            "No valid models specified. Please provide a torchscript model file or model name (defined in hub.py) or model_hf name in huggingface models "
        )

    backends = parse_backends(params["backends"])
    for backend in backends:
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Backend {backend} is not supported. Please provide a valid backend."
            )
        if backend in ["dynamo", "torch_compile", "onnx_trt", "all"] and (
            model_torch is None
        ):
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
