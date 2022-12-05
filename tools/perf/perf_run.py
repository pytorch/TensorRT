from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import time
import timeit
import warnings
import numpy as np
import torch.backends.cudnn as cudnn

# Config parsers and report generations
import argparse
import yaml
import os
import pandas as pd

# Importing supported Backends
import torch
import torch_tensorrt as torchtrt
from torch_tensorrt.fx.lower import compile
from torch_tensorrt.fx.utils import LowerPrecision

import tensorrt as trt
from utils import (
    parse_inputs,
    parse_backends,
    precision_to_dtype,
    parse_precisions,
    BENCHMARK_MODELS,
)

WARMUP_ITER = 10
results = []

# YAML Parser class for parsing the run configurations
class ConfigParser:
    def __init__(self, config_file):
        self.parser = None
        self.config = config_file
        self.params = None

    # Reads and loads the yaml file
    def read_config(self):
        with open(self.config, "r") as stream:
            try:
                self.params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return self.params

    # Retrieves the value from the configuration else uses default values
    def get(self, key, default_value=None):
        if not key in self.params:
            if not default_value:
                raise ValueError(
                    "Key {} is not present and default_value is not configured. Please run it with default value",
                    key,
                )
            self.params[key] = default_value
        return self.params[key]


# Runs inference using Torch backend
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
def run_torch_tensorrt(
    model, input_tensors, params, precision, truncate_long_and_double, batch_size
):
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
        "truncate_long_and_double": truncate_long_and_double,
    }

    if precision == "int8":
        compile_settings.update({"calib": params.get("calibration_cache")})

    start_compile = time.time_ns()
    model = torchtrt.compile(model, **compile_settings)
    end_compile = time.time_ns()
    compile_time_ms = (end_compile - start_compile) / 1e6

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

    recordStats("Torch-TensorRT", timings, precision, batch_size, compile_time_ms)


# Runs inference using FX2TRT backend
def run_fx2trt(model, input_tensors, params, precision, batch_size):
    print("Running FX2TRT for precision: ", precision, " batch_size : ", batch_size)
    if precision == "fp32":
        precision = LowerPrecision.FP32
    elif precision == "fp16":
        precision = LowerPrecision.FP16
        model.half()
        input_tensors = [tensor.half() for tensor in input_tensors]
    # Run lowering eager mode benchmark
    start_compile = time.time_ns()
    model = compile(
        model,
        input_tensors,
        max_batch_size=batch_size,
        lower_precision=precision,
        verbose_log=False,
        explicit_batch_dimension=True,
    )
    end_compile = time.time_ns()
    compile_time_ms = (end_compile - start_compile) / 1e6

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

    recordStats("FX-TensorRT", timings, precision, batch_size, compile_time_ms)


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


def run_tensorrt(
    model,
    input_tensors,
    params,
    precision,
    truncate_long_and_double=False,
    is_trt_engine=False,
    batch_size=1,
):
    engine = None

    # If the model file is a TensorRT engine then directly deserialize and run inference
    # else convert the torch module to a TensorRT engine first and then run inference
    if not is_trt_engine:
        compile_settings = {
            "inputs": input_tensors,
            "enabled_precisions": {precision_to_dtype(precision)},
            "truncate_long_and_double": truncate_long_and_double,
        }

        print("Converting method to TensorRT engine...")
        with torch.no_grad(), torchtrt.logging.errors():
            model = torchtrt.ts.convert_method_to_trt_engine(
                model, "forward", **compile_settings
            )

    # Deserialize the TensorRT engine
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(model)

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
    truncate_long_and_double=False,
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
                or backend == "torch_tensorrt"
                or params.get("calibration_cache", None) == None
            ):
                print("int8 precision expects calibration cache file for inference")
                return False

        if backend == "all":
            run_torch(model, input_tensors, params, precision, batch_size)
            run_torch_tensorrt(
                model,
                input_tensors,
                params,
                precision,
                truncate_long_and_double,
                batch_size,
            )
            run_tensorrt(
                model,
                input_tensors,
                params,
                precision,
                truncate_long_and_double,
                is_trt_engine,
                batch_size,
            )

        elif backend == "torch":
            run_torch(model, input_tensors, params, precision, batch_size)

        elif backend == "torch_tensorrt":
            run_torch_tensorrt(
                model,
                input_tensors,
                params,
                precision,
                truncate_long_and_double,
                batch_size,
            )

        elif backend == "fx2trt":
            if model_torch is None:
                warnings.warn(
                    "Requested backend fx2trt without specifying a PyTorch Model, "
                    + "skipping this backend"
                )
                continue
            run_fx2trt(model_torch, input_tensors, params, precision, batch_size)

        elif backend == "tensorrt":
            run_tensorrt(
                model,
                input_tensors,
                params,
                precision,
                truncate_long_and_double,
                is_trt_engine,
                batch_size,
            )


# Generate report
def recordStats(backend, timings, precision, batch_size=1, compile_time_ms=None):
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
        "Compile Time(ms)": compile_time_ms,
    }
    results.append(stats)


def load_model(params):
    model = None
    is_trt_engine = False
    # Load torch model traced/scripted
    model_file = params.get("model").get("filename")
    try:
        model_name = params.get("model").get("name")
    except:
        model_name = model_file

    print("Loading model: ", model_file)
    if model_file.endswith(".plan"):
        is_trt_engine = True
        # Read the TensorRT engine file
        with open(model_file, "rb") as fin:
            model = fin.read()
    else:
        model = torch.jit.load(model_file).cuda()

    return model, model_name, is_trt_engine


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run inference on a model with random input values"
    )
    arg_parser.add_argument(
        "--config",
        type=str,
        help="Load YAML based configuration file to run the inference. If this is used other params will be ignored",
    )
    # The following options are manual user provided settings
    arg_parser.add_argument(
        "--backends",
        type=str,
        help="Comma separated string of backends. Eg: torch,torch_tensorrt,fx2trt,tensorrt",
    )
    arg_parser.add_argument("--model", type=str, help="Name of torchscript model file")
    arg_parser.add_argument(
        "--model_torch",
        type=str,
        default="",
        help="Name of torch model file (used for fx2trt)",
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
    arg_parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate long and double weights in the network  in Torch-TensorRT",
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

    cudnn.benchmark = True
    # Create random input tensor of certain size
    torch.manual_seed(12345)
    model_name = "Model"
    if args.config:
        parser = ConfigParser(args.config)
        # Load YAML params
        params = parser.read_config()
        model, model_name, is_trt_engine = load_model(params)

        # Default device is set to 0. Configurable using yaml config file.
        torch.cuda.set_device(params.get("runtime").get("device", 0))

        num_input = params.get("input").get("num_inputs")
        truncate_long_and_double = params.get("runtime").get(
            "truncate_long_and_double", False
        )
        batch_size = params.get("input").get("batch_size", 1)
        for precision in params.get("runtime").get("precision", "fp32"):
            input_tensors = []
            num_input = params.get("input").get("num_inputs", 1)
            for i in range(num_input):
                inp_tensor = params.get("input").get("input" + str(i))
                input_tensors.append(
                    torch.randint(
                        0,
                        2,
                        tuple(d for d in inp_tensor),
                        dtype=precision_to_dtype(precision),
                    ).cuda()
                )

            if is_trt_engine:
                print(
                    "Warning, TensorRT engine file is configured. Please make sure the precision matches with the TRT engine for reliable results"
                )

            if not is_trt_engine and (precision == "fp16" or precision == "half"):
                # If model is TensorRT serialized engine then model.half will report failure
                model = model.half()

            backends = params.get("backend")
            # Run inference
            status = run(
                model,
                backends,
                input_tensors,
                params,
                precision,
                truncate_long_and_double,
                batch_size,
                is_trt_engine,
            )
    else:
        params = vars(args)
        model_name = params["model"]
        model = None

        model_name_torch = params["model_torch"]
        model_torch = None

        # Load TorchScript model
        if os.path.exists(model_name):
            print("Loading user provided torchscript model: ", model_name)
            model = torch.jit.load(model_name).cuda().eval()
        elif model_name in BENCHMARK_MODELS:
            print("Loading torchscript model from BENCHMARK_MODELS for: ", model_name)
            model = BENCHMARK_MODELS[model_name]["model"].eval().cuda()
        else:
            raise ValueError(
                "Invalid model name. Please provide a torchscript model file or model name (among the following options vgg16|resnet50|efficientnet_b0|vit)"
            )

        # Load PyTorch Model, if provided
        if len(model_name_torch) > 0 and os.path.exists(model_name_torch):
            print("Loading user provided torch model: ", model_name_torch)
            model_torch = torch.load(model_name_torch).eval().cuda()

        backends = parse_backends(params["backends"])
        truncate_long_and_double = params["truncate"]
        batch_size = params["batch_size"]
        is_trt_engine = params["is_trt_engine"]
        precisions = parse_precisions(params["precision"])

        for precision in precisions:
            input_tensors = parse_inputs(
                params["inputs"], precision_to_dtype(precision)
            )
            if not is_trt_engine and (precision == "fp16" or precision == "half"):
                # If model is TensorRT serialized engine then model.half will report failure
                model = model.half()
            status = run(
                model,
                backends,
                input_tensors,
                params,
                precision,
                truncate_long_and_double,
                batch_size,
                is_trt_engine,
                model_torch=model_torch,
            )

    # Generate report
    print("Model Summary: ", model_name)
    summary = pd.DataFrame(results)
    print(summary)
    if args.report:
        with open(args.report, "w") as file:
            file.write("Model Summary: " + model_name + "\n")
            file.write(summary.to_string())
        file.close()
