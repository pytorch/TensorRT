from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import timeit
import numpy as np
import torch.backends.cudnn as cudnn
import yaml
import os
import pandas as pd

# Backend
import torch
import torch_tensorrt as torchtrt
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

results = []

def run_torch(model, input_tensors, params, precision):
    print("Running Torch for precision: ", precision)

    iters = 20 if not "iterations" in params else params['iterations']

    # Warm up
    with torch.no_grad():
        for _ in range(20):
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
            print("Iteration {}: {:.6f} s".format(i, end_time - start_time))
    
    printStats("Torch", timings, precision)

def onnx_to_trt_engine(onnx_model, precision):
    
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
        config.max_workspace_size = 1 << 28 # 256MiB
        builder.max_batch_size = 1

        if precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
        elif precision == 'fp16' or precision == 'half':
            config.set_flag(trt.BuilderFlag.HALF)

        plan = builder.build_serialized_network(network, config)
        model = runtime.deserialize_cuda_engine(plan)
    return model

def run_torch_tensorrt(model, input_tensors, params, precision):
    print("Running Torch-TensorRT")
    
    # Compiling Torch-TensorRT model
    compile_settings = {
       "inputs": input_tensors,
       "enabled_precisions": {precision_to_dtype(precision)} 
    }
    
    model = torchtrt.compile(model, **compile_settings)
 
    iters = 20 if not "iterations" in params else params['iterations']
    # Warm up
    with torch.no_grad():
        for _ in range(20):
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
            print("Iteration {}: {:.6f} s".format(i, end_time - start_time))
    
    printStats("Torch-TensorRT", timings, precision)

def run_tensorrt(model, input_tensors, params, precision):
    print("Running TensorRT")
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    iters = 20 if not "iterations" in params else params['iterations']

    if not "batch" in params:
        batch_size = 1
    else:
        batch_size = params['batch_size']

    with onnx_to_trt_engine(model, precision) as engine, engine.create_execution_context() as context:
        
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            # Input already allocated in input_tensors
            mem = cuda.mem_alloc() 
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            """
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
            if not engine.binding_is_input(binding):
                outputs.append(cuda.mem_alloc(cuda.pagelocked_empty(size, dtype).nbytes))
            else:
                bindings.append(input_tensors)
            """
        # Warm up
        for _ in range(20):
            context.execute_async(batch_size, bindings, stream.handle)
        
        stream.synchronize()

        for i in range(iters):
            start_time = timeit.default_timer()
            context.execute_async(batch_size, bindings, stream.handle)
            stream.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time



    iters = 20 if not "iterations" in params else params['iterations']
    # Warm up
    with torch.no_grad():
        for _ in range(20):
            features = model(input_tensors)

    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for i in range(iters):
            start_time = timeit.default_timer()
            features = model(input_tensors)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            timings.append(meas_time)
            print("Iteration {}: {:.6f} s".format(i, end_time - start_time))
    
    printStats("TensorRT", timings, precision)

def run(model, input_tensors, params, precision):
    for backend in params['backend']:
        if backend == 'all':
            run_torch(model, input_tensors, params, precision)
            run_torch_tensorrt(model, input_tensors, params, precision)
            run_tensorrt(model, input_tensors, params, precision)
    
        elif backend == "torch":
            run_torch(model, input_tensors, params, precision)
    
        elif backend == "torch_tensorrt":
            run_torch_tensorrt(model, input_tensors, params, precision)
    
        elif backend == "tensorrt":
            run_tensorrt(model, input_tensors, params, precision)


def printStats(backend, timings, precision, batch_size = 1):
    times = np.array(timings)
    steps = len(times)
    speeds = batch_size / times
    time_mean = np.mean(times)
    time_med = np.median(times)
    time_99th = np.percentile(times, 99)
    time_std = np.std(times, ddof=0)
    speed_mean = np.mean(speeds)
    speed_med = np.median(speeds)

    msg = ("\n%s =================================\n"
            "batch size=%d, num iterations=%d\n"
            "  Median FPS: %.1f, mean: %.1f\n"
            "  Median latency: %.6f, mean: %.6f, 99th_p: %.6f, std_dev: %.6f\n"
            ) % (backend,
                batch_size, steps,
                speed_med, speed_mean,
                time_med, time_mean, time_99th, time_std)
    print(msg)
    meas = {
        'Backend' : backend,
        'precision' : precision,
        'Median(FPS)' : speed_med,
        'Mean(FPS)' : speed_mean,
        'Median-Latency(ms)' : time_med,
        'Mean-Latency(ms)' : time_mean,
        '99th_p' : time_99th,
        'std_dev': time_std
    }
    results.append(meas)

def read_config(config_file):
    with open(config_file, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params

def precision_to_dtype(pr):
    if pr == 'fp32':
        return torch.float
    elif pr == 'fp16' or pr == 'half':
        return torch.half
    else:
        return torch.int8

def load_model(params):
    model = None
    # Load traced model
    if "torch" in params['backend'] or "torch_tensorrt" in params['backend']:
        model_path = os.path.join("models", params['model']['filename'])
        model = torch.jit.load(model_path).cuda()

    elif "tensorrt" in params['backend']:
        onnx_model_file = os.path.join("models", params['model']['onnx_file'])
        with open(onnx_model_file, 'rb') as onnx_model:
            print('Beginning ONNX file parsing')
            model = onnx_model.read()

    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run inference on a model with random input values")
    parser.add_argument("--config", help="Load YAML based configuration file to run the inference. If this is used other params will be ignored")
    args = parser.parse_args()
    
    # Load YAML params
    params = read_config(args.config)
    
    print("Loading model: ", params['model']['filename'])

    model = None

    if "device" in params['runtime']:
        torch.cuda.set_device(params['runtime']['device'])

    model = load_model(params)
    
    cudnn.benchmark = True

    # Create random input tensor of certain size
    torch.manual_seed(12345)

    num_input = params['input']['num_of_input']
    for precision in params['runtime']['precision']:
        input_tensors = []
        num_input = params['input']['num_of_input']
        for i in range(num_input):
            inp_tensor = params['input']['input' + str(i)]
            input_tensors.append(torch.randint(0, 2, tuple(d for d in inp_tensor), dtype=precision_to_dtype(precision)).cuda())

        if precision == "fp16" or precision == "half":
            #input_tensors = [x.half() for x in input_tensors]
            model = model.half()

        run(model, input_tensors, params, precision)

    print('Model Summary:')
    summary = pd.DataFrame(results)
    print(summary)