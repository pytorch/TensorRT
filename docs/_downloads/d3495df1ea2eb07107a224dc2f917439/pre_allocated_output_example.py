"""
.. _pre_allocated_output_example:

Pre-allocated output buffer
======================================================

The TensorRT runtime module acts as a wrapper around a PyTorch model (or subgraph) that has been compiled and optimized into a TensorRT engine.

When the compiled module is executed, input and output tensors are set to TensorRT context for processing.
If output buffer allocation is moved after the execution of the TensorRT context and used it for next inference, GPU tasks and memory allocation tasks can operate concurrently. This overlap allows for more efficient use of GPU resources, potentially improving the performance of inference.

This optimization is particularly effective in below cases

1. Small inference time
    - The allocation of output buffers typically requires minimal CPU cycles, as the caching mechanism efficiently handles memory reuse. The time taken for this allocation is relatively constant compared to the overall inference time, leading to noticeable performance improvements, especially in scenarios involving small inference workloads. This is because the reduced allocation time contributes to faster execution when the computational workload is not large enough to overshadow these savings.
2. Multiple graph breaks
    - If the module contains operations that are not supported by TensorRT, the unsupported parts are handled by PyTorch and this fallback results in a graph break. The cumulative effect of optimized buffer allocations across multiple subgraphs can enhance overall inference performance.
    - While optimizing output buffers can mitigate some of this overhead, reducing or removing graph breaks should be prioritized as it enables more comprehensive optimizations
3. Static input or infrequent input shape change
    - If shape is changed, pre-allocated buffer cannot be used for next inference and there will new allocation before executing the TensorRT context. This feature is not suitable for use cases with frequent input shape changes
"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import timeit

import numpy as np
import torch
import torch_tensorrt
from transformers import BertModel

# %%
# Define function to measure inference performance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def test_module_perf(model, *input):
    timings = []

    # Warm-up phase to ensure consistent and accurate performance measurements.
    with torch.no_grad():
        for _ in range(3):
            model(*input)
    torch.cuda.synchronize()

    # Timing phase to measure inference performance
    with torch.no_grad():
        for i in range(10):
            start_time = timeit.default_timer()
            model(*input)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            timings.append(end_time - start_time)
    times = np.array(timings)
    time_med = np.median(times)

    # Return the median time as a representative performance metric
    return time_med


# %%
# Load model and compile
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Load bert model
model = (
    BertModel.from_pretrained("bert-base-uncased", torchscript=True)
    .eval()
    .half()
    .to("cuda")
)
# Define sample inputs
inputs = [
    torch.randint(0, 5, (1, 128), dtype=torch.int32).to("cuda"),
    torch.randint(0, 5, (1, 128), dtype=torch.int32).to("cuda"),
]
# Next, we compile the model using torch_tensorrt.compile
optimized_model = torch_tensorrt.compile(
    model,
    ir="dynamo",
    enabled_precisions={torch.half},
    inputs=inputs,
)

# %%
# Enable/Disable pre-allocated output buffer feature using runtime api
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Enable pre-allocated output buffer using a context manager
with torch_tensorrt.runtime.enable_pre_allocated_outputs(optimized_model):
    out_trt = optimized_model(*inputs)
    # Subsequent inferences can use the pre-allocated output buffer (no shape change)
    out_trt = optimized_model(*inputs)

# Alternatively, we can enable the feature using a context object
pre_allocated_output_ctx = torch_tensorrt.runtime.enable_pre_allocated_outputs(
    optimized_model
)
pre_allocated_output_ctx.set_pre_allocated_output(True)
time_opt = test_module_perf(optimized_model, *inputs)

# Disable the pre-allocated output buffer feature and perform inference normally
pre_allocated_output_ctx.set_pre_allocated_output(False)
out_trt = optimized_model(*inputs)
time_normal = test_module_perf(optimized_model, *inputs)

time_opt_ms = time_opt * 1000
time_normal_ms = time_normal * 1000

print(f"normal trt model time: {time_normal_ms:.3f} ms")
print(f"pre-allocated output buffer model time: {time_opt_ms:.3f} ms")
