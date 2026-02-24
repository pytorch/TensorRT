"""
.. _dynamic_memory_allocation:

Dynamic Memory Allocation
==========================================================

This script demonstrates how to use dynamic memory allocation with Torch-TensorRT
to reduce GPU memory footprint. When enabled, TensorRT engines allocate and deallocate resources
dynamically during inference, which can significantly reduce peak memory usage.

This is particularly useful when:

- Running multiple models on the same GPU
- Working with limited GPU memory
- Memory usage needs to be minimized between inference calls
"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import gc
import time

import numpy as np
import torch
import torch_tensorrt as torch_trt
import torchvision.models as models

np.random.seed(5)
torch.manual_seed(5)
inputs = [torch.rand((100, 3, 224, 224)).to("cuda")]

# %%
# Compilation Settings with Dynamic Memory Allocation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Key settings for dynamic memory allocation:
#
# - ``dynamically_allocate_resources=True``: Enables dynamic resource allocation
# - ``lazy_engine_init=True``: Delays engine initialization until first inference
# - ``immutable_weights=False``: Allows weight refitting if needed
#
# With these settings, the engine will allocate GPU memory only when needed
# and deallocate it after inference completes.

settings = {
    "ir": "dynamo",
    "use_python_runtime": False,
    "enabled_precisions": {torch.float32},
    "immutable_weights": False,
    "lazy_engine_init": True,
    "dynamically_allocate_resources": True,
}

model = models.resnet152(pretrained=True).eval().to("cuda")
compiled_module = torch_trt.compile(model, inputs=inputs, **settings)
print((torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3)
compiled_module(*inputs)

# %%
# Runtime Resource Allocation Control
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You can control resource allocation behavior at runtime using the
# ``ResourceAllocationStrategy`` context manager. This allows you to:
#
# - Switch between dynamic and static allocation modes
# - Control when resources are allocated and deallocated
# - Optimize memory usage for specific inference patterns
#
# In this example, we temporarily disable dynamic allocation to keep
# resources allocated between inference calls, which can improve performance
# when running multiple consecutive inferences.

time.sleep(30)
with torch_trt.dynamo.runtime.ResourceAllocationStrategy(
    compiled_module, dynamically_allocate_resources=False
):
    print(
        "Memory used (GB):",
        (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3,
    )
    compiled_module(*inputs)
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(30)
    print(
        "Memory used (GB):",
        (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3,
    )
    compiled_module(*inputs)

# %%
# Memory Usage Comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Dynamic memory allocation trades off some performance for reduced memory footprint:
#
# **Benefits:**
#
# - Lower peak GPU memory usage
# - Reduced memory pressure on shared GPUs
#
# **Considerations:**
#
# - Slight overhead from allocation/deallocation
# - Best suited for scenarios where memory is constrained
# - May not be necessary for single-model deployments with ample memory
