"""
.. _tensor_parallel_simple_example:

Torch Parallel Distributed example for simple model
=========================================

Below example shows how to use Torch-TensorRT backend for distributed inference with tensor parallelism.

This example demonstrates:
    - Setting up distributed environment for tensor parallelism
    - Model sharding across multiple GPUs
    - Compilation with Torch-TensorRT
    - Distributed inference execution

Usage
-----
.. code-block:: bash

    mpirun -n 2 --allow-run-as-root python tensor_parallel_simple_example.py
"""

import time

import tensorrt as trt
import torch
import torch.distributed as dist
import torch.nn as nn
import torch_tensorrt
from tensor_parallel_initialize_dist import (
    cleanup_distributed_env,
    initialize_distributed_env,
)
from torch.distributed._tensor import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch_tensorrt.dynamo.distributed.utils import (
    cleanup_distributed_env,
    get_tensor_parallel_device_mesh,
    initialize_distributed_env,
    initialize_logger,
)

if not dist.is_initialized():
    initialize_distributed_env()

device_mesh, _world_size, _rank = get_tensor_parallel_device_mesh()
logger = initialize_logger(_rank, "tensor_parallel_simple_example")


"""
This example takes some code from https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/tensor_parallel_example.py
"""


class ToyModel(nn.Module):
    """MLP based model"""

    def __init__(self):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(10, 3200)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(3200, 1600)
        self.in_proj2 = nn.Linear(1600, 500)
        self.out_proj2 = nn.Linear(500, 100)

    def forward(self, x):
        x = self.out_proj(self.relu(self.in_proj(x)))
        x = self.relu(x)
        x = self.out_proj2(self.relu(self.in_proj2(x)))
        return x


logger.info(f"Starting PyTorch TP example on rank {_rank}.")
assert (
    _world_size % 2 == 0
), f"TP examples require even number of GPUs, but got {_world_size} gpus"


# # create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
tp_model = ToyModel().to("cuda")


# Custom parallelization plan for the model
tp_model = parallelize_module(
    module=tp_model,
    device_mesh=device_mesh,
    parallelize_plan={
        "in_proj": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj": RowwiseParallel(output_layouts=Shard(0)),
        "in_proj2": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj2": RowwiseParallel(output_layouts=Shard(0)),
    },
)
torch.manual_seed(0)
inp = torch.rand(20, 10, device="cuda")
python_result = tp_model(inp)

backend = "torch_tensorrt"
tp_model = torch.compile(
    tp_model,
    backend=backend,
    options={
        "truncate_long_and_double": True,
        "enabled_precisions": {torch.float32, torch.float16},
        "use_python_runtime": True,
        "min_block_size": 1,
        "use_distributed_mode_trace": True,
    },
    dynamic=None,
)

# For TP, input needs to be same across all TP ranks.
# Setting the random seed is to mimic the behavior of dataloader.
torch.manual_seed(0)
inp = torch.rand(20, 10, device="cuda")
start = time.time()
output = tp_model(inp)
end = time.time()
logger.info(f"Compilation time is {end - start}")
assert (python_result - output).std() < 0.01, "Result is not correct."

# This cleans up the distributed process group
cleanup_distributed_env()
