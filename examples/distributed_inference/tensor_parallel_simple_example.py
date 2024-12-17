import time

import tensorrt as trt
import tensorrt_llm
import torch
import torch.nn as nn
import torch_tensorrt
from tensor_parallel_nccl_ops import register_nccl_ops
from torch.distributed._tensor import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

device_mesh, _world_size, _rank, logger = register_nccl_ops(
    "./tensor_parallel_simple_example"
)

"""
This example copies some code from https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/tensor_parallel_example.py
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
        "use_aot_joint_export": False,
    },
    dynamic=False,
)

for i in range(10):
    # For TP, input needs to be same across all TP ranks.
    # Setting the random seed is to mimic the behavior of dataloader.
    torch.manual_seed(i)
    inp = torch.rand(20, 10, device="cuda")
    start = time.time()
    output = tp_model(inp)
    end = time.time()
    if i == 0:
        logger.info(f"Compilation time is {end-start}")
        assert (
            python_result - output
        ).std() < 0.01, "Compilation result is not correct."
    elif _rank == 0:
        logger.info(f"Inference time is {end-start}")
