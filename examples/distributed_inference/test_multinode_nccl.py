"""
Two-node native TensorRT NCCL test.

Reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from the environment so it
works correctly with torchrun or plain env-var injection across nodes.

Usage
-----
# Rank 0 (spirit):
  RANK=0 WORLD_SIZE=2 MASTER_ADDR=169.254.204.57 MASTER_PORT=29500 \\
    uv run python examples/distributed_inference/test_multinode_nccl.py

# Rank 1 (opportunity):
  RANK=1 WORLD_SIZE=2 MASTER_ADDR=169.254.204.57 MASTER_PORT=29500 \\
    uv run python examples/distributed_inference/test_multinode_nccl.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree
from torch.distributed._tensor import Shard
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

import torch_tensorrt
from torch_tensorrt.dynamo.runtime._nccl_utils import setup_nccl_library

torch.utils._pytree.register_constant(
    torch.distributed.tensor._dtensor_spec.DTensorSpec
)

setup_nccl_library()

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(0)  # one GPU per node
dist.init_process_group("nccl", rank=rank, world_size=world_size)
torch.manual_seed(0)

device_mesh = init_device_mesh("cuda", (world_size,))
print(f"[Rank {rank}/{world_size}] distributed init OK", flush=True)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
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


tp_model = ToyModel().to("cuda")
tp_model = parallelize_module(
    tp_model,
    device_mesh,
    {
        "in_proj": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj": RowwiseParallel(output_layouts=Shard(0)),
        "in_proj2": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj2": RowwiseParallel(output_layouts=Shard(0)),
    },
)

inp = torch.rand(20, 10, device="cuda")
python_result = tp_model(inp)
print(f"[Rank {rank}] PyTorch baseline OK, shape={python_result.shape}", flush=True)

PASSED = []
FAILED = []

for runtime, use_python in [("cpp", False), ("python", True)]:
    try:
        trt_model = torch.compile(
            tp_model,
            backend="torch_tensorrt",
            options={
                "truncate_long_and_double": True,
                "enabled_precisions": {torch.float32, torch.float16},
                "use_python_runtime": use_python,
                "min_block_size": 1,
                "use_distributed_mode_trace": True,
            },
        )
        output = trt_model(inp)
        std = float((python_result - output).std())
        if std < 0.01:
            print(f"[Rank {rank}] PASS  {runtime} runtime  (std={std:.6f})", flush=True)
            PASSED.append(runtime)
        else:
            print(f"[Rank {rank}] FAIL  {runtime} runtime  (std={std:.6f} >= 0.01)", flush=True)
            FAILED.append(runtime)
    except Exception as e:
        print(f"[Rank {rank}] ERROR {runtime} runtime: {e}", flush=True)
        FAILED.append(runtime)

dist.destroy_process_group()

print(f"[Rank {rank}] Results — passed: {PASSED}  failed: {FAILED}", flush=True)
sys.exit(0 if not FAILED else 1)
