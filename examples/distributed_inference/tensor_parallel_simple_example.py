"""
.. _tensor_parallel_simple_example:

Tensor Parallel Distributed Inference with Torch-TensorRT
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

   # JIT mode python runtime
   mpirun -n 2 python tensor_parallel_simple_example.py --mode jit_cpp

   # JIT mode cpp runtime
   mpirun -n 2 python tensor_parallel_simple_example.py --mode jit_python

   WIP: Export and load mode
    mpirun -n 2 python tensor_parallel_simple_example.py --mode export --save-path /tmp/tp_model.ep
    mpirun -n 2 python tensor_parallel_simple_example.py --mode load --save-path /tmp/tp_model.ep

"""

import argparse
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree
from tensor_parallel_initialize_dist import (
    cleanup_distributed_env,
    initialize_distributed_env,
)

torch.utils._pytree.register_constant(
    torch.distributed.tensor._dtensor_spec.DTensorSpec
)

parser = argparse.ArgumentParser(description="Tensor Parallel Simple Example")
parser.add_argument(
    "--mode",
    type=str,
    choices=["jit_python", "jit_cpp", "export", "load"],
    default="jit_python",
)
parser.add_argument("--save-path", type=str, default="/tmp/tp_model.ep")
args = parser.parse_args()

device_mesh, _world_size, _rank, logger = initialize_distributed_env(
    "tensor_parallel_simple_example"
)
import torch_tensorrt
from torch_tensorrt.distributed import setup_nccl_for_torch_tensorrt

setup_nccl_for_torch_tensorrt()
from torch.distributed._tensor import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

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

if args.mode == "load":
    # Load per-rank model: /tmp/tp_model.ep -> /tmp/tp_model_rank0_of_2.ep
    logger.info(f"Loading from {args.save_path}")
    loaded_program = torch_tensorrt.load(args.save_path)
    output = loaded_program.module()(inp)
    dist.barrier()
    assert (python_result - output).std() < 0.01, "Result mismatch"
    logger.info("Load successful!")

elif args.mode == "jit_python":
    trt_model = torch.compile(
        tp_model,
        backend="torch_tensorrt",
        options={
            "truncate_long_and_double": True,
            "use_python_runtime": True,
            "min_block_size": 1,
        },
    )
    output = trt_model(inp)
    dist.barrier()

    assert (python_result - output).std() < 0.01, "Result mismatch"
    logger.info("JIT compile successful!")

elif args.mode == "jit_cpp":
    trt_model = torch.compile(
        tp_model,
        backend="torch_tensorrt",
        options={
            "truncate_long_and_double": True,
            "use_python_runtime": False,
            "min_block_size": 1,
        },
    )
    output = trt_model(inp)
    dist.barrier()
    assert (python_result - output).std() < 0.01, "Result mismatch"
    logger.info("JIT compile successful!")

elif args.mode == "export":
    # Export: torch.export + dynamo.compile - AOT compilation, can save
    exported_program = torch.export.export(tp_model, (inp,), strict=False)
    trt_model = torch_tensorrt.dynamo.compile(
        exported_program,
        inputs=[inp],
        truncate_double=True,
        use_python_runtime=False,
        min_block_size=1,
        use_distributed_mode_trace=True,
    )
    output = trt_model(inp)
    dist.barrier()
    assert (python_result - output).std() < 0.01, "Result mismatch"

    # Save per-rank: /tmp/tp_model.ep -> /tmp/tp_model_rank0_of_2.ep
    save_path = torch_tensorrt.save(trt_model, args.save_path, inputs=[inp])
    logger.info(f"Saved to {save_path}")
    dist.barrier()

cleanup_distributed_env()
logger.info("Done!")
