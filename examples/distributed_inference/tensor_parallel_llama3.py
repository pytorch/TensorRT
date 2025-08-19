# Taken and modified pytorch lightening
# https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning
import logging
import os
import time

import torch
import torch_tensorrt
from llama3_model import ModelArgs, ParallelTransformer
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
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

logger.info(f"Starting PyTorch TP example on rank {_rank}.")
assert (
    _world_size % 2 == 0
), f"TP examples require even number of GPUs, but got {_world_size} gpus"

model_args = ModelArgs(
    vocab_size=32000,
    dim=1024,
    n_layers=4,
    n_heads=8,
    rope_theta=500000.0,
    n_kv_heads=8,
    device="cuda",
)

with torch.no_grad():
    model = ParallelTransformer(model_args, device_mesh)
    torch.manual_seed(0)
    inp = torch.randint(32000, (8, 256), device="cuda")
    python_result = model(inp)
    torch_tensorrt.runtime.set_multi_device_safe_mode(True)
    model = torch.compile(
        model,
        fullgraph=True,
        backend="torch_tensorrt",
        options={
            "use_python_runtime": True,
            "use_distributed_mode_trace": True,
            "debug": True,
        },
        dynamic=False,
    )

    start = time.time()
    output = model(inp)
    end = time.time()
    logger.info(f"Compilation time is {end-start}")
    assert (python_result - output).std() < 0.01, "Compilation result is not correct."

    cleanup_distributed_env()
