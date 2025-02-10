# Taken and modified pytorch lightening
# https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning
"""
.. _tensor_parallel_llama:

Torch distributed example for llama3-7B model
======================================================

As model sizes are increasing, large models with billions of parameters are trained with many GPUs, where regular data parallel training is no longer possible. In this example, we illustrate the Llama3-7B model inference using Torch-TensorRT backend, split across multiple GPUs using a form of model parallelism called Tensor Parallelism. We make use of Pytorch Distributed Tensor Parallelism Module. Please refer to these tutorials- https://pytorch.org/tutorials/intermediate/TP_tutorial.html and  https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning?section=featured"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import logging
import os
import time

import torch

# %%
# Pytorch Tensor Parallel APIs offer set of module level primitives(ParallelStyle) to configure the sharding of tensors in each layer of the model
# ParallelTransformer creates the parallelize_plan for the FeedForward layer of the model
from llama3_model import ModelArgs, ParallelTransformer
from tensor_parallel_initialize_dist import initialize_distributed_env
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

# %%
# Initialize the distributed environment
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Depending on the inputs/outputs sharded DTensors layout specified above, proper communication operations are required to transform DTensor layouts
# eg operations: allreduce, allgather, reduce_gather
# NCCL operations enable these operations.
# The below API does the following
# Initialize the communicators and the distributed environment
# Sets the path for the TRT-LLM plugin .so path which is required for the NCCL operations in Torch-TRT backend. Please note that if you are in python3.10 environment, `import tensorrt_llm` should be enough
# Initialize the logger. eg: In case of 2 GPUs, the log files are `./tensor_parallel_llama3_0.log` and `./tensor_parallel_llama3_1.log`
device_mesh, _world_size, _rank, logger = initialize_distributed_env(
    "./tensor_parallel_llama3"
)

# %%
# Model initialization with torch distributed parallel plan
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
    # The plan is
    # plan = {
    # "attention": PrepareModuleInput(
    #     input_layouts=(Shard(1), None),
    #     desired_input_layouts=(Replicate(), None),
    # ),
    # "attention.wq": ColwiseParallel(),
    # "attention.wk": ColwiseParallel(),
    # "attention.wv": ColwiseParallel(),
    # "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    # "attention_norm": SequenceParallel(),
    # "feed_forward": PrepareModuleInput(
    #     input_layouts=(Shard(1),),
    #     desired_input_layouts=(Replicate(),),
    # ),
    # "feed_forward.w1": ColwiseParallel(),
    # "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
    # "feed_forward.w3": ColwiseParallel(),
    # "ffn_norm": SequenceParallel(),
    # }

    model = ParallelTransformer(model_args, device_mesh)

    # %%
    # Model inference with Torch-TensorRT backend
    # -------------------------------------------
    #  When we compile the distributed model using Torch-TensorRT backend, pytorch distributed libraries create the sharded model
    #  on multiple GPUs and the communicator operations are used for proper communication. In the above,
    # `ColwiseParallel` and `RowwiseParallel` shard the attention layers in the column or row fashion.
    # `SequenceParallel` performs sharded computations of the normalization layer
    # `PrepareModuleInput` configures the model input with proper communication operations
    # The NCCL operations used in the distributed backend is handled by the TensorRT-LLM NCCL plugins, which causes no graph breaks now

    torch.manual_seed(0)
    inp = torch.randint(32000, (8, 256), device="cuda")
    python_result = model(inp)
    torch_tensorrt.runtime.set_multi_device_safe_mode(True)
    model = torch.compile(
        model,
        fullgraph=True,
        backend="torch_tensorrt",
        options={
            "truncate_long_and_double": True,
            "enabled_precisions": {torch.float32, torch.float16},
            "use_python_runtime": True,
            "workspace_size": 1 << 33,
            "debug": False,
            "use_aot_joint_export": False,
        },
        dynamic=False,
    )
    for i in range(15):
        # seeding with dp_rank to ensure identical inputs for TP groups
        torch.manual_seed(i)
        start = time.time()
        output = model(inp)
        end = time.time()
        if i == 0:
            # Logging the Compilation time
            logger.info(f"Compilation time is {end-start}")
            assert (
                python_result - output
            ).std() < 0.01, "Compilation result is not correct."
        elif _rank == 0:
            # Logging the inference time
            logger.info(f"Inference time is {end-start}")
