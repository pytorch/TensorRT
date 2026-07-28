"""
.. _tensor_parallel_rotary_embedding:
Tensor Parallel Rotary Embedding Example
=======================================

This example demonstrates how to use Torch-TensorRT with tensor parallel distributed inference
for models that use rotary positional embeddings (RoPE). It lowers the complex
operations in attention models with rotary embeddings across multiple GPUs.

"""

import time

import torch
from rotary_embedding import RotaryAttention, parallel_rotary_block
from tensor_parallel_initialize_dist import (
    cleanup_distributed_env,
    initialize_distributed_env,
)

# Initialize distributed environment and logger BEFORE importing torch_tensorrt
# This ensures logging is configured before any import-time log messages
device_mesh, _world_size, _rank, logger = initialize_distributed_env(
    "tensor_parallel_rotary_embedding"
)


"""
This example covers the rotary embedding in Llama3 model and is derived from https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning
Command to run with single GPU: mpirun -n 1 --allow-run-as-root python tensor_parallel_rotary_embedding.py
"""

BATCH = 2
SEQ_LEN = 128
HEADS = 4
DIM = 128

with torch.no_grad():
    model = RotaryAttention(DIM, SEQ_LEN)
    parallel_rotary_block(model, device_mesh)
    device = torch.device("cuda", device_mesh.get_rank())
    model.to(device)
    x = torch.randn(BATCH, SEQ_LEN, HEADS, DIM).to(device)

    python_result = model(x)

    logger.info("Torch-tensorrt compilation for rotary embedding")

    model = torch.compile(model, backend="torch_tensorrt")

    torch.manual_seed(0)
    start = time.time()
    output = model(x)
    end = time.time()
    logger.info(f"Compilation time is {end-start}")
    assert (python_result - output).std() < 0.01, "Compilation result is not correct."

    cleanup_distributed_env()
