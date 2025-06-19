import logging
import os
import time

import torch
import torch_tensorrt
from rotary_embedding import RotaryAttention, parallel_rotary_block
from tensor_parallel_initialize_dist import initialize_distributed_env

device_mesh, _world_size, _rank, logger = initialize_distributed_env(
    "./tensor_parallel_rotary_embedding"
)


"""
This example covers the rotary embedding in Llama3 model and is derived from https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning
"""

BATCH = 2
SEQ_LEN = 128
HEADS = 4
DIM = 128

with torch.no_grad():
    xq = torch.randn(BATCH, SEQ_LEN, HEADS, DIM)
    xk = torch.randn(BATCH, SEQ_LEN, HEADS, DIM)
    model = RotaryAttention(DIM, SEQ_LEN)
    parallel_rotary_block(model, device_mesh)
    device = torch.device("cuda", device_mesh.get_rank())
    model.to(device)
    x = torch.randn(BATCH, SEQ_LEN, HEADS, DIM).to(device)

    python_result = model(x)

    logger.info("Torch-tensorrt compilation for rotary embedding")

    model = torch.compile(model, backend="torch_tensorrt", options={"debug": True})

    for i in range(15):
        # seeding with dp_rank to ensure identical inputs for TP groups
        torch.manual_seed(i)
        start = time.time()
        output = model(x)
        end = time.time()
        if i == 0:
            logger.info(f"Compilation time is {end-start}")
            assert (
                python_result - output
            ).std() < 0.01, "Compilation result is not correct."
        elif _rank == 0:
            logger.info(f"Inference time is {end-start}")
