"""
.. _tensor_parallel_initialize_dist:
Tensor Parallel Initialize Distributed Environment
==================================================

This module provides functions to initialize and clean up the distributed environment for tensor parallel distributed inference.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
import torch.distributed as dist
from torch.distributed._tensor.device_mesh import DeviceMesh, init_device_mesh

logger = logging.getLogger(__name__)


# this is kept at the application level, when mpirun is used to run the application
def initialize_distributed_env(rank=0, world_size=1, port=29500):
    local_rank = int(
        os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank % torch.cuda.device_count())
    )
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", world_size))

    # Set up environment variable to run with mpirun
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    # Necessary to assign a device to each rank.
    torch.cuda.set_device(local_rank)

    # We use nccl backend
    dist.init_process_group("nccl")

    # set a manual seed for reproducibility
    torch.manual_seed(1111)

    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,))
    rank = device_mesh.get_rank()
    assert rank == local_rank
    device_id = (
        rank % torch.cuda.device_count()
    )  # Ensure each rank gets a unique device
    torch.cuda.set_device(device_id)

    return device_mesh, world_size, rank


def cleanup_distributed_env():
    """Clean up distributed process group to prevent resource leaks."""
    if dist.is_initialized():
        dist.destroy_process_group()


def check_tensor_parallel_device_number(world_size: int) -> None:
    if world_size % 2 != 0:
        raise ValueError(
            f"TP examples require even number of GPUs, but got {world_size} gpus"
        )


def get_tensor_parallel_device_mesh(
    rank: int = 0, world_size: int = 1
) -> tuple[DeviceMesh, int, int]:
    local_rank = int(
        os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank % torch.cuda.device_count())
    )
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", world_size))
    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,))
    rank = device_mesh.get_rank()
    assert rank == local_rank
    device_id = (
        rank % torch.cuda.device_count()
    )  # Ensure each rank gets a unique device
    torch.cuda.set_device(device_id)

    return device_mesh, world_size, rank


def initialize_distributed_logger(rank: int, logger_file_name: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logger_file_name + f"_{rank}.log", mode="w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger
