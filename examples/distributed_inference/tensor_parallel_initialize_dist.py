"""
.. _tensor_parallel_initialize_dist:
Tensor Parallel Initialize Distributed Environment
==================================================

This module provides functions to initialize and clean up the distributed environment for tensor parallel distributed inference. These utilities are useful for tensor parallel distributed inference examples using torch.distributed.
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


def initialize_logger(
    rank, logger_file_name, file_level=logging.DEBUG, console_level=logging.INFO
):
    """Initialize rank-specific Torch-TensorRT logger with configurable handler levels.

    Logger level is set to DEBUG (pass-through), handlers control filtering for files and stream buffers

    Args:
        rank: Process rank for multi-GPU
        logger_file_name: Base name for log file (will add _rank.log)
        file_level: What goes to file - default DEBUG (everything)
        console_level: What prints to console - default INFO (clean output)
    """
    logger = logging.getLogger("torch_tensorrt")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(logger_file_name + f"_{rank}.log", mode="w")
    fh.setLevel(file_level)
    fh.setFormatter(
        logging.Formatter(
            f"[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(fh)

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(
        console_level
    )  # Console handler controls what's printed in console output
    ch.setFormatter(logging.Formatter(f"[Rank {rank}] %(levelname)s: %(message)s"))
    logger.addHandler(ch)

    # safegauard though not reqd
    logger.propagate = False
    return logger


# This is required for env initialization since we use mpirun
def initialize_distributed_env(
    logger_file_name,
    rank=0,
    world_size=1,
    port=29500,
    file_level="debug",
    console_level="info",
):
    """Initialize distributed environment with handler-based logging.

    Args:
        logger_file_name: Base name for log files
        rank: Initial rank (overridden by OMPI env vars)
        world_size: Initial world size (overridden by OMPI env vars)
        port: Master port for distributed communication
        file_level: File handler level - "debug", "info", "warning" (default: "debug")
        console_level: Console handler level - "debug", "info", "warning" (default: "info")
    """
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
    # Convert string handler levels to logging constants
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    file_level_int = level_map.get(file_level.lower(), logging.DEBUG)
    console_level_int = level_map.get(console_level.lower(), logging.INFO)

    # Initialize logger with handler-specific levels
    # Logger itself is always DEBUG - handlers do the filtering
    logger = initialize_logger(
        rank,
        logger_file_name,
        file_level=file_level_int,
        console_level=console_level_int,
    )
    device_id = (
        rank % torch.cuda.device_count()
    )  # Ensure each rank gets a unique device
    torch.cuda.set_device(device_id)

    # Set C++ TensorRT runtime log level based on most verbose handler
    # Use the most verbose level to ensure all important logs are captured
    cpp_level = min(file_level_int, console_level_int)
    try:
        import torch_tensorrt.logging as torchtrt_logging

        torchtrt_logging.set_level(cpp_level)
    except Exception as e:
        logger.warning(f"Could not set C++ TensorRT log level: {e}")

    return device_mesh, world_size, rank, logger


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
