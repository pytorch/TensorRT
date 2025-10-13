import logging
import os

import torch
from torch.distributed._tensor.device_mesh import DeviceMesh, init_device_mesh

logger = logging.getLogger(__name__)


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
