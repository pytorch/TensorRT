import logging
import os

import numpy as np
import tensorrt as trt
import torch
import torch.distributed as dist
from torch.distributed._tensor.device_mesh import init_device_mesh


def set_environment_variables_pytest():
    os.environ["WORLD_SIZE"] = str(1)
    os.environ["RANK"] = str(0)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29500)
    os.environ["USE_TRTLLM_PLUGINS"] = "1"


def find_repo_root(max_depth=10):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for i in range(max_depth):
        files = os.listdir(dir_path)
        if "MODULE.bazel" in files:
            return dir_path
        else:
            dir_path = os.path.dirname(dir_path)

    raise RuntimeError("Could not find repo root")


def initialize_logger(rank, logger_file_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logger_file_name + f"_{rank}.log", mode="w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger


# This is required for env initialization since we use mpirun
def initialize_distributed_env(logger_file_name, rank=0, world_size=1, port=29500):
    local_rank = int(
        os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank % torch.cuda.device_count())
    )
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", world_size))

    # Set up environment variable to run with mpirun
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["TRTLLM_PLUGINS_PATH"] = (
        find_repo_root() + "/lib/libnvinfer_plugin_tensorrt_llm.so"
    )

    # Necessary to assign a device to each rank.
    torch.cuda.set_device(local_rank)

    # We use nccl backend
    dist.init_process_group("nccl")

    # set a manual seed for reproducibility
    torch.manual_seed(1111)

    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,))
    rank = device_mesh.get_rank()
    assert rank == local_rank
    logger = initialize_logger(rank, logger_file_name)
    device_id = (
        rank % torch.cuda.device_count()
    )  # Ensure each rank gets a unique device
    torch.cuda.set_device(device_id)

    return device_mesh, world_size, rank, logger
