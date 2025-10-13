import logging
import os
import random

import numpy as np
import tensorrt as trt
import torch
import torch.distributed as dist
from torch.distributed._tensor.device_mesh import init_device_mesh


# the below two functions are used to set the environment variables for the pytest single and multi process
# this is for the github CI where we use pytest
def set_environment_variables_pytest_single_process():
    port = 29500 + random.randint(1, 1000)
    os.environ["WORLD_SIZE"] = str(1)
    os.environ["RANK"] = str(0)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)


def set_environment_variables_pytest_multi_process(
    rank: int = 0, world_size: int = 1
) -> None:
    port = 29500 + random.randint(1, 1000)
    # these variables are set by mpirun -n 2
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
