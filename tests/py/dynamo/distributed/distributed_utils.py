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
