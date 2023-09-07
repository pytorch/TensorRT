import torch
from torch_tensorrt._Device import Device

PRECISION = torch.float32
DEBUG = False
DEVICE = None
WORKSPACE_SIZE = 0
MIN_BLOCK_SIZE = 5
PASS_THROUGH_BUILD_FAILURES = False
MAX_AUX_STREAMS = None
VERSION_COMPATIBLE = False
OPTIMIZATION_LEVEL = None
TRUNCATE_LONG_AND_DOUBLE = False
USE_PYTHON_RUNTIME = False
USE_FAST_PARTITIONER = True
ENABLE_EXPERIMENTAL_DECOMPOSITIONS = False
REQUIRE_FULL_COMPILATION = False
FALLBACK_TO_INDUCTOR = True


def default_device() -> Device:
    return Device(gpu_id=torch.cuda.current_device())
