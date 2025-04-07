import os
import tempfile

import torch
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import EngineCapability, dtype

ENABLED_PRECISIONS = {dtype.f32}
DEBUG = False
DEVICE = None
DISABLE_TF32 = False
ASSUME_DYNAMIC_SHAPE_SUPPORT = False
DLA_LOCAL_DRAM_SIZE = 1073741824
DLA_GLOBAL_DRAM_SIZE = 536870912
DLA_SRAM_SIZE = 1048576
ENGINE_CAPABILITY = EngineCapability.STANDARD
WORKSPACE_SIZE = 0
MIN_BLOCK_SIZE = 5
PASS_THROUGH_BUILD_FAILURES = False
MAX_AUX_STREAMS = None
NUM_AVG_TIMING_ITERS = 1
VERSION_COMPATIBLE = False
OPTIMIZATION_LEVEL = None
SPARSE_WEIGHTS = False
TRUNCATE_DOUBLE = False
USE_PYTHON_RUNTIME = False
USE_FAST_PARTITIONER = True
ENABLE_EXPERIMENTAL_DECOMPOSITIONS = False
REQUIRE_FULL_COMPILATION = False
DRYRUN = False
HARDWARE_COMPATIBLE = False
SUPPORTED_KERNEL_PRECISIONS = {dtype.f32, dtype.f16, dtype.bf16, dtype.i8, dtype.f8}
TIMING_CACHE_PATH = os.path.join(
    tempfile.gettempdir(), "torch_tensorrt_engine_cache", "timing_cache.bin"
)
LAZY_ENGINE_INIT = False
CACHE_BUILT_ENGINES = False
REUSE_CACHED_ENGINES = False
ENGINE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "torch_tensorrt_engine_cache")
ENGINE_CACHE_SIZE = 5368709120  # 5GB
CUSTOM_ENGINE_CACHE = None
USE_EXPLICIT_TYPING = False
USE_FP32_ACC = False
REFIT_IDENTICAL_ENGINE_WEIGHTS = False
STRIP_ENGINE_WEIGHTS = False
IMMUTABLE_WEIGHTS = True
ENABLE_WEIGHT_STREAMING = False
ENABLE_CROSS_COMPILE_FOR_WINDOWS = False
TILING_OPTIMIZATION_LEVEL = "none"
L2_LIMIT_FOR_TILING = -1
USE_DISTRIBUTED_MODE_TRACE = False


def default_device() -> Device:
    return Device(gpu_id=torch.cuda.current_device())
