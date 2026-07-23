import os
import platform
import tempfile

import torch
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import EngineCapability, dtype

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
# When True, use PyTorch's upstream complex decomposition (pytorch/pytorch#169832)
# via complex_decomposition_adapter instead of the legacy hand-rolled
# complex_graph_detection pass.  Default False until the new path is validated
# against the complex/RoPE test suites (issue #4390).
USE_COMPLEX_DECOMPOSITION = False
USE_FAST_PARTITIONER = True
ENABLE_EXPERIMENTAL_DECOMPOSITIONS = False
REQUIRE_FULL_COMPILATION = False
DRYRUN = False
HARDWARE_COMPATIBLE = False
SUPPORTED_KERNEL_PRECISIONS = {
    dtype.f32,
    dtype.f16,
    dtype.bf16,
    dtype.i8,
    dtype.f8,
    dtype.f4,
}
TIMING_CACHE_PATH = os.path.join(
    tempfile.gettempdir(), "torch_tensorrt_engine_cache", "timing_cache.bin"
)
LAZY_ENGINE_INIT = False
CACHE_BUILT_ENGINES = False
REUSE_CACHED_ENGINES = False
ENGINE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "torch_tensorrt_engine_cache")
ENGINE_CACHE_SIZE = 5368709120  # 5GB
CUSTOM_ENGINE_CACHE = None
USE_FP32_ACC = False
REFIT_IDENTICAL_ENGINE_WEIGHTS = False
STRIP_ENGINE_WEIGHTS = False
IMMUTABLE_WEIGHTS = True
ENABLE_WEIGHT_STREAMING = False
ENABLE_CROSS_COMPILE_FOR_WINDOWS = False
TILING_OPTIMIZATION_LEVEL = "none"
L2_LIMIT_FOR_TILING = -1
USE_DISTRIBUTED_MODE_TRACE = False
OFFLOAD_MODULE_TO_CPU = False
ENABLE_AUTOCAST = False
AUTOCAST_LOW_PRECISION_TYPE = None
AUTOCAST_EXCLUDED_NODES = set[str]()
AUTOCAST_EXCLUDED_OPS = set[torch.fx.node.Target]()
AUTOCAST_MAX_OUTPUT_THRESHOLD = 512
AUTOCAST_MAX_DEPTH_OF_REDUCTION = None
AUTOCAST_CALIBRATION_DATALOADER = None
ENABLE_RESOURCE_PARTITIONING = False
CPU_MEMORY_BUDGET = None
DYNAMICALLY_ALLOCATE_RESOURCES = False
DECOMPOSE_ATTENTION = False
ATTN_BIAS_IS_CAUSAL = True
FALLBACK_DATA_DEPENDENT_OPS = False

if platform.system() == "Linux":
    import pwd

    current_user = pwd.getpwuid(os.getuid())[0]
else:
    import getpass

    current_user = getpass.getuser()

DEBUG_LOGGING_DIR = os.path.join(
    tempfile.gettempdir(), f"torch_tensorrt_{current_user}/debug_logs"
)

# ---------------------------------------------------------------------------
# Runtime-only knobs (see torch_tensorrt.runtime.RuntimeSettings). Defaults
# live here to mirror compilation-settings convention; the dataclass imports
# from this module.
# ---------------------------------------------------------------------------
DYNAMIC_SHAPES_KERNEL_SPECIALIZATION_STRATEGY = "lazy"
CUDA_GRAPH_STRATEGY = "disabled"
# Default to a per-user temp file (mirrors ENGINE_CACHE_DIR). Users can override
# via ``RuntimeSettings(runtime_cache="/different/path")`` or a runtime CM.
RUNTIME_CACHE_PATH = os.path.join(
    tempfile.gettempdir(), f"torch_tensorrt_{current_user}/runtime_cache.bin"
)


def default_device() -> Device:
    return Device(gpu_id=torch.cuda.current_device())
