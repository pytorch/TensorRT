from executorch.exir import EdgeCompileConfig
from torch_tensorrt.executorch.backend import TensorRTBackend
from torch_tensorrt.executorch.partitioner import TensorRTPartitioner


def get_edge_compile_config() -> EdgeCompileConfig:
    """Return the EdgeCompileConfig used for Torch-TensorRT ExecuTorch export."""
    return EdgeCompileConfig(_check_ir_validity=False)


__all__ = [
    "get_edge_compile_config",
    "TensorRTPartitioner",
    "TensorRTBackend",
]
