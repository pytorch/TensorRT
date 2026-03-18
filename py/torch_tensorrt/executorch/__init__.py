# ExecuTorch backend for Torch-TensorRT: save/load .pte with TensorRT delegate.

from torch_tensorrt.executorch.backend import TensorRTBackend
from torch_tensorrt.executorch.partitioner import TensorRTPartitioner

__all__ = [
    "TensorRTBackend",
    "TensorRTPartitioner",
]
