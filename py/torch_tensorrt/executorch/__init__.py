import importlib
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from executorch.exir import EdgeCompileConfig

if importlib.util.find_spec("executorch") is None:

    def __getattr__(name: str) -> NoReturn:
        raise ImportError(
            f"Cannot access torch_tensorrt.executorch.{name}: "
            "ExecuTorch is required. Install with: pip install executorch"
        )

    __all__ = [
        "get_edge_compile_config",
        "TensorRTPartitioner",
        "TensorRTBackend",
    ]
else:
    from torch_tensorrt.executorch.backend import TensorRTBackend
    from torch_tensorrt.executorch.partitioner import TensorRTPartitioner

    def get_edge_compile_config() -> "EdgeCompileConfig":
        """Return the EdgeCompileConfig used for Torch-TensorRT ExecuTorch export."""
        from executorch.exir import EdgeCompileConfig

        return EdgeCompileConfig(_check_ir_validity=False)

    __all__ = [
        "get_edge_compile_config",
        "TensorRTPartitioner",
        "TensorRTBackend",
    ]
