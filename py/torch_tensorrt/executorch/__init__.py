import importlib
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from executorch.exir import EdgeCompileConfig


def _executorch_exir_available() -> bool:
    try:
        return importlib.util.find_spec("executorch.exir") is not None
    except (ImportError, AttributeError, ValueError):
        return False


if not _executorch_exir_available():

    def __getattr__(name: str) -> NoReturn:
        raise ImportError(
            f"Cannot access torch_tensorrt.executorch.{name}: "
            "ExecuTorch is required with executorch.exir. "
            "Install an ExecuTorch package that provides executorch.exir"
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
