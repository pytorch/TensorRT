import importlib
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from executorch.exir import EdgeCompileConfig


def _has_executorch_exir() -> bool:
    try:
        return importlib.util.find_spec("executorch.exir") is not None
    except ModuleNotFoundError:
        return False


if not _has_executorch_exir():

    def __getattr__(name: str) -> NoReturn:
        raise ImportError(
            f"Cannot access torch_tensorrt.executorch.{name}: "
            "ExecuTorch with executorch.exir is required. "
            'Install with: pip install "torch_tensorrt[executorch]"'
        )

    __all__ = [
        "get_edge_compile_config",
        "TensorRTPartitioner",
        "TensorRTBackend",
    ]
else:
    from torch_tensorrt.executorch.backend import TensorRTBackend
    from torch_tensorrt.executorch.partitioner import TensorRTPartitioner
    from torch_tensorrt.executorch.runtime import Program, load

    def get_edge_compile_config() -> "EdgeCompileConfig":
        """Return the EdgeCompileConfig used for Torch-TensorRT ExecuTorch export."""
        from executorch.exir import EdgeCompileConfig

        return EdgeCompileConfig(_check_ir_validity=False)

    __all__ = [
        "get_edge_compile_config",
        "TensorRTPartitioner",
        "TensorRTBackend",
        "Program",
        "load",
    ]
