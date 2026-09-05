"""ExecuTorch compilation and export integration.

Runtime loading is provided by the optional
``torch-tensorrt-executorch-runtime`` distribution and dispatched through
``executorch.runtime`` after importing ``torch_tensorrt_executorch_runtime``.
"""

import importlib.util
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
        from torch_tensorrt._utils import executorch_install_command

        raise ImportError(
            f"Cannot access torch_tensorrt.executorch.{name}: "
            "ExecuTorch with executorch.exir is required, and is published for "
            "Linux only. Install with: " + executorch_install_command()
        )

    __all__ = [
        "get_edge_compile_config",
        "TensorRTPartitioner",
        "TensorRTBackend",
        "export",
    ]
else:
    from torch_tensorrt.executorch._export import export
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
        "export",
    ]
