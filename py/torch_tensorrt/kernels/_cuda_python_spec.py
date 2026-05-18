from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


def _default_cuda_include_paths() -> List[str]:
    """Resolve CUDA include dir from CUDA_HOME / CUDA_PATH, else default."""
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env_var)
        if root:
            return [os.path.join(root, "include")]
    return ["/usr/local/cuda/include"]


@dataclass
class CudaPythonSpec:
    """Specification for a CUDA C++ kernel compiled via NVRTC (cuda-python).

    Create instances via :func:`cuda_python` rather than constructing directly.
    """

    kernel_source: str
    kernel_name: str
    aot_fn: Optional[Callable[..., Any]]
    eager_fn: Optional[Callable[..., Any]] = None
    include_paths: List[str] = field(default_factory=_default_cuda_include_paths)
    compile_std: str = "c++17"
    arch_override: Optional[str] = None
