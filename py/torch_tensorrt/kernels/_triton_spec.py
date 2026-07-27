from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class TritonSpec:
    """Specification for a ``@triton.jit`` kernel registered as an AOT QDP plugin.

    Built by :func:`torch_tensorrt.kernels.triton_op`; not intended to be
    constructed directly. Carries the compiled entry name, the AOT launch
    function, an optional CUDA eager implementation, and the Triton
    ``signature`` / ``constexprs`` for provenance. The compiled PTX itself is
    passed to the registrar separately as precompiled bytes.

    This is the Triton counterpart to
    :class:`torch_tensorrt.kernels._cuda_python_spec.CudaPythonSpec`; both are
    accepted by ``register_qdp_plugin``.
    """

    kernel_name: str
    aot_fn: Optional[Callable[..., Any]]
    eager_fn: Optional[Callable[..., Any]] = None
    signature: Dict[str, str] = field(default_factory=dict)
    constexprs: Dict[str, Any] = field(default_factory=dict)
