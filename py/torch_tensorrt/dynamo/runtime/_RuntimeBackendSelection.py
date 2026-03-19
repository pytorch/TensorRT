from __future__ import annotations

import logging
from enum import Enum
from typing import Union

import torch_tensorrt

logger = logging.getLogger(__name__)


class RuntimeBackend(str, Enum):
    """Which Torch-TensorRT engine execution stack to use."""

    CPP = "cpp"
    PYTHON = "python"


_RuntimeBackendArg = Union[RuntimeBackend, str]


def _default_runtime_backend() -> RuntimeBackend:
    return (
        RuntimeBackend.CPP
        if torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime
        else RuntimeBackend.PYTHON
    )


_RUNTIME_BACKEND: RuntimeBackend = _default_runtime_backend()


def _normalize_runtime_backend(backend: _RuntimeBackendArg) -> RuntimeBackend:
    if isinstance(backend, RuntimeBackend):
        if (
            backend is RuntimeBackend.CPP
            and not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime
        ):
            raise RuntimeError(
                "C++ Torch-TensorRT runtime is not available in this build"
            )
        return backend

    normalized = backend.lower()
    if normalized not in ("cpp", "python"):
        raise ValueError(f"Unsupported runtime backend: {backend}")
    member = RuntimeBackend(normalized)
    if (
        member is RuntimeBackend.CPP
        and not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime
    ):
        raise RuntimeError("C++ Torch-TensorRT runtime is not available in this build")
    return member


def get_runtime_backend() -> RuntimeBackend:
    """Return the process-wide default backend (``cpp`` or ``python``)."""
    return _RUNTIME_BACKEND


class _RuntimeBackendContextManager:
    def __init__(self, old_backend: RuntimeBackend) -> None:
        self.old_backend = old_backend

    def __enter__(self) -> "_RuntimeBackendContextManager":
        return self

    def __exit__(self, *args: object) -> None:
        global _RUNTIME_BACKEND
        _RUNTIME_BACKEND = self.old_backend


def set_runtime_backend(backend: _RuntimeBackendArg) -> _RuntimeBackendContextManager:
    """Context manager: set global C++ vs Python engine path for unpinned modules.

    Use around compile and forward so :class:`~torch_tensorrt.runtime.TorchTensorRTModule`
    picks up the intended backend when it is constructed:

    .. code-block:: python

        with torch_tensorrt.runtime.set_runtime_backend("python"):
            trt_gm = torch_tensorrt.dynamo.compile(...)

    If the return value is not used with ``with``, the backend remains changed until you
    call ``__exit__`` on the returned object (or enter another ``set_runtime_backend`` context).
    """
    global _RUNTIME_BACKEND
    old_backend = _RUNTIME_BACKEND
    _RUNTIME_BACKEND = _normalize_runtime_backend(backend)
    logger.info(f"Set Torch-TensorRT runtime backend to {_RUNTIME_BACKEND}")
    return _RuntimeBackendContextManager(old_backend)
