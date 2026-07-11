"""Activate the TensorRT delegate-enabled ExecuTorch Python runtime."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

BACKEND_NAME = "TensorRTBackend"
_NATIVE_NAME = "executorch.extension.pybindings._portable_lib"
_WRAPPER_NAME = "executorch.extension.pybindings.portable_lib"
_DATA_LOADER_NAME = "executorch.extension.pybindings.data_loader"


class DelegateCompatibilityError(ImportError):
    """The delegate wheel is incompatible with the active native runtime."""


def activate() -> ModuleType:
    """Make the delegate-enabled portable runtime back ``executorch.runtime``."""
    existing = sys.modules.get(_NATIVE_NAME)
    if existing is not None:
        if existing.__name__ != __name__ + "._portable_lib":
            raise DelegateCompatibilityError(
                "ExecuTorch's stock runtime was imported first. Import "
                "torch_tensorrt.executorch.runtime before executorch.runtime."
            )
        return existing
    try:
        data_loader = importlib.import_module(__name__ + ".data_loader")
        sys.modules[_DATA_LOADER_NAME] = data_loader
        native = importlib.import_module(__name__ + "._portable_lib")
    except ImportError as error:
        raise DelegateCompatibilityError(
            "Could not load the prebuilt Torch-TensorRT ExecuTorch runtime. "
            "Install torch, executorch, torch-tensorrt, and the delegate from "
            "the same release matrix."
        ) from error
    sys.modules[_NATIVE_NAME] = native
    sys.modules.pop(_WRAPPER_NAME, None)
    return native


def runtime():
    """Return the activated ExecuTorch Runtime singleton."""
    activate()
    from executorch.runtime import Runtime

    value = Runtime.get()
    if not value.backend_registry.is_available(BACKEND_NAME):
        raise DelegateCompatibilityError(f"{BACKEND_NAME} is not registered")
    return value


__all__ = ["BACKEND_NAME", "DelegateCompatibilityError", "activate", "runtime"]
