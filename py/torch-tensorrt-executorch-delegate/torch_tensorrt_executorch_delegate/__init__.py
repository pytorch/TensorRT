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
    if existing is not None and existing.__name__ == __name__ + "._portable_lib":
        return existing
    if existing is not None or _WRAPPER_NAME in sys.modules:
        raise DelegateCompatibilityError(
            "ExecuTorch's stock runtime was imported first. Call "
            "torch_tensorrt.executorch.load(...) before importing "
            "executorch.runtime."
        )
    missing = object()
    previous_data_loader = sys.modules.get(_DATA_LOADER_NAME, missing)
    try:
        data_loader = importlib.import_module(__name__ + ".data_loader")
        native = importlib.import_module(__name__ + "._portable_lib")
    except ImportError as error:
        if previous_data_loader is missing:
            sys.modules.pop(_DATA_LOADER_NAME, None)
        else:
            sys.modules[_DATA_LOADER_NAME] = previous_data_loader
        raise DelegateCompatibilityError(
            "Could not load the prebuilt Torch-TensorRT ExecuTorch runtime. "
            "Install torch, executorch, torch-tensorrt, and the delegate from "
            "the same release matrix."
        ) from error
    sys.modules[_DATA_LOADER_NAME] = data_loader
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
