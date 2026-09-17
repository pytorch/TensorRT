# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated: loading moved to ExecuTorch's own Module API.

The published Torch-TensorRT wheel imports ``load`` from this submodule by name, so removing it
would turn ``torch_tensorrt.load(..., format="executorch")`` into a ModuleNotFoundError for anyone
who upgrades this package on its own. It stays until that call is gone from a released main wheel.
"""

from __future__ import annotations

import warnings
from typing import Any

__all__ = ["load", "Program"]


def __getattr__(name: str) -> Any:
    """Resolve Program from the main wheel on first use.

    The published package exported this name alongside load, so code that imported it by name has to
    keep working. Resolving it lazily rather than at module import keeps this module importable
    against a main wheel that does not have it, which is the same reason load defers its own import.
    """
    if name == "Program":
        from torch_tensorrt._executorch_compat import Program

        return Program
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load(file_path: str) -> Any:
    """Deprecated: register the delegate and load through ExecuTorch.

    Registration is what this package exists for, and ExecuTorch owns execution, so this does the
    first and forwards the second rather than carrying a loader of its own.
    """
    warnings.warn(
        "torch_tensorrt_executorch_runtime.runtime.load() is deprecated; use "
        'torch_tensorrt.load(path, format="executorch") instead, which returns the same object.',
        DeprecationWarning,
        stacklevel=2,
    )
    # Forward to the main wheel's loader rather than ExecuTorch's. The API this replaces returned a
    # Program carrying run() and forward() and raised FileNotFoundError for a missing path, and
    # ExecuTorch's own loader returns neither, so a caller of the published API would break on the
    # return value instead of on the import. The main wheel is always present: this package declares
    # it as a dependency.
    # The loader this forwards to is part of the main wheel, and a main wheel old enough to import
    # this submodule by name does not carry it. That pairing should not arise, because this package
    # requires the main wheel of its own build exactly, so installing it moves the main wheel too.
    # If it does arise, through an install that skipped dependency resolution, say which of the two
    # is too old rather than reporting a module nobody asked for.
    try:
        from torch_tensorrt._executorch_compat import load as _load
    except ImportError as error:
        raise ImportError(
            "This deprecated loader forwards into torch_tensorrt, and the installed Torch-TensorRT "
            "is older than the one this package was built against, so it does not carry the "
            "receiving module. Install the Torch-TensorRT this package requires, or call "
            f'torch_tensorrt.load(path, format="executorch") directly. Underlying error: {error}'
        ) from error

    return _load(file_path)
