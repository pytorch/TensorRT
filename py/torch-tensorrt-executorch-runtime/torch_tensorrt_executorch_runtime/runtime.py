# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated: loading moved to ExecuTorch's own Module API.

The published Torch-TensorRT wheel imports ``load`` from this submodule by name, so removing it
would turn ``torch_tensorrt.load(..., format="executorch")`` into a ModuleNotFoundError for anyone
who upgrades this package on its own. It stays until that call is gone from a released main wheel.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Union

# Program resolves through __getattr__ below rather than being bound here, so the linter cannot see
# it and is told so. It stays exported because the published package exported it.
__all__ = ["load", "Program"]  # noqa: F822


def _cannot_forward(error: ImportError) -> ImportError:
    """Name the case the reader is actually in.

    This package does not declare Torch-TensorRT, so absent is the ordinary state and too old is the
    rare one. Reporting the rare one told a reader with no Torch-TensorRT at all to go and upgrade it,
    and offered them a call in the module they do not have.

    Read from sys.modules: the failed import above already put the package there if it imported at
    all, so absent here means absent.
    """
    if "torch_tensorrt" not in sys.modules:
        return ImportError(
            "This deprecated door forwards into torch_tensorrt, which is not installed. This "
            "package does not require it. Either install torch-tensorrt, or drop the deprecated "
            "call: import this package to register the TensorRT delegate, then load with "
            "executorch.runtime.Runtime.get().load_program(path). "
            f"Underlying error: {error}"
        )
    # No advice to call torch_tensorrt.load(format="executorch") here: on the wheel that reaches
    # this branch, that call is what forwards into this function, so it returns the reader to the
    # error they already have.
    return ImportError(
        "This deprecated door forwards into torch_tensorrt, and the installed Torch-TensorRT does "
        "not carry the receiving module. Upgrade Torch-TensorRT to a build that has it, or drop "
        "the deprecated call: import this package to register the TensorRT delegate, then load "
        "with executorch.runtime.Runtime.get().load_program(path). "
        f"Underlying error: {error}"
    )


def __getattr__(name: str) -> Any:
    """Resolve Program from the main wheel on first use.

    The published package exported this name alongside load, so code that imported it by name has to
    keep working. Resolving it lazily rather than at module import keeps this module importable
    against a main wheel that does not have it, which is the same reason load defers its own import.
    """
    if name == "Program":
        # Guarded for the same reason load is, and with the same message. Left bare, this door
        # produced the very error the other door's guard exists to avoid, so an install that skipped
        # dependency resolution got a bare module-not-found naming a module the caller never asked
        # for, depending only on which name they reached for first.
        try:
            from torch_tensorrt._executorch_compat import Program
        except ImportError as error:
            raise _cannot_forward(error) from error

        return Program
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load(path: Union[str, Path]) -> Any:
    """Deprecated: register the delegate and load through ExecuTorch.

    Registration is what this package exists for, and ExecuTorch owns execution, so this does the
    first and forwards the second rather than carrying a loader of its own.

    The parameter matches the shim that forwards to it; it is spelled the same way as its only
    caller for readability.
    """
    warnings.warn(
        "torch_tensorrt_executorch_runtime.runtime.load() is deprecated; import this package to "
        "register the TensorRT delegate, then load with "
        "executorch.runtime.Runtime.get().load_program(path) and run with "
        "program.load_method('forward').execute(inputs).",
        DeprecationWarning,
        stacklevel=2,
    )
    # Forward to the main wheel's loader rather than ExecuTorch's. The API this replaces returned a
    # Program carrying run() and forward() and raised FileNotFoundError for a missing path, and
    # ExecuTorch's own loader returns neither, so a caller of the published API would break on the
    # return value instead of on the import.
    try:
        from torch_tensorrt._executorch_compat import load as _load
    except ImportError as error:
        raise _cannot_forward(error) from error

    return _load(path)
