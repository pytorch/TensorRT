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

__all__ = ["load"]


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
    from torch_tensorrt._executorch_compat import load as _load

    return _load(file_path)
