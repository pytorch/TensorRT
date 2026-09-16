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
        "torch_tensorrt_executorch_runtime.runtime.load() is deprecated; import "
        "torch_tensorrt_executorch_runtime to register the delegate, then use "
        "executorch.extension.pybindings.portable_lib._load_for_executorch(path).",
        DeprecationWarning,
        stacklevel=2,
    )
    from . import register

    register()
    from executorch.extension.pybindings.portable_lib import _load_for_executorch

    return _load_for_executorch(file_path)
