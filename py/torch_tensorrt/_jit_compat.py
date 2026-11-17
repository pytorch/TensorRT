# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve retained TorchScript implementations after public exports disappear.

Prefer the public entry points on PyTorch versions that provide them. Newer
versions retain the implementations in their original submodules. Returning
the original callable also preserves its signature for keyword filtering.
This module does not modify the global ``torch.jit`` namespace.
"""

import importlib
from typing import Any, Callable

import torch

_IMPLEMENTATIONS = {
    "load": "torch.jit._serialization",
    "save": "torch.jit._serialization",
    "script": "torch.jit._script",
    "trace": "torch.jit._trace",
}


def torchscript_api(name: str) -> Callable[..., Any]:
    if name not in _IMPLEMENTATIONS:
        raise ValueError(f"Unknown TorchScript entry point {name!r}")
    public = getattr(torch.jit, name, None)
    if callable(public):
        return public
    implementation = getattr(
        importlib.import_module(_IMPLEMENTATIONS[name]), name, None
    )
    if not callable(implementation):
        raise RuntimeError(f"Installed PyTorch does not provide TorchScript {name}")
    return implementation
