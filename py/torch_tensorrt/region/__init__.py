# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Placement annotations for Torch-TensorRT's Dynamo frontend.

Mark a region in the model's source with ``execute_in_torch``, ``fused_op`` or
``fused_kernel``, or select whole submodules at compile time with
``regions={selector: ExecuteInTorch() | FusedOp(...) | FusedKernel(...)}``.
"""

# Importing the package registers the marker operators, so that a saved program
# that holds them can be loaded.
from . import _marker_ops  # noqa: F401
from ._context import execute_in_torch, fused_kernel, fused_op, ignore_regions
from ._errors import RegionCaptureError, RegionError
from ._rules import ExecuteInTorch, FusedKernel, FusedOp

__all__ = [
    "execute_in_torch",
    "fused_op",
    "fused_kernel",
    "ExecuteInTorch",
    "FusedOp",
    "FusedKernel",
    "ignore_regions",
    "RegionError",
    "RegionCaptureError",
]
