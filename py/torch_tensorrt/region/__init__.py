# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Invocation-local placement annotations for Torch-TensorRT's Dynamo frontend."""

from ._context import execute_in_torch, fused_kernel, fused_op
from ._errors import RegionCaptureError, RegionError
from ._generation import KernelGenerator

__all__ = [
    "execute_in_torch",
    "fused_op",
    "fused_kernel",
    "KernelGenerator",
    "RegionError",
    "RegionCaptureError",
]
