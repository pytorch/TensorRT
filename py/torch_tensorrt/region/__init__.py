# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Invocation-local placement annotations for Torch-TensorRT's Dynamo frontend."""

from ._context import execute_in_torch, fused_kernel, fused_op
from ._errors import RegionCaptureError, RegionError
from ._generation import (
    GeneratedKernel,
    KernelGenerator,
    MissingKernel,
    SearchPolicy,
    generated_kernels,
    missing_kernels,
    record_generated_kernel,
    record_missing_kernel,
)

__all__ = [
    "execute_in_torch",
    "fused_op",
    "fused_kernel",
    "KernelGenerator",
    "SearchPolicy",
    "GeneratedKernel",
    "generated_kernels",
    "record_generated_kernel",
    "MissingKernel",
    "missing_kernels",
    "record_missing_kernel",
    "RegionError",
    "RegionCaptureError",
]
