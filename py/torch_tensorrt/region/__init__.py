# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Invocation-local placement annotations for Torch-TensorRT's Dynamo frontend."""

from ._context import execute_in_torch, fuse_in_plugin
from ._errors import RegionCaptureError, RegionError

__all__ = ["execute_in_torch", "fuse_in_plugin", "RegionError", "RegionCaptureError"]
