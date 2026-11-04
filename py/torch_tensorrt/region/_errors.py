# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause


class RegionError(RuntimeError):
    """An explicit region contract could not be honored."""


class RegionCaptureError(RegionError):
    """An annotated scope could not be captured without ambiguity."""
