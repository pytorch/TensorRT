# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Reproducible, stateless validation and serial inference timing.

Models are copied and evaluated in eval/inference mode. Caller model state,
inputs, modes, and random generators are preserved. The current protocol rejects
input/registered-state mutation and multi-device execution. Mutation checks do
not cover ordinary Python attributes or unregistered tensor attributes.
"""

from ._benchmark import benchmark
from ._inputs import InputSample, VerificationError
from ._validation import validate

__all__ = ["InputSample", "VerificationError", "validate", "benchmark"]
