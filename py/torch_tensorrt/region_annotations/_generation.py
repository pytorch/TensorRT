# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for compile-time, agent-authored region kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelGenerator:
    """Generate and evaluate a static region kernel using a logged-in agent CLI.

    ``provider`` selects ``claude`` or ``codex``; ``backend`` selects ``triton``
    or NVIDIA ``cutile``. ``model=None`` retains the CLI's configured model.
    Budgets include model calls and candidate evaluation. Passing kernels are
    cached and rechecked before reuse. Generated source executes locally.
    """

    provider: str = "claude"
    backend: str = "triton"
    model: str | None = None
    max_attempts: int = 4
    timeout_s: float = 600
    candidate_timeout_s: float = 120
    cache_dir: str | Path | None = None
    use_cache: bool = True
    num_tests: int = 3
    benchmark_iters: int = 20
    check_nonfinite: bool = True
    rtol: float | None = None
    atol: float | None = None

    def __post_init__(self) -> None:
        if self.provider not in ("claude", "codex"):
            raise ValueError("KernelGenerator provider must be 'claude' or 'codex'")
        if self.backend not in ("triton", "cutile"):
            raise ValueError("KernelGenerator backend must be 'triton' or 'cutile'")
        if self.model is not None and (
            not isinstance(self.model, str) or not self.model.strip()
        ):
            raise ValueError("KernelGenerator model must be a nonempty string or None")
        for name in ("max_attempts", "num_tests", "benchmark_iters"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"KernelGenerator {name} must be a positive integer")
        for name in ("timeout_s", "candidate_timeout_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"KernelGenerator {name} must be finite and positive")
        if (self.rtol is None) != (self.atol is None):
            raise ValueError("Supply both rtol and atol, or neither")
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"KernelGenerator {name} must be finite and nonnegative"
                )
        for name in ("use_cache", "check_nonfinite"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"KernelGenerator {name} must be boolean")
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser())
