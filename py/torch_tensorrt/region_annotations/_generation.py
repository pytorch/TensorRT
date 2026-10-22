# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for compile-time, agent-authored region kernels."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelGenerator:
    """Generate and evaluate a static region kernel using a logged-in agent CLI.

    ``provider`` selects ``claude`` or ``codex``; ``backend`` selects ``triton``
    or NVIDIA ``cutile``, or ``cuda_cpp`` compiled through CUDA Python.
    Explicit ``module:attribute`` providers and backends can extend these choices.
    ``model=None`` retains the provider's configured model. Instructions supplement
    packaged backend skills; instruction files are snapshotted for each search.

    Budgets include authoring and evaluation. ``author_timeout_s`` bounds one
    proposal, including at most ``max_context_requests`` evidence reads. Search
    inputs are shared across candidates; the winner receives a fresh final check.
    ``objective`` selects candidate ranking, or a required speedup over eager
    PyTorch or a fully compiled TensorRT reference. ``min_speedup=1.05`` requires
    baseline/candidate latency of at least 1.05 after a median absolute deviation
    allowance for those baseline objectives.
    ``benchmark_repeats`` controls interleaved measurement samples. Passing
    kernels are cached and rechecked before reuse. Generated source executes locally.
    """

    provider: str = "claude"
    backend: str = "triton"
    model: str | None = None
    instructions: str | None = None
    instruction_files: tuple[str | Path, ...] = ()
    max_attempts: int = 4
    timeout_s: float = 600
    candidate_timeout_s: float = 120
    author_timeout_s: float = 120
    max_context_requests: int = 4
    cache_dir: str | Path | None = None
    use_cache: bool = True
    num_tests: int = 3
    benchmark_iters: int = 20
    check_nonfinite: bool = True
    benchmark_repeats: int = 5
    objective: str = "candidate"
    min_speedup: float = 1.0
    rtol: float | None = None
    atol: float | None = None

    def __post_init__(self) -> None:
        for name, builtins in (
            ("provider", ("claude", "codex")),
            ("backend", ("triton", "cutile", "cuda_cpp")),
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or (
                value not in builtins
                and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", value)
                is None
            ):
                raise ValueError(
                    f"KernelGenerator {name} must be one of {builtins} or module:attribute"
                )
        if self.instructions is not None and not isinstance(self.instructions, str):
            raise TypeError("KernelGenerator instructions must be a string or None")
        if isinstance(self.instruction_files, (str, bytes, Path)):
            raise TypeError("instruction_files must be a sequence of paths")
        object.__setattr__(
            self,
            "instruction_files",
            tuple(Path(p).expanduser() for p in self.instruction_files),
        )
        if self.objective not in ("candidate", "eager", "tensorrt"):
            raise ValueError("objective must be 'candidate', 'eager', or 'tensorrt'")
        if (
            isinstance(self.min_speedup, bool)
            or not isinstance(self.min_speedup, (int, float))
            or not math.isfinite(self.min_speedup)
            or self.min_speedup < 1
        ):
            raise ValueError("min_speedup must be finite and at least 1")
        if self.model is not None and (
            not isinstance(self.model, str) or not self.model.strip()
        ):
            raise ValueError("KernelGenerator model must be a nonempty string or None")
        for name in (
            "max_attempts",
            "num_tests",
            "benchmark_iters",
            "benchmark_repeats",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"KernelGenerator {name} must be a positive integer")
        if type(self.max_context_requests) is not int or self.max_context_requests < 0:
            raise ValueError("max_context_requests must be a nonnegative integer")
        for name in ("timeout_s", "candidate_timeout_s", "author_timeout_s"):
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
