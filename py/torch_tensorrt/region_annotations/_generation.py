# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for compile-time, agent-authored region kernels."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _json_tuning_value(value: Any, depth: int = 0) -> Any:
    """Snapshot plain finite JSON values; tuples are convenient array inputs."""
    if depth > 16:
        raise ValueError("Tuning values cannot exceed 16 nesting levels")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple) and len(value) <= 1024:
        return [_json_tuning_value(item, depth + 1) for item in value]
    if type(value) is dict and len(value) <= 1024:
        if any(type(key) is not str for key in value):
            raise TypeError("Tuning JSON object keys must be strings")
        return {key: _json_tuning_value(item, depth + 1) for key, item in value.items()}
    raise ValueError(
        "Tuning values must be finite JSON data with containers of at most 1024 items"
    )


def _snapshot_tuning(tuning: Any) -> dict[str, tuple[Any, ...]]:
    if not isinstance(tuning, dict) or len(tuning) > 32:
        raise ValueError("tuning must be a mapping with at most 32 knob names")
    result = {}
    for name, choices in tuning.items():
        if not isinstance(name, str) or not name or len(name) > 256:
            raise ValueError(
                "Tuning knob names must be nonempty strings under 257 characters"
            )
        if not isinstance(choices, (list, tuple)) or not 1 <= len(choices) <= 128:
            raise ValueError(f"Tuning knob {name!r} needs 1 to 128 choices")
        result[name] = tuple(_json_tuning_value(value) for value in choices)
    if len(json.dumps(result, allow_nan=False).encode("utf-8")) > 65536:
        raise ValueError("Tuning configuration must fit within 64 KiB of JSON")
    return result


@dataclass(frozen=True)
class SearchPolicy:
    """Bounded author batches and deterministic backend launch tuning.

    ``max_candidates=None`` uses ``KernelGenerator.max_attempts * batch_size``;
    tuned variants count toward that same candidate budget. ``tuning`` maps
    backend proposal fields (or ``constants.NAME``) to explicit allowed choices.
    Every configured field is overridden before evaluation. Inputs are copied,
    so changing the caller's tuning containers does not change this policy.
    """

    batch_size: int = 1
    max_candidates: int | None = None
    max_no_improvement: int | None = None
    max_variants: int = 8
    finalists: int = 1
    resume: bool = False
    tuning: dict[str, tuple[Any, ...]] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        for name, bound in (
            ("batch_size", 32),
            ("max_candidates", 4096),
            ("max_no_improvement", 4096),
            ("max_variants", 128),
            ("finalists", 32),
        ):
            value = getattr(self, name)
            if value is None and name in ("max_candidates", "max_no_improvement"):
                continue
            if type(value) is not int or not 1 <= value <= bound:
                raise ValueError(
                    f"SearchPolicy {name} must be an integer in [1, {bound}]"
                )
        if type(self.resume) is not bool:
            raise ValueError("SearchPolicy resume must be boolean")
        object.__setattr__(self, "tuning", _snapshot_tuning(self.tuning))


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
    search: SearchPolicy = field(default_factory=SearchPolicy)
    diagnostics: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.search, SearchPolicy):
            raise TypeError("KernelGenerator search must be a SearchPolicy")
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
        for name in ("use_cache", "check_nonfinite", "diagnostics"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"KernelGenerator {name} must be boolean")
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser())
