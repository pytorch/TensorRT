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

#: The ``module:attribute`` form of an extension ``provider`` or ``backend``.
IMPORT_PATH = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", re.ASCII)
_PROVIDERS = ("claude", "codex")
_BACKENDS = ("triton", "cutile", "cuda_cpp")


def split_import_path(path: str) -> tuple[str, str]:
    """Split a ``module:attribute`` extension path, rejecting any other form."""
    if not isinstance(path, str) or IMPORT_PATH.fullmatch(path) is None:
        raise ValueError(f"Expected a 'module:attribute' import path; got {path!r}")
    module, _, attribute = path.partition(":")
    return module, attribute


def _check(
    owner: object,
    name: str,
    kind: type,
    *,
    positive: bool = True,
    optional: bool = False,
) -> None:
    """Raise TypeError for a field of the wrong type, ValueError for a bad number.

    Numbers must be finite and positive, or nonnegative when ``positive=False``.
    """
    value = getattr(owner, name)
    if optional and value is None:
        return
    sign = "positive" if positive else "nonnegative"
    expected = {
        bool: "a bool",
        str: "a string",
        int: f"a {sign} integer",
        float: f"a finite {sign} number",
    }[kind]
    message = (
        f"{type(owner).__name__} {name} must be {expected}"
        f"{' or None' if optional else ''}; got {value!r}"
    )
    numeric = kind in (int, float)
    # bool subclasses int, but True is neither a count nor a duration.
    if not isinstance(value, (int, float) if kind is float else kind) or (
        numeric and isinstance(value, bool)
    ):
        raise TypeError(message)
    if numeric and (not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(message)


def _snapshot_tuning(tuning: Any) -> dict[str, tuple[Any, ...]]:
    """Copy tuning choices as finite JSON; tuples are accepted as arrays."""
    if not isinstance(tuning, dict):
        raise TypeError(f"Tuning must be a dict of knob choices; got {tuning!r}")
    snapshot = {}
    for name, choices in tuning.items():
        if not isinstance(name, str) or not isinstance(choices, (list, tuple)):
            raise TypeError(
                f"Tuning must map knob names to lists of choices; got {name!r}: "
                f"{choices!r}"
            )
        if not name or not choices:
            raise ValueError(
                f"Tuning knobs need a nonempty name and choices; got {name!r}"
            )
        try:
            snapshot[name] = tuple(json.loads(json.dumps(choices, allow_nan=False)))
        except (TypeError, ValueError) as error:
            raise type(error)(
                f"Tuning choices for {name!r} must be finite JSON values: {error}"
            ) from None
    return snapshot


@dataclass(frozen=True)
class SearchPolicy:
    """Search budget and launch tuning for a :class:`KernelGenerator`.

    Args:
        batch_size: Proposals the author may return per round.
        max_candidates: Search evaluations, counting tuning variants, resumed
            candidates and diagnostic runs, but not final validation or cache
            checks. ``None`` budgets ``max_attempts`` rounds of ``batch_size``
            proposals together with their tuning variants.
        max_no_improvement: Stop after this many consecutive author rounds
            without a better correct candidate; ``None`` does not stop early.
        max_variants: Tuning variants evaluated per proposal, at most 128.
        finalists: Best candidates eligible for the final validation on fresh
            inputs; if one fails, the next is tried.
        resume: Continue the checkpointed search of the same task, validating
            and measuring its retained candidates again.
        tuning: Allowed values for backend proposal fields, or for named
            constants as ``"constants.NAME"``; for example
            ``{"num_warps": (4, 8)}``. Each proposal is evaluated with up to
            ``max_variants`` combinations of these choices. The choices are
            copied as JSON data.
    """

    batch_size: int = 1
    max_candidates: int | None = None
    max_no_improvement: int | None = None
    max_variants: int = 8
    finalists: int = 1
    resume: bool = False
    tuning: dict[str, tuple[Any, ...]] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_variants", "finalists"):
            _check(self, name, int)
        for name in ("max_candidates", "max_no_improvement"):
            _check(self, name, int, optional=True)
        _check(self, "resume", bool)
        # Every variant of a proposal is materialized before evaluation.
        if self.max_variants > 128:
            raise ValueError(
                f"SearchPolicy max_variants must be at most 128; got {self.max_variants}"
            )
        object.__setattr__(self, "tuning", _snapshot_tuning(self.tuning))


@dataclass(frozen=True)
class KernelGenerator:
    """Generate a region kernel during compilation with a logged-in agent CLI.

    Pass it to :func:`fused_kernel`, creating it outside the traced
    ``forward``. The author proposes kernels; worker processes build each into
    a trial TensorRT engine, check it against the captured PyTorch region and
    time it. The selected kernel is checked again on fresh inputs before it is
    cached and registered. Model requests include the captured graph, and
    generated source runs locally.

    Args:
        provider: Author CLI, ``"claude"`` or ``"codex"``, or a trusted
            ``module:attribute`` provider adapter.
        backend: Kernel language: ``"triton"``, NVIDIA ``"cutile"``, or
            ``"cuda_cpp"`` (CUDA C++ compiled through CUDA Python), or a
            trusted ``module:attribute`` backend adapter.
        model: Model name for the provider; ``None`` keeps the CLI's default.
        instructions: Guidance added to the packaged backend skill.
        instruction_files: Guidance files. They must exist when the generator
            is created and are read again when each search starts.
        max_attempts: Author rounds per search.
        timeout_s: Time budget for generating one region, in seconds, covering
            authoring, evaluation and final validation. Each generated region
            has its own budget. Waiting for another process that is generating
            the same kernel is bounded separately by the same value.
        candidate_timeout_s: Time limit for evaluating one candidate.
        author_timeout_s: Time limit for one author round, including its
            evidence reads.
        max_context_requests: Evidence reads and diagnostic requests allowed
            per author round.
        cache_dir: Directory for prompts, captured reference graphs, generated
            source, feedback, logs, checkpoints and accepted kernels. Defaults
            to ``$XDG_CACHE_HOME/torch_tensorrt/region_kernels``, or
            ``~/.cache/torch_tensorrt/region_kernels``.
        use_cache: Reuse an accepted kernel after validating it again, and
            publish newly accepted kernels. ``False`` neither reads nor
            publishes accepted kernels, but still writes run artifacts.
        num_tests: Random input sets per correctness check.
        benchmark_iters: Kernel launches per timing sample.
        check_nonfinite: Also check inputs containing NaN and infinities.
        benchmark_repeats: Interleaved timing samples of each compared
            implementation.
        objective: ``"candidate"`` accepts the fastest correct candidate.
            ``"eager"`` and ``"tensorrt"`` also require ``min_speedup`` over the
            captured PyTorch region or over a fully TensorRT-compiled version
            of it.
        min_speedup: Required baseline/candidate latency ratio after a
            median-absolute-deviation noise allowance. Values above 1 require
            the ``"eager"`` or ``"tensorrt"`` objective.
        rtol: Relative output tolerance, set together with ``atol``. When both
            are ``None``, outputs are compared with a float64 evaluation of the
            reference and may deviate by a small multiple of the error of the
            PyTorch reference itself.
        atol: Absolute output tolerance; see ``rtol``.
        search: Search budget and launch tuning.
        diagnostics: Include compiler and execution facts in feedback, and let
            the author request a diagnostic run of the best kernel so far.
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
        for name in ("provider", "backend", "objective"):
            _check(self, name, str)
        for name in ("model", "instructions"):
            _check(self, name, str, optional=True)
        for name in (
            "max_attempts",
            "num_tests",
            "benchmark_iters",
            "benchmark_repeats",
        ):
            _check(self, name, int)
        _check(self, "max_context_requests", int, positive=False)
        for name in ("timeout_s", "candidate_timeout_s", "author_timeout_s"):
            _check(self, name, float)
        _check(self, "min_speedup", float)
        for name in ("rtol", "atol"):
            _check(self, name, float, positive=False, optional=True)
        for name in ("use_cache", "check_nonfinite", "diagnostics"):
            _check(self, name, bool)
        if not isinstance(self.search, SearchPolicy):
            raise TypeError(
                f"KernelGenerator search must be a SearchPolicy; got {self.search!r}"
            )
        if isinstance(self.instruction_files, (str, bytes, Path)):
            raise TypeError(
                "KernelGenerator instruction_files must be a sequence of paths; "
                f"got {self.instruction_files!r}"
            )

        for name, builtins in (("provider", _PROVIDERS), ("backend", _BACKENDS)):
            value = getattr(self, name)
            if value not in builtins and not IMPORT_PATH.fullmatch(value):
                raise ValueError(
                    f"KernelGenerator {name} must be one of {builtins} or a "
                    f"'module:attribute' import path; got {value!r}"
                )
        if self.objective not in ("candidate", "eager", "tensorrt"):
            raise ValueError(
                "KernelGenerator objective must be 'candidate', 'eager' or "
                f"'tensorrt'; got {self.objective!r}"
            )
        if self.model is not None and not self.model.strip():
            raise ValueError("KernelGenerator model must not be blank")
        if self.min_speedup < 1:
            raise ValueError(
                f"KernelGenerator min_speedup must be at least 1; got {self.min_speedup!r}"
            )
        if self.min_speedup != 1 and self.objective == "candidate":
            raise ValueError(
                "KernelGenerator min_speedup requires objective='eager' or "
                "'tensorrt'; the 'candidate' objective has no baseline"
            )
        if (self.rtol is None) != (self.atol is None):
            raise ValueError("KernelGenerator rtol and atol must be set together")
        files = tuple(Path(p).expanduser().resolve() for p in self.instruction_files)
        for path in files:
            if not path.is_file():
                raise ValueError(
                    f"KernelGenerator instruction file {path} is not an existing file"
                )
        object.__setattr__(self, "instruction_files", files)
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser())
        if self.backend in _BACKENDS:
            # Extension backends are imported only when generation runs, so
            # only built-in tuning schemas can be checked this early.
            from torch_tensorrt.dynamo.region_compilation._generation._backends import (
                validate_tuning_space,
            )

            validate_tuning_space(self.backend, self.search.tuning)
