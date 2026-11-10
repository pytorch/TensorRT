# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""A session's settings: what to tune, toward what, within what, and how to measure.

* :class:`WorkloadFactory`: which workload to tune (each worker rebuilds it).
* :class:`Goal`: the metric to improve and the smallest gain that counts.
* :class:`Limits`: the accuracy and memory every accepted candidate must meet.
* :class:`Budget`: how many trials and how much time the search may spend.
* :class:`Timing`: how a worker times a candidate.
* :class:`NoiseControl`: how the session tells a real gain from noise.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import operator
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:
    from torch_tensorrt.autotune._workload import Workload

# A config maps knob names to values. Knobs left at their default are
# omitted, so equal configurations have equal digests. An edited file appears
# as an ``edit.<name>`` key whose value is the SHA-256 of the edited source.
Config = dict[str, Any]

COMPARATORS: dict[str, Callable[[float, float], bool]] = {
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
}
METRICS = ("latency_ms", "throughput")


def canonical(value: Any) -> str:
    """Return the stable JSON text identifying ``value``."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    """The SHA-256 of ``value``'s canonical JSON: how a session recognizes a config."""
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _check_positive(name: str, value: float | None) -> None:
    """Raise ValueError unless ``value`` is None or a finite number above zero."""
    if value is not None and not (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    ):
        raise ValueError(f"{name} must be a positive number or None; got {value!r}")


@dataclass(frozen=True)
class WorkloadFactory:
    """The function that creates the workload: ``factory(**kwargs)``.

    Every worker process calls it to rebuild the workload, and a resumed
    session calls it again, so the session records it instead of a workload
    object. ``factory`` is ``"package.module:function"`` or
    ``"path/to/file.py:function"``; a file path is made absolute, and the
    file's directory is put on ``sys.path`` so its sibling imports work. The
    function returns a :class:`~torch_tensorrt.autotune.Workload`. ``kwargs``
    must be JSON data, because they are recorded with the session.
    """

    factory: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        module, sep, attribute = self.factory.rpartition(":")
        if not sep or not module or not attribute.isidentifier():
            raise ValueError(
                "Workload factory must be 'module:function' or "
                f"'path/to/file.py:function'; got {self.factory!r}"
            )
        if module.endswith(".py"):
            path = Path(module).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Workload factory file {path} does not exist")
            object.__setattr__(self, "factory", f"{path}:{attribute}")
        kwargs = dict(self.kwargs)
        try:
            canonical(kwargs)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Workload factory kwargs must be JSON data: {error}"
            ) from None
        object.__setattr__(self, "kwargs", kwargs)

    def load(self) -> Workload:
        """Import the factory and build the workload (in the calling process)."""
        from torch_tensorrt.autotune._workload import Workload

        module_name, _, attribute = self.factory.rpartition(":")
        if module_name.endswith(".py"):
            path = Path(module_name)
            if str(path.parent) not in sys.path:
                sys.path.insert(0, str(path.parent))
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot import workload factory file {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = module
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)
        workload = getattr(module, attribute)(**self.kwargs)
        if not isinstance(workload, Workload):
            raise TypeError(
                f"Workload factory {self.factory} returned {type(workload).__name__}, "
                "not a torch_tensorrt.autotune.Workload"
            )
        return workload

    def to_json(self) -> dict[str, Any]:
        return {"factory": self.factory, "kwargs": dict(self.kwargs)}


@dataclass(frozen=True)
class Goal:
    """What the session improves, and the smallest gain that counts.

    ``metric`` is ``"latency_ms"`` (milliseconds per work unit, lower is
    better) or ``"throughput"`` (items per second, higher is better). The
    workload defines its unit, for example one step of an iterative loop.
    A candidate replaces the best config only when it improves the metric by
    more than ``min_improvement`` (a fraction: 0.02 is 2%) and by more than
    the measured noise. The search stops early once the best config reaches
    ``target``.
    """

    metric: str = "latency_ms"
    target: float | None = None
    min_improvement: float = 0.02

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(
                f"Goal metric must be one of {METRICS}; got {self.metric!r}"
            )
        _check_positive("Goal target", self.target)
        if not (
            isinstance(self.min_improvement, (int, float))
            and 0 <= self.min_improvement < 1
        ):
            raise ValueError("Goal min_improvement must be in [0, 1)")

    @property
    def lower_is_better(self) -> bool:
        return self.metric == "latency_ms"

    def gain(self, before: float, after: float) -> float:
        """Relative improvement of ``after`` over ``before``; positive is better."""
        change = (after - before) / before
        return -change if self.lower_is_better else change

    def reached(self, value: float) -> bool:
        """Whether a metric ``value`` meets the target (never, without a target)."""
        if self.target is None:
            return False
        return value <= self.target if self.lower_is_better else value >= self.target


@dataclass(frozen=True)
class Limits:
    """Limits every accepted candidate must meet.

    ``accuracy`` maps an accuracy metric reported by the workload's
    ``compare`` to a bound, for example ``{"cosine_similarity": (">=",
    0.999)}``; metrics not listed keep the workload's default bounds.
    ``max_memory_mb`` bounds the candidate's GPU memory (see
    :class:`~torch_tensorrt.autotune.TuningResult` for how it is measured).
    """

    accuracy: Mapping[str, tuple[str, float]] = field(default_factory=dict)
    max_memory_mb: float | None = None

    def __post_init__(self) -> None:
        bounds = {}
        for name, bound in dict(self.accuracy).items():
            comparator, value = _checked_bound(name, bound)
            bounds[name] = (comparator, value)
        object.__setattr__(self, "accuracy", bounds)
        _check_positive("Limits max_memory_mb", self.max_memory_mb)

    def bounds(
        self, defaults: Mapping[str, tuple[str, float]]
    ) -> dict[str, tuple[str, float]]:
        """The workload's default accuracy bounds, overridden by these limits."""
        merged = {name: _checked_bound(name, bound) for name, bound in defaults.items()}
        merged.update(self.accuracy)
        return merged

    def to_json(self) -> dict[str, Any]:
        return {
            "accuracy": {name: list(bound) for name, bound in self.accuracy.items()},
            "max_memory_mb": self.max_memory_mb,
        }


def _checked_bound(name: str, bound: Any) -> tuple[str, float]:
    """``bound`` as ``(comparator, float)``; ValueError if it is not one."""
    if (
        not isinstance(bound, (tuple, list))
        or len(bound) != 2
        or bound[0] not in COMPARATORS
        or isinstance(bound[1], bool)
        or not isinstance(bound[1], (int, float))
        or not math.isfinite(bound[1])
    ):
        raise ValueError(
            f"Accuracy bound for {name!r} must be (comparator, number) with a "
            f"comparator in {tuple(COMPARATORS)}; got {bound!r}"
        )
    return bound[0], float(bound[1])


def violations(
    metrics: Mapping[str, float], bounds: Mapping[str, tuple[str, float]]
) -> list[str]:
    """Describe every bound ``metrics`` miss; a missing or non-finite metric misses."""
    missed = []
    for name, (comparator, value) in sorted(bounds.items()):
        actual = metrics.get(name)
        if not isinstance(actual, (int, float)) or not math.isfinite(actual):
            missed.append(f"{name} is unavailable (needs {comparator} {value:g})")
        elif not COMPARATORS[comparator](actual, value):
            missed.append(f"{name}={actual:.6g} (needs {comparator} {value:g})")
    return missed


@dataclass(frozen=True)
class Budget:
    """How much the search may spend.

    ``max_trials`` counts the measured candidate trials; the baseline, the
    noise runs, the head-to-head runs, the final check and invalid proposals
    do not count. ``max_time_s`` limits the wall-clock time, summed across
    resumes. It is checked before each trial: a trial that is running when
    the time is up still finishes, and the final check and the export still
    run afterwards. ``patience`` stops the search after that many trials in a
    row without an accepted candidate. ``trial_timeout_s`` kills any worker
    process (reference, describe, trial, noise run, head-to-head run, final
    check run or export) that runs for longer. ``prepare_timeout_s`` does the
    same for each worker that prepares a knob value before the search (see
    :class:`~torch_tensorrt.autotune.Knob` ``prepare``), which may generate
    region kernels for a long time; preparation time counts toward
    ``max_time_s`` like every other step.
    """

    max_trials: int = 8
    max_time_s: float | None = None
    patience: int | None = None
    trial_timeout_s: float = 1800.0
    prepare_timeout_s: float = 7200.0

    def __post_init__(self) -> None:
        for name in ("max_trials", "patience"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"Budget {name} must be a non-negative integer")
        _check_positive("Budget max_time_s", self.max_time_s)
        _check_positive("Budget trial_timeout_s", self.trial_timeout_s)
        _check_positive("Budget prepare_timeout_s", self.prepare_timeout_s)


@dataclass(frozen=True)
class Timing:
    """How a worker times a candidate; every candidate is timed the same way.

    The worker first makes ``warmup`` untimed calls. It then times
    ``rounds`` rounds. A round is ``calls`` back-to-back calls between two
    device synchronizations (one before the first call, one after the last)
    and gives one sample of the latency per work unit. The candidate's
    latency is the median of the samples, reported with their quartiles.
    ``profile_calls`` more calls run under the PyTorch profiler, to find the
    bottlenecks; profiling slows them down, so they do not count toward the
    latency.
    """

    warmup: int = 5
    rounds: int = 15
    calls: int = 3
    profile_calls: int = 3

    def __post_init__(self) -> None:
        for name in ("warmup", "rounds", "calls", "profile_calls"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Timing {name} must be a non-negative integer")
        if self.rounds < 3 or self.calls < 1:
            raise ValueError("Timing needs at least 3 rounds of at least 1 call")


@dataclass(frozen=True)
class NoiseControl:
    """How the session tells a real gain from run-to-run noise.

    The same config measured in two fresh processes can differ by a few
    percent (engine builds pick different tactics, GPU clocks vary, other
    jobs share the GPU), much more than the samples of one run differ from
    each other. So one run cannot show a gain of a few percent. Three
    measures guard against that:

    * **Noise runs.** After the baseline's first measurement, ``noise_runs``
      more fresh-process runs of the baseline show how far its metric moves
      between runs (the spread). A candidate must then improve by more than
      the noise threshold, ``max(goal.min_improvement, noise_factor *
      spread)``.
    * **Head-to-head runs.** A candidate that beats the best config by more
      than the noise threshold in its first measurement then runs
      head-to-head against it: ``head_to_head_pairs`` pairs of fresh runs,
      the best config and the candidate back to back, alternating which runs
      first. The candidate is accepted only if every run passes its checks
      (status, accuracy, memory, stability), the candidate gains in every
      pair, and the median gain clears the noise threshold. ``0`` decides on
      the first measurement alone. An even number of pairs balances the
      order (with an odd number, the best config runs first once more). The
      final check runs ``final_check_pairs`` pairs of the baseline and the
      best config the same way.
    * **A quiet GPU and stable samples.** Before timing, each worker waits
      for the GPU to be idle (at most 5% busy, by nvidia-smi, in two
      consecutive one-second samples). It times again, up to ``retries``
      times, when the samples' interquartile range exceeds ``max_spread`` of
      their median or the GPU was busy. The first attempt waits up to
      ``quiet_gpu_s``; a retry waits for what is left of it, but always
      looks for at least 3 s, so a busy GPU is still noticed. A measurement
      that stays unstable is never accepted, and an unstable baseline ends
      the session. ``quiet_gpu_s=0`` turns the wait off. Keep the worst case
      (``quiet_gpu_s``, plus 3 s per retry, plus the build and the timing)
      below the budget's ``trial_timeout_s``.
    """

    noise_runs: int = 2
    head_to_head_pairs: int = 2
    noise_factor: float = 2.0
    quiet_gpu_s: float = 600.0
    retries: int = 2
    max_spread: float = 0.03

    def __post_init__(self) -> None:
        for name in ("noise_runs", "head_to_head_pairs", "retries"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"NoiseControl {name} must be a non-negative integer")
        for name in ("noise_factor", "max_spread"):
            value = getattr(self, name)
            if not (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value > 0
            ):
                raise ValueError(
                    f"NoiseControl {name} must be a positive number; got {value!r}"
                )
        if not (
            isinstance(self.quiet_gpu_s, (int, float))
            and not isinstance(self.quiet_gpu_s, bool)
            and math.isfinite(self.quiet_gpu_s)
            and self.quiet_gpu_s >= 0
        ):
            raise ValueError("NoiseControl quiet_gpu_s must be a non-negative number")
