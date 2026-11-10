# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Measuring a candidate: the same checks and timing for every candidate.

:func:`measure_candidate` runs inside a worker process and never sees the
planner. It builds the candidate, checks its outputs against the reference,
times it, profiles it and, at the end of a session, exports it. A
candidate's result is what this module measures, not what the planner
expected. :func:`prepare_candidate` only builds one, before the search, so
that what its build generates (region kernels) is ready for later builds.
"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import statistics
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch
from torch_tensorrt.autotune._profile import bottleneck_hints, profile_calls
from torch_tensorrt.autotune._settings import Timing
from torch_tensorrt.autotune._workload import Candidate, Workload


def summarize_samples(samples: list[float]) -> dict[str, float]:
    """The median, quartiles, extremes, mean and count of latency samples."""
    ordered = sorted(samples)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "median": statistics.median(ordered),
        "p25": quartiles[0],
        "p75": quartiles[2],
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "count": len(ordered),
    }


def synchronize() -> None:
    """Wait for the GPU to finish all queued work (nothing to do without CUDA)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_utilization() -> float | None:
    """This process's GPU's utilization in percent from nvidia-smi; None if unknown."""
    binary = shutil.which("nvidia-smi")
    if binary is None or not torch.cuda.is_available():
        return None
    query = ["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
    with contextlib.suppress(Exception):
        uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        query.insert(0, "--id=" + (uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"))
    try:
        output = subprocess.run(
            [binary, *query], capture_output=True, text=True, timeout=30, check=True
        ).stdout
        return float(output.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def wait_for_quiet_gpu(
    max_wait_s: float, *, threshold: float = 5.0, settle: int = 2
) -> dict[str, Any]:
    """Wait until the GPU is idle for ``settle`` consecutive one-second samples.

    Idle means at most ``threshold`` percent busy. Call it between this
    process's own work, so any load it sees belongs to other jobs. Returns
    the seconds waited and ``quiet``: whether the GPU went idle, or None
    when its utilization cannot be read.
    """
    record: dict[str, Any] = {"waited_s": 0.0, "quiet": None}
    if max_wait_s <= 0:
        return record
    synchronize()
    start = time.monotonic()
    quiet = 0
    while True:
        utilization = gpu_utilization()
        if utilization is None:
            return record
        quiet = quiet + 1 if utilization <= threshold else 0
        waited = time.monotonic() - start
        if quiet >= settle or waited >= max_wait_s:
            record.update(waited_s=waited, quiet=quiet >= settle)
            return record
        time.sleep(1.0)


# Once the quiet-GPU wait is used up, each further timing attempt still
# watches the GPU this long, so it notices a busy GPU.
_MIN_GPU_LOOK_S = 3.0

# This GPU's measured memory bandwidth, cached for the process.
_PEAK_READ_GBPS: float | None = None


def peak_read_gbps() -> float | None:
    """Achievable DRAM read bandwidth of this GPU in GB/s, measured once per process.

    A reduction over 512 MiB, more than any GPU's L2 cache, reads every byte
    from device memory once.
    """
    global _PEAK_READ_GBPS
    if _PEAK_READ_GBPS is None and torch.cuda.is_available():
        try:
            data = torch.ones(256 * 2**20, dtype=torch.float16, device="cuda")
        except RuntimeError:  # not enough free memory
            return None
        best_ms = float("inf")
        for _ in range(6):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            data.sum(dtype=torch.float32)
            end.record()
            end.synchronize()
            best_ms = min(best_ms, start.elapsed_time(end))
        _PEAK_READ_GBPS = data.numel() * data.element_size() / best_ms / 1e6
        del data
        torch.cuda.empty_cache()
    return _PEAK_READ_GBPS


def memory_roofline(bytes_per_unit: float, latency_ms: float) -> dict[str, Any] | None:
    """The memory-bandwidth floor of a work unit, and how close the latency gets.

    The floor is the time to read ``bytes_per_unit`` at this GPU's measured
    bandwidth; ``fraction`` is the floor over the latency (1.0 would be the
    floor itself).
    """
    peak = peak_read_gbps()
    if not peak or bytes_per_unit <= 0:
        return None
    floor_ms = bytes_per_unit / (peak * 1e9) * 1e3
    return {
        "bytes_per_unit": float(bytes_per_unit),
        "peak_read_gbps": peak,
        "floor_ms": floor_ms,
        "fraction": floor_ms / latency_ms if latency_ms > 0 else None,
    }


def _positive_number(value: Any) -> float | None:
    """``value`` as a float if it is a finite number above zero; otherwise None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


def compute_roofline(
    candidate: Any, structure: Mapping[str, Any], latency_ms: float
) -> dict[str, Any] | None:
    """The compute floor of a candidate and how close its latency gets, if it has one.

    The floor is ``flops_per_unit / peak_tflops``: the candidate's own
    attributes first, else its structure's ``gflop_per_unit`` and
    ``peak_tflops``. A structure's ``compute_floor_ms`` stands for the floor
    unless the candidate declares its ``flops_per_unit``.
    """
    declared = _positive_number(getattr(candidate, "flops_per_unit", None))
    gflop = _positive_number(structure.get("gflop_per_unit"))
    flops = declared or (gflop * 1e9 if gflop else None)
    peak = _positive_number(getattr(candidate, "peak_tflops", None))
    peak = peak or _positive_number(structure.get("peak_tflops"))
    floor_ms = None if declared else _positive_number(structure.get("compute_floor_ms"))
    if floor_ms is None and flops and peak:
        floor_ms = flops / (peak * 1e9)
    if floor_ms is None or latency_ms <= 0:
        return None
    return {
        "flops_per_unit": flops,
        "peak_tflops": peak,
        "floor_ms": floor_ms,
        "fraction": floor_ms / latency_ms,
    }


def export_unsupported(error: BaseException) -> bool:
    """Whether an export error means the candidate cannot be saved, not that saving broke.

    :meth:`Candidate.export` raises ``NotImplementedError`` for that, and
    ``torch_tensorrt.save`` raises a ``RegionError`` saying an
    ``execute_in_torch`` region's backend "is not supported".
    """
    if isinstance(error, NotImplementedError):
        return True
    try:
        from torch_tensorrt.region import RegionError
    except ImportError:  # pragma: no cover - the region API ships with the package
        return False
    return isinstance(error, RegionError) and "is not supported" in str(error)


def _spread(samples: list[float]) -> float:
    """Interquartile range relative to the median."""
    summary = summarize_samples(samples)
    return (summary["p75"] - summary["p25"]) / summary["median"]


def _time_rounds(candidate: Candidate, timing: Timing) -> list[float]:
    """Milliseconds per work unit, one sample per round of back-to-back calls."""
    units = max(int(candidate.units), 1)
    samples = []
    for _ in range(timing.rounds):
        synchronize()
        start = time.perf_counter()
        for _ in range(timing.calls):
            candidate()
        synchronize()
        samples.append((time.perf_counter() - start) * 1e3 / (timing.calls * units))
    return samples


def _time_until_stable(
    candidate: Candidate,
    timing: Timing,
    *,
    quiet_gpu_s: float,
    retries: int,
    max_spread: float,
) -> tuple[list[float], dict[str, Any]]:
    """Time ``candidate``, and time it again while the samples are unstable.

    Each attempt waits for a quiet GPU, warms up and times the rounds. An
    attempt is stable when its samples spread at most ``max_spread`` and the
    GPU was not seen busy; up to ``retries`` more attempts follow an
    unstable one. Returns the last attempt's samples and the timing-stability
    record of the attempts (``stable`` says whether the last one was stable).
    """
    spreads: list[float] = []
    waited_s, quiet = 0.0, None
    for _ in range(1 + retries):
        # quiet_gpu_s bounds the whole wait; once it is used up, a short look
        # still tells whether the GPU is busy for this attempt.
        gate = wait_for_quiet_gpu(
            max(quiet_gpu_s - waited_s, _MIN_GPU_LOOK_S) if quiet_gpu_s > 0 else 0.0
        )
        waited_s += gate["waited_s"]
        quiet = gate["quiet"]
        # Warm up right before timing: the wait may have let the clocks drop.
        for _ in range(timing.warmup):
            candidate()
        samples = _time_rounds(candidate, timing)
        spreads.append(_spread(samples))
        if spreads[-1] <= max_spread and quiet is not False:
            break
    return samples, {
        "attempts": len(spreads),
        "spreads": spreads,
        "gpu_wait_s": waited_s,
        "gpu_quiet": quiet,
        "stable": spreads[-1] <= max_spread and quiet is not False,
    }


def process_gpu_memory_mb() -> float | None:
    """GPU memory this process holds, from NVML via nvidia-smi; None if unknown.

    It includes TensorRT's own allocations, which PyTorch's allocator does not
    see. In containers whose process IDs differ from the host's, it is unknown.
    """
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        output = subprocess.run(
            [
                binary,
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in output.splitlines():
        pid, _, used = line.partition(",")
        if pid.strip() == str(os.getpid()):
            with contextlib.suppress(ValueError):
                return float(used)
    return None


def _failed_result(stage: str, error: BaseException) -> dict[str, Any]:
    """The result of a measurement that failed in ``stage``, with its error."""
    lines = traceback.format_exception_only(type(error), error)
    message = "".join(lines).strip()
    return {
        "status": "failed",
        "stage": stage,
        "error": message if len(message) <= 2000 else message[:1997] + "...",
    }


def prepare_candidate(workload: Workload, values: Mapping[str, Any]) -> dict[str, Any]:
    """Build one candidate and close it, without checking or timing it.

    The build generates or loads what later builds of the config reuse,
    such as region kernels, which the worker reports. The result holds
    ``build_s`` and ``structure``, or, if the build fails, stage ``build``
    and the error.
    """
    result: dict[str, Any] = {"status": "ok"}
    candidate: Candidate | None = None
    try:
        start = time.perf_counter()
        candidate = workload.build(values)
        result["build_s"] = time.perf_counter() - start
        result["structure"] = candidate.describe()
    except Exception as error:  # noqa: BLE001 - every failure is recorded
        partial = {key: result[key] for key in ("build_s",) if key in result}
        result = {**_failed_result("build", error), **partial}
        traceback.print_exc()
    finally:
        if candidate is not None:
            with contextlib.suppress(Exception):
                candidate.close()
    return result


def measure_candidate(
    workload: Workload,
    values: Mapping[str, Any],
    reference: Any,
    timing: Timing,
    *,
    profile: bool = True,
    export_dir: Path | None = None,
    noise_control: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build, check, time and (optionally) profile and export one candidate.

    The stages run in this order: ``build``, ``setup``, ``check`` (the
    outputs against the reference), ``time``, ``check`` again, ``profile``
    and ``export``. A failure records its stage and error and ends the
    measurement. An export the candidate cannot make (see
    :func:`export_unsupported`) also records ``export_unsupported``.
    Accuracy is checked on the first call and again after timing, so outputs
    replayed from a captured CUDA graph are checked too.

    ``noise_control`` (``quiet_gpu_s``, ``retries``, ``max_spread``; see
    :class:`~torch_tensorrt.autotune.NoiseControl`) makes timing wait for an
    idle GPU and time again when the samples spread too widely or the GPU
    was busy. The first attempt waits up to ``quiet_gpu_s``; each later one
    waits for what is left of it, but always looks for at least 3 s, so a
    busy GPU is still noticed once the wait is used up. A measurement is
    stable when its last attempt's spread is within ``max_spread`` and the
    GPU was not seen busy before it; ``timing_stability`` records the
    attempts. A candidate with ``bytes_per_unit`` also gets its
    memory-bandwidth ``roofline``, and one with a compute floor (see
    :class:`~torch_tensorrt.autotune.Candidate`) its ``compute_roofline``.
    Every result, failed or not, gets its ``bottleneck_hints``.
    """
    settings = dict(noise_control or {})
    quiet_gpu_s = float(settings.get("quiet_gpu_s", 0.0))
    max_spread = float(settings.get("max_spread", float("inf")))
    retries = int(settings.get("retries", 0))
    result: dict[str, Any] = {"status": "ok"}
    candidate: Candidate | None = None
    stage = "build"
    try:
        start = time.perf_counter()
        candidate = workload.build(values)
        result["build_s"] = time.perf_counter() - start
        result["structure"] = candidate.describe()
        result["units"] = int(candidate.units)
        result["items"] = int(candidate.items)
        stage = "setup"
        candidate.setup()
        stage = "check"
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        result["accuracy"] = workload.compare(candidate(), reference)
        stage = "time"
        samples, result["timing_stability"] = _time_until_stable(
            candidate,
            timing,
            quiet_gpu_s=quiet_gpu_s,
            retries=retries,
            max_spread=max_spread,
        )
        stage = "check"
        result["accuracy_after_timing"] = workload.compare(candidate(), reference)
        latency = summarize_samples(samples)
        result["latency_ms"] = latency
        result["throughput"] = result["items"] * 1e3 / latency["median"]
        result["samples_ms"] = samples
        result["memory"] = {
            "process_gpu_mb": process_gpu_memory_mb() if cuda else None,
            "torch_peak_mb": (
                torch.cuda.max_memory_allocated() / 2**20 if cuda else None
            ),
        }
        # After the memory readings: the bandwidth probe allocates 512 MiB.
        bytes_per_unit = getattr(candidate, "bytes_per_unit", None)
        if bytes_per_unit:
            line = memory_roofline(float(bytes_per_unit), latency["median"])
            if line is not None:
                result["roofline"] = line
        compute = compute_roofline(
            candidate, result.get("structure") or {}, latency["median"]
        )
        if compute is not None:
            result["compute_roofline"] = compute
        if profile and timing.profile_calls:
            stage = "profile"
            result["profile"] = profile_calls(
                candidate, timing.profile_calls, result["units"], latency["median"]
            )
        if export_dir is not None:
            stage = "export"
            export_dir.mkdir(parents=True, exist_ok=True)
            result["exported"] = candidate.export(export_dir)
    except Exception as error:  # noqa: BLE001 - every failure is recorded
        partial = {
            key: result[key]
            for key in ("build_s", "structure", "units", "items")
            if key in result
        }
        result = {**_failed_result(stage, error), **partial}
        if stage == "export":
            result["export_unsupported"] = export_unsupported(error)
        traceback.print_exc()
    finally:
        if candidate is not None:
            with contextlib.suppress(Exception):
                candidate.close()
    result["bottleneck_hints"] = bottleneck_hints(
        result.get("profile"),
        result.get("structure", {}),
        values,
        result.get("roofline"),
        result.get("compute_roofline"),
    )
    return result
