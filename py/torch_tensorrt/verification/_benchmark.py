# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Serial inference timing with explicit CUDA and synchronized wall scopes."""

from __future__ import annotations

import math
import statistics
import time
from typing import Sequence

import torch
from torch_tensorrt.reporting import (
    BenchmarkProtocol,
    BenchmarkReport,
    TimingMeasurements,
)

from ._execution import (
    _device,
    _isolated_module,
    _prepare,
    _preserve_rng,
    _reject_module_input_aliases,
)
from ._inputs import InputSample, VerificationError, _fingerprint


def _statistics(samples: list[float], percentiles: Sequence[float]) -> dict[str, float]:
    """Summarize timing samples with fixed statistics and linear percentiles."""
    ordered = sorted(samples)
    result = {
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }
    for percentile in percentiles:
        position = percentile / 100 * (len(ordered) - 1)
        low, high = math.floor(position), math.ceil(position)
        value = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
        result[f"p{percentile:g}"] = value
    return result


def _measure(
    module: torch.nn.Module,
    sample: InputSample,
    device: torch.device,
    seed: int,
    warmup: int,
    iterations: int,
    percentiles: Sequence[float],
) -> TimingMeasurements:
    """Measure repeated forward calls while excluding input reset and checks."""
    wall: list[float] = []
    cuda: list[float] = []
    for index in range(warmup + iterations):
        prepared, guard = _prepare(module, sample, seed)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            with torch.cuda.device(device):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                # Allocate the event handles outside the measured interval.
                start.record(torch.cuda.current_stream(device))
                end.record(torch.cuda.current_stream(device))
                end.synchronize()
                with torch.inference_mode():
                    wall_start = time.perf_counter_ns()
                    start.record(torch.cuda.current_stream(device))
                    output = module(*prepared.args, **prepared.kwargs)
                    end.record(torch.cuda.current_stream(device))
                    torch.cuda.synchronize(device)
                    wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
                cuda_ms = start.elapsed_time(end)
        else:
            with torch.inference_mode():
                wall_start = time.perf_counter_ns()
                output = module(*prepared.args, **prepared.kwargs)
                wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
            cuda_ms = None
        # Keep outputs alive through synchronization; checking/resetting is untimed.
        guard.check()
        del output
        if index >= warmup:
            if not math.isfinite(wall_ms) or wall_ms <= 0:
                raise VerificationError(
                    "Timer returned a nonpositive or nonfinite wall duration"
                )
            wall.append(wall_ms)
            if cuda_ms is not None:
                if not math.isfinite(cuda_ms) or cuda_ms < 0:
                    raise VerificationError("Timer returned an invalid CUDA duration")
                cuda.append(cuda_ms)
    return TimingMeasurements(
        wall_samples_ms=wall,
        device_samples_ms=cuda,
        wall_statistics=_statistics(wall, percentiles),
        device_statistics=_statistics(cuda, percentiles) if cuda else {},
    )


def benchmark(
    reference: torch.nn.Module,
    compiled: torch.nn.Module,
    inputs: InputSample,
    *,
    warmup: int = 10,
    iterations: int = 100,
    percentiles: Sequence[float] = (50, 90, 99),
    seed: int = 0,
    batch_size: int | None = None,
    gpu_isolation: str = "unknown",
    job_id: str | None = None,
) -> BenchmarkReport:
    """Measure prepared forward calls, with per-call synchronization on CUDA.

    Input cloning, state checks, transfers, preprocessing, and model copying are
    outside both timers. CUDA events cover the selected device's current stream;
    work launched on other streams must join that stream before forward returns.
    The device is inferred from inputs and registered module state; forward calls
    must not create work on another device.
    Speedup is reference median / compiled median in the same timing scope.
    Throughput is inverse serial wall latency (optionally scaled by batch_size),
    not a concurrent saturation measurement. GPU isolation is caller supplied.
    """
    if not isinstance(inputs, InputSample):
        raise VerificationError("inputs must be an InputSample")
    for value, name, minimum in ((warmup, "warmup", 0), (iterations, "iterations", 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise VerificationError(f"{name} must be an integer >= {minimum}")
    if batch_size is not None and (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise VerificationError("batch_size must be a positive integer or None")
    try:
        percentiles = list(percentiles)
    except TypeError as error:
        raise VerificationError("percentiles must be an iterable of numbers") from error
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or value > 100
        for value in percentiles
    ):
        raise VerificationError("percentiles must be finite numbers in [0, 100]")
    if len(set(percentiles)) != len(percentiles):
        raise VerificationError("percentiles must be unique")
    if gpu_isolation not in {"unknown", "isolated", "shared"}:
        raise VerificationError("gpu_isolation must be unknown, isolated, or shared")
    _reject_module_input_aliases(reference, inputs)
    _reject_module_input_aliases(compiled, inputs)
    device = _device(inputs, reference, compiled)
    protocol = BenchmarkProtocol(
        warmup=warmup,
        iterations=iterations,
        percentiles=percentiles,
        gpu_isolation=gpu_isolation,
        model_mode="eval_or_frozen_export_capture",
        synchronization=(
            "per_invocation" if device.type == "cuda" else "synchronous_cpu"
        ),
    )
    transfers = {"included": False, "inputs_prepared_on_device": True}
    preprocessing = {
        "included": False,
        "input_fingerprint": _fingerprint(inputs),
        "profile": inputs.profile,
        "seed_reset": "per_invocation",
        "input_reset": "alias_preserving_snapshot",
        "model_mode": "eval_or_frozen_export_capture",
        "synchronization": (
            "per_invocation" if device.type == "cuda" else "synchronous_cpu"
        ),
        "speedup_method": "reference_median_over_compiled_median",
        "batch_size": batch_size,
    }
    with _preserve_rng(seed):
        reference_copy = _isolated_module(reference)
        compiled_copy = _isolated_module(compiled)
        reference_timings = _measure(
            reference_copy, inputs, device, seed, warmup, iterations, percentiles
        )
        compiled_timings = _measure(
            compiled_copy, inputs, device, seed, warmup, iterations, percentiles
        )
    speedup = {}
    throughput = {}
    speedup["wall"] = (
        reference_timings.wall_statistics["median"]
        / compiled_timings.wall_statistics["median"]
    )
    if (
        reference_timings.device_samples_ms
        and compiled_timings.device_statistics["median"] > 0
    ):
        speedup["device"] = (
            reference_timings.device_statistics["median"]
            / compiled_timings.device_statistics["median"]
        )
    for name, measurement in (
        ("reference", reference_timings),
        ("compiled", compiled_timings),
    ):
        rate = 1000 / measurement.wall_statistics["median"]
        throughput[f"{name}_invocations_per_second"] = rate
        if batch_size is not None:
            throughput[f"{name}_items_per_second"] = rate * batch_size
    return BenchmarkReport(
        job_id=job_id or "",
        device=str(device),
        seed=seed,
        protocol=protocol,
        reference=reference_timings,
        compiled=compiled_timings,
        speedup=speedup,
        throughput=throughput,
        transfers=transfers,
        preprocessing=preprocessing,
    )
