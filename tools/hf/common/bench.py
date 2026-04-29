"""
Shared benchmarking harness for all HF model strategies.
"""
from __future__ import annotations

import timeit
from typing import Callable, Sequence

import numpy as np
import torch


WARMUP_ITERS = 5


def warmup_and_time(
    fn: Callable,
    args: Sequence,
    iterations: int = 10,
    warmup: int = WARMUP_ITERS,
) -> list[float]:
    """Run fn(*args) with warmup and return per-iteration wall-clock times (seconds)."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(iterations):
        start = timeit.default_timer()
        fn(*args)
        torch.cuda.synchronize()
        timings.append(timeit.default_timer() - start)
    return timings


def compute_stats(timings: list[float], batch_size: int = 1) -> dict:
    t = np.array(timings)
    fps = batch_size / t
    return {
        "mean_latency_ms": float(np.mean(t) * 1000),
        "median_latency_ms": float(np.median(t) * 1000),
        "p99_latency_ms": float(np.percentile(t, 99) * 1000),
        "std_latency_ms": float(np.std(t) * 1000),
        "mean_throughput": float(np.mean(fps)),
        "median_throughput": float(np.median(fps)),
    }
