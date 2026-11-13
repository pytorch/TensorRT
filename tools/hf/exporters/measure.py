from __future__ import annotations

import time
from collections.abc import Callable, Mapping

import torch


def parity(name: str, eager: torch.Tensor, trt: torch.Tensor) -> None:
    """Print eager-vs-TRT numeric agreement for one tensor."""
    a, b = eager.float(), trt.float()
    delta = a - b
    diff = delta.abs()
    rel_l2 = delta.norm() / b.norm().clamp_min(1e-8)
    close = torch.isclose(a, b, rtol=1e-2, atol=1e-2).float().mean() * 100
    print(
        f"{name:<36} mean_abs={float(diff.mean()):.6f}  "
        f"max_abs={float(diff.max()):.6f}  rel_l2={float(rel_l2):.4f}  "
        f"close%={float(close):.1f}"
    )


def cuda_ms(fn: Callable[[], object], *, warmup: int = 10, iters: int = 100) -> float:
    """Average runtime of ``fn`` in milliseconds (CUDA events, else wall time)."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / iters
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) * 1000.0 / iters


def speedup(eager_ms: float, trt_ms: float) -> str:
    if eager_ms <= 0.0 or trt_ms <= 0.0:
        return "n/a"
    return f"{eager_ms / trt_ms:.3f}x"


def print_bench(bench: Mapping[str, tuple[float, float]]) -> None:
    """Print per-component CUDA timings collected during ``export()``."""
    if not bench:
        return
    eager_total = trt_total = 0.0
    for name, (eager_ms, trt_ms) in bench.items():
        print(f"{name} eager execute: {eager_ms:.3f} ms")
        print(f"{name} trt execute: {trt_ms:.3f} ms")
        print(f"{name} speedup: {speedup(eager_ms, trt_ms)}")
        eager_total += eager_ms
        trt_total += trt_ms
    print(f"total eager execute: {eager_total:.3f} ms")
    print(f"total trt execute: {trt_total:.3f} ms")
    print(f"total speedup: {speedup(eager_total, trt_total)}")
