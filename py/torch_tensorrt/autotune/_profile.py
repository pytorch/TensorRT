# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnosis: where a candidate's time goes, and which bottlenecks that suggests.

:func:`profile_calls` traces a few calls with the PyTorch profiler and
summarizes them per work unit. :func:`bottleneck_hints` turns that summary,
the candidate's structure and its rooflines into named hints, such as
``launch_overhead`` or ``dominant_kernel``. The hints only guide the
planners; whether a candidate is accepted is decided by measurement alone.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping

import torch

# Thresholds for reporting a bottleneck hint. They choose a starting point
# for the search; every change is still judged by whole-workload measurement.
# launch_overhead: at least this share of wall time is host work the GPU waits for.
LAUNCH_OVERHEAD_FRACTION = 0.2
# dominant_kernel: one kernel takes at least this share of GPU time.
DOMINANT_KERNEL_SHARE = 0.35
# roofline_gap: below this fraction of the memory-bandwidth floor, a
# memory-bound workload has room for better kernels or fewer bytes.
ROOFLINE_FRACTION = 0.8
# compute_gap: below this fraction of the compute floor (FLOPs at the peak
# rate), a compute-bound workload has room for faster math or fewer FLOPs.
COMPUTE_FRACTION = 0.8
# copy_overhead: memory copies take at least this share of GPU time.
COPY_SHARE = 0.03
# How many of the most expensive kernels a profile lists.
_TOP_KERNELS = 8


def profile_calls(
    call: Callable[[], Any], calls: int, units: int, wall_ms: float
) -> dict[str, Any]:
    """Trace ``calls`` calls of ``call`` and summarize them per work unit.

    ``device_ms`` sums GPU kernel and copy time. Tracing slows the host, so the
    host overhead compares it with ``wall_ms``, the untraced latency per unit:
    ``1 - device_ms / wall_ms`` estimates the share of wall time the GPU waits
    for host work. ``host_syncs`` counts device-to-host memory copies, the
    usual reason the host waits for the GPU: asynchronous copies count too,
    although they need not stall it, and synchronizations that copy nothing
    (stream, event or device synchronizations) do not. Spans that annotate a
    range of device work, rather than run it, are skipped, as PyTorch's own
    profiler tables skip them.
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    from torch_tensorrt.autotune._measure import synchronize

    synchronize()
    with torch.profiler.profile(activities=activities) as profiler:
        start = time.perf_counter()
        for _ in range(calls):
            call()
        synchronize()
        traced_s = time.perf_counter() - start
    return summarize_events(
        profiler.events(), max(calls * units, 1), wall_ms, traced_s * 1e3
    )


def summarize_events(
    events: Iterable[Any], per_unit: int, wall_ms: float, traced_ms: float
) -> dict[str, Any]:
    """Per-unit device time, copies, launches and top kernels of profiler ``events``."""
    kernels: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    device_us = copy_us = 0.0
    launches = syncs = 0
    for event in events:
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        if getattr(event, "is_user_annotation", False):
            # A record_function range mirrored on the device: it spans kernels
            # that are counted on their own.
            continue
        total = getattr(event, "device_time_total", None)
        if total is None:
            total = getattr(event, "cuda_time_total", 0.0)
        duration = float(total or 0.0)
        device_us += duration
        name = event.name
        if name.startswith("Memcpy") or name.startswith("Memset"):
            syncs += int("DtoH" in name)
            copy_us += duration
        else:
            launches += 1
        kernels[name][0] += duration
        kernels[name][1] += 1
    device_ms = device_us / 1e3 / per_unit
    top = sorted(kernels.items(), key=lambda item: -item[1][0])[:_TOP_KERNELS]
    return {
        "wall_ms": wall_ms,
        "traced_wall_ms": traced_ms / per_unit,
        "device_ms": device_ms,
        "host_overhead_fraction": (
            max(0.0, 1.0 - device_ms / wall_ms) if wall_ms > 0 else 0.0
        ),
        "kernels": launches / per_unit,
        "host_syncs": syncs / per_unit,
        "copy_ms": copy_us / 1e3 / per_unit,
        "top_kernels": [
            {
                "name": name if len(name) <= 120 else name[:117] + "...",
                "ms": total / 1e3 / per_unit,
                "share": total / device_us if device_us else 0.0,
                "calls": count / per_unit,
            }
            for name, (total, count) in top
        ],
    }


def bottleneck_hints(
    profile: Mapping[str, Any] | None,
    structure: Mapping[str, Any],
    values: Mapping[str, Any],
    roofline: Mapping[str, Any] | None = None,
    compute: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Bottleneck hints, strongest first.

    Each hint has a ``name`` (one of ``_knobs.BOTTLENECK_HINTS``), a
    ``strength`` in [0, 1] and a ``detail`` that says what was measured:

    * ``roofline_gap`` / ``compute_gap``: the latency is far above the
      memory-bandwidth or compute floor;
    * ``copy_overhead``: memory copies take a large share of GPU time;
    * ``launch_overhead``: the GPU waits for host work (kernel launches);
    * ``host_syncs``: device-to-host copies make the host wait for the GPU;
    * ``dominant_kernel``: one kernel takes a large share of the GPU time;
    * ``fallback``: the graph is split into several engines or PyTorch parts;
    * ``precision``: TensorRT layers run in FP32.

    The hints order the rule-based planner's search, strongest first, and
    the LLM planner reads them all; none decides whether a candidate is
    accepted. ``roofline`` (memory bandwidth) and ``compute`` (FLOPs at the
    peak rate) are the candidate's floors: when both are known, only the
    higher floor, the one that binds, can raise its gap hint.
    """
    found: list[dict[str, Any]] = []
    memory_floor = _floor_ms(roofline)
    compute_floor = _floor_ms(compute)
    if (
        roofline is not None
        and memory_floor is not None
        and (compute_floor is None or memory_floor >= compute_floor)
    ):
        fraction = float(roofline["fraction"])
        if fraction < ROOFLINE_FRACTION:
            found.append(
                {
                    "name": "roofline_gap",
                    "strength": min(1.0, 1.0 - fraction),
                    "detail": (
                        f"it runs at {fraction:.0%} of memory-bandwidth-bound speed: "
                        f"its latency is {1.0 / fraction:.2g}x the floor of "
                        f"{roofline['floor_ms']:.3g} ms "
                        f"({roofline['bytes_per_unit'] / 1e9:.3g} GB per unit at "
                        f"{roofline['peak_read_gbps']:.0f} GB/s)"
                    ),
                }
            )
    if (
        compute is not None
        and compute_floor is not None
        and (memory_floor is None or compute_floor > memory_floor)
    ):
        fraction = float(compute["fraction"])
        if fraction < COMPUTE_FRACTION:
            flops, peak = compute.get("flops_per_unit"), compute.get("peak_tflops")
            rate = (
                f" ({flops / 1e9:.4g} GFLOP per unit at {peak:.4g} TFLOP/s)"
                if flops and peak
                else ""
            )
            found.append(
                {
                    "name": "compute_gap",
                    "strength": min(1.0, 1.0 - fraction),
                    "detail": (
                        f"it runs at {fraction:.0%} of compute-bound speed: its "
                        f"latency is {1.0 / fraction:.2g}x the compute floor of "
                        f"{compute['floor_ms']:.3g} ms{rate}"
                    ),
                }
            )
    if profile:
        copies = float(profile.get("copy_ms", 0.0))
        device = float(profile.get("device_ms", 0.0))
        if device > 0 and copies / device >= COPY_SHARE:
            found.append(
                {
                    "name": "copy_overhead",
                    "strength": min(1.0, copies / device * 5),
                    "detail": (
                        f"{copies / device:.0%} of GPU time is memory copies "
                        f"({copies:.3g} ms per unit)"
                    ),
                }
            )
        overhead = float(profile.get("host_overhead_fraction", 0.0))
        if overhead >= LAUNCH_OVERHEAD_FRACTION:
            found.append(
                {
                    "name": "launch_overhead",
                    "strength": overhead,
                    "detail": (
                        f"{overhead:.0%} of wall time is host work the GPU waits for "
                        f"({profile.get('kernels', 0):.0f} kernel launches per unit)"
                    ),
                }
            )
        host_syncs = float(profile.get("host_syncs", 0.0))
        if host_syncs >= 1:
            found.append(
                {
                    "name": "host_syncs",
                    "strength": min(1.0, 0.5 + host_syncs / 100),
                    "detail": (
                        f"{host_syncs:.0f} device-to-host memory copies per unit: "
                        "a synchronous copy makes the host wait for the GPU, which "
                        "prevents CPU/GPU overlap and usually blocks CUDA-graph "
                        "capture (synchronizations that copy nothing are not "
                        "counted)"
                    ),
                }
            )
        top = (profile.get("top_kernels") or [{}])[0]
        if top.get("share", 0.0) >= DOMINANT_KERNEL_SHARE:
            found.append(
                {
                    "name": "dominant_kernel",
                    "strength": float(top["share"]),
                    "detail": (
                        f"{top['share']:.0%} of GPU time is spent in one kernel "
                        f"({top['calls']:.3g} launches per unit): {top['name']}"
                    ),
                }
            )
    partitions = int(structure.get("torch_partitions", 0) or 0)
    engines = int(structure.get("trt_engines", 0) or 0)
    if partitions or engines > 1:
        found.append(
            {
                "name": "fallback",
                "strength": min(1.0, 0.3 + 0.1 * (partitions + engines)),
                "detail": (
                    f"{engines} TensorRT engines and {partitions} PyTorch partitions; "
                    "every boundary costs a launch and blocks fusion"
                ),
            }
        )
    if values.get("engine.precision") == "fp32":
        found.append(
            {
                "name": "precision",
                "strength": 0.4,
                "detail": "TensorRT layers run in FP32",
            }
        )
    return sorted(found, key=lambda hint: -hint["strength"])


def _floor_ms(line: Mapping[str, Any] | None) -> float | None:
    """A roofline's floor in ms per unit, if it has one and a measured fraction."""
    if not line or line.get("fraction") is None or not line.get("floor_ms"):
        return None
    return float(line["floor_ms"])
