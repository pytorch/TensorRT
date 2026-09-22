"""trtexec-style JSONL cache for Global Performance Tuning sweeps."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class TuningCacheHeader:
    """Configuration saved on the first line of a tuning cache."""

    tuning_expr: str
    completed_iterations: int
    tuner_version: str = "unknown"
    accuracy_algorithm: str = "l0"
    accuracy_threshold: Optional[float] = None
    accuracy_atol: float = 1e-5
    accuracy_rtol: float = 1e-5
    searching_algorithm: str = "fast"


def subgraph_partition_key(module: torch.fx.GraphModule) -> str:
    """Return a stable graph-only key used to separate TRT partitions."""
    digest = hashlib.sha256()
    for node in module.graph.nodes:
        payload = {
            "op": node.op,
            "target": str(node.target),
            "name": node.name,
            "args": str(node.args),
            "kwargs": str(node.kwargs),
        }
        digest.update(repr(payload).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def resolve_partition_tuning_cache_path(
    base_path: Optional[str],
    module: torch.fx.GraphModule,
) -> Optional[str]:
    """Derive a per-partition cache path so multi-subgraph sweeps do not clobber.

    ``/tmp/tune.jsonl`` for partition ``abcd1234ef56`` becomes
    ``/tmp/tune.abcd1234ef56.jsonl``.
    """
    if not base_path:
        return None
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".jsonl"
    return f"{root}.{subgraph_partition_key(module)}{ext}"


def write_header(path: str, header: Dict[str, Any]) -> None:
    """Create a tuning cache and write its trtexec-like header."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as cache_file:
        cache_file.write(json.dumps(header) + "\n")


def append_iteration(
    path: str,
    *,
    iter_idx: int,
    build_route: str,
    crashed: bool,
    error_message: str = "",
    accuracy_loss: Optional[Dict[str, float]] = None,
    gpu_time_ms: Optional[float] = None,
) -> None:
    """Append one trtexec-style trial record."""
    finite_losses = (
        dict(accuracy_loss)
        if accuracy_loss is not None
        and all(math.isfinite(value) for value in accuracy_loss.values())
        else None
    )
    row: Dict[str, Any] = {
        "iter": iter_idx,
        "build_route": build_route,
        "crash": crashed,
        "error_message": error_message,
        "accuracy_loss": None if crashed else finite_losses,
        "gpu_time": (
            gpu_time_ms
            if not crashed and gpu_time_ms is not None and math.isfinite(gpu_time_ms)
            else None
        ),
    }
    with open(path, "a", encoding="utf-8") as cache_file:
        cache_file.write(json.dumps(row, allow_nan=False) + "\n")


def read_iterations(path: str) -> List[Dict[str, Any]]:
    """Read the per-trial records after the header line."""
    with open(path, "r", encoding="utf-8") as cache_file:
        lines = [line.strip() for line in cache_file if line.strip()]
    return [json.loads(line) for line in lines[1:]]


def read_cache(path: str) -> TuningCacheHeader:
    """Read a trtexec-style tuning cache header and completed row count."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"tuning_cache_file not found: {path}")
    with open(path, "r", encoding="utf-8") as cache_file:
        lines = [line.strip() for line in cache_file if line.strip()]
    if not lines:
        raise ValueError(f"Empty tuning cache file: {path}")

    header = json.loads(lines[0])
    accuracy = header.get("accuracy_parameter") or {}
    return TuningCacheHeader(
        tuning_expr=header.get("tuning_expr", ""),
        completed_iterations=max(0, len(lines) - 1),
        tuner_version=header.get("tuner_version", "unknown"),
        searching_algorithm=header.get("searching_algorithm", "fast"),
        accuracy_algorithm=header.get("accuracy_algorithm", "l0"),
        accuracy_threshold=accuracy.get("epsilon"),
        accuracy_atol=accuracy.get("atol", 1e-5),
        accuracy_rtol=accuracy.get("rtol", 1e-5),
    )


def read_iteration_gpu_times(path: str, max_iters: int) -> List[Optional[float]]:
    """Read recorded GPU times in global iteration order."""
    times: List[Optional[float]] = []
    for row in read_iterations(path)[:max_iters]:
        times.append(None if row.get("crash") else row.get("gpu_time"))
    return times
