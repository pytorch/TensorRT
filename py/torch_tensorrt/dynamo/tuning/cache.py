"""JSONL tuning cache (trtexec-inspired header + per-iteration lines)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

_LOGGER = logging.getLogger(__name__)


@dataclass
class TuningCacheHeader:
    """Header of a tuning cache file."""

    argv_like: Dict[str, Any]
    tuning_expr: str
    completed_iterations: int
    tuner_version: str = "unknown"


def subgraph_partition_key(module: torch.fx.GraphModule) -> str:
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
    key = subgraph_partition_key(module)
    return f"{root}.{key}{ext}"


def write_header(path: str, header: Dict[str, Any]) -> None:
    """Write the header of a tuning cache file."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header, sort_keys=False) + "\n")


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
    """Append an iteration to a tuning cache file."""
    row: Dict[str, Any] = {
        "iter": iter_idx,
        "build_route": build_route,
        "crash": crashed,
        "error_message": error_message,
        "accuracy_loss": None if crashed or accuracy_loss is None else accuracy_loss,
        "gpu_time": None if crashed or gpu_time_ms is None else gpu_time_ms,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def read_cache(path: str) -> TuningCacheHeader:
    """Read the header of a tuning cache file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"tuning_cache_file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty tuning cache file: {path}")
    header = json.loads(lines[0])
    return TuningCacheHeader(
        argv_like=header,
        tuning_expr=header.get("tuning_expr", ""),
        completed_iterations=max(0, len(lines) - 1),
        tuner_version=header.get("tuner_version", "unknown"),
    )


def read_iteration_gpu_times(path: str, max_iters: int) -> List[Optional[float]]:
    """Read the GPU times of a tuning cache file."""
    times: List[Optional[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    for line in lines[1 : 1 + max_iters]:
        row = json.loads(line)
        if row.get("crash"):
            times.append(None)
        else:
            times.append(row.get("gpu_time"))
    return times
