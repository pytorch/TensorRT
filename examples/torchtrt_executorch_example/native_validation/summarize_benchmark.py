# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3
"""Summarize checked C++ timing logs offline; no Torch or target Python needed."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_run(directory):
    if not (directory / "PASS").is_file():
        raise ValueError(f"No PASS marker: {directory}")
    metadata = {row["key"]: row["value"] for row in rows(directory / "metadata.tsv")}
    comparisons = rows(directory / "comparisons.tsv")
    phases = {row["phase"] for row in comparisons}
    if phases != {"first", "after_warmup", "after_measurement"} or any(
        int(row["mismatches"]) != 0 or not math.isfinite(float(row["max_abs_error"]))
        for row in comparisons
    ):
        raise ValueError(f"Missing or failing correctness checks: {directory}")
    group_outputs = [
        {
            (row["output"], row["elements"])
            for row in comparisons
            if row["phase"] == phase
        }
        for phase in sorted(phases)
    ]
    if not group_outputs[0] or any(
        group != group_outputs[0] for group in group_outputs
    ):
        raise ValueError(f"Output comparisons differ across gates: {directory}")
    timing = rows(directory / "latency.tsv")
    measured = [float(row["wall_ms"]) for row in timing if row["phase"] == "measured"]
    warmup = [row for row in timing if row["phase"] == "warmup"]
    first = [float(row["wall_ms"]) for row in timing if row["phase"] == "first"]
    if (
        len(first) != 1
        or len(measured) != int(metadata["iterations"])
        or len(warmup) != int(metadata["warmup"])
        or not measured
        or any(
            not math.isfinite(float(row["wall_ms"])) or float(row["wall_ms"]) <= 0
            for row in timing
        )
    ):
        raise ValueError(f"Invalid/incomplete latency samples: {directory}")
    maps = (directory / "loaded-maps.txt").read_text()
    if any(
        name in maps
        for name in ("libtorch", "libc10", "libpython", "/site-packages/", "/venvs/")
    ):
        raise ValueError(f"Unexpected Python/Torch native dependency: {directory}")
    memory = rows(directory / "memory.tsv")
    return {
        "run": directory.name,
        "metadata": metadata,
        "startup_phases_ms": {
            row["phase"]: float(row["wall_ms"])
            for row in rows(directory / "phases.tsv")
        },
        "first_inference_ms": first[0],
        "measured_count": len(measured),
        "mean_ms": statistics.mean(measured),
        "median_ms": statistics.median(measured),
        "p95_ms": percentile(measured, 0.95),
        "min_ms": min(measured),
        "max_ms": max(measured),
        "stdev_ms": statistics.stdev(measured) if len(measured) > 1 else 0,
        "max_abs_error": max(float(row["max_abs_error"]) for row in comparisons),
        "compared_elements_per_gate": sum(int(count) for _, count in group_outputs[0]),
        "memory_snapshots": [
            {key: value if key == "phase" else int(value) for key, value in row.items()}
            for row in memory
        ],
    }


def summarize(directory):
    paths = sorted(path for path in directory.glob("repeat-*") if path.is_dir())
    if not paths:
        raise ValueError(f"No repeat directories: {directory}")
    runs = [summarize_run(path) for path in paths]
    reference = runs[0]["metadata"]
    if any(run["metadata"] != reference for run in runs[1:]):
        raise ValueError("Repetitions do not have identical model/fixture/settings")
    return {
        "scope": "native fixed-input host-boundary pilot, not recurrent/camera pipeline or comparative speedup",
        "percentile_method": "linear interpolation at (n-1)*fraction, per process",
        "memory_scope": "process RSS/HWM/PSS include fixtures and references; CUDA free/total are device-wide snapshots, not process peak GPU memory; do not add them",
        "replicates": runs,
        "median_range_ms": [
            min(run["median_ms"] for run in runs),
            max(run["median_ms"] for run in runs),
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.results)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "runs": len(result["replicates"]),
                "median_range_ms": result["median_range_ms"],
            }
        )
    )
