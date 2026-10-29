# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the SegFormer matrix before producing publication-candidate numbers.

Only process memory is compared. CUDA device-wide snapshots are not summed
with RSS/PSS and are not labelled as process GPU usage.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native_validation"))
from summarize_benchmark import percentile, summarize_run


def checked_python_run(path, mode):
    if not (path / "PASS").is_file():
        raise ValueError(f"Missing PASS: {path}")
    report = json.loads((path / "report.json").read_text())
    if (
        report["mode"],
        report["model_family"],
        report["warmup"],
        report["iterations"],
    ) != (mode, "segformer", 20, 100):
        raise ValueError("Unexpected benchmark profile/settings")
    checks = report["comparisons"]
    if len(checks) != 3 or {r["phase"] for r in checks} != {
        "first",
        "after_warmup",
        "after_measurement",
    }:
        raise ValueError("Missing correctness gate")
    if any(
        r["output"] != 0
        or r["elements"] != 1245184
        or r["mismatches"] != 0
        or not math.isfinite(r["max_abs_error"])
        for r in checks
    ):
        raise ValueError("Invalid/failing all-logit comparisons")
    with (path / "latency.tsv").open() as file:
        timing = list(csv.DictReader(file, delimiter="\t"))
    if len(timing) != 121 or any(
        sum(r["phase"] == phase for r in timing) != count
        for phase, count in (("first", 1), ("warmup", 20), ("measured", 100))
    ):
        raise ValueError("Incomplete timing samples")
    if any(
        not math.isfinite(float(r["wall_ms"])) or float(r["wall_ms"]) <= 0
        for r in timing
    ):
        raise ValueError("Invalid timing samples")
    measured = [float(r["wall_ms"]) for r in timing if r["phase"] == "measured"]
    if not math.isclose(statistics.median(measured), report["median_ms"], abs_tol=1e-9):
        raise ValueError("Summary median differs from raw samples")
    if "libtorch" not in (path / "loaded-maps.txt").read_text():
        raise ValueError("Expected Torch-based Python control")
    memory = report["memory"][-1]
    if memory["phase"] != "after_measurement" or memory["errors"]:
        raise ValueError("Incomplete memory snapshot")
    if any(
        not isinstance(memory[k], int) or memory[k] <= 0
        for k in ("rss_kib", "pss_kib", "process_hwm_kib")
    ):
        raise ValueError("Missing/invalid process memory")
    return {
        "median_ms": statistics.median(measured),
        "p95_ms": percentile(measured, 0.95),
        "first_ms": float(timing[0]["wall_ms"]),
        "rss_mib": memory["rss_kib"] / 1024,
        "pss_mib": memory["pss_kib"] / 1024,
        "hwm_mib": memory["process_hwm_kib"] / 1024,
        "max_abs_error": max(r["max_abs_error"] for r in checks),
        "versions": report["versions"],
        "fixture_sha256": report["fixture_sha256"],
        "artifact_sha256": report["artifact_sha256"],
    }


def summarize(root):
    digests = {}
    manifest_paths = {}
    for line in (root / "sha256.txt").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        name = Path(path).name
        if name in digests:
            raise ValueError("Ambiguous checksum manifest")
        digests[name] = digest
        manifest_paths[name] = path
    result = {}
    for mode in ("eager", "executorch", "native"):
        paths = sorted(p for p in (root / mode).glob("repeat-*") if p.is_dir())
        if [p.name for p in paths] != [f"repeat-{i}" for i in range(3)]:
            raise ValueError("Expected exactly three independent processes")
        runs = []
        for path in paths:
            if mode != "native":
                run = checked_python_run(path, mode)
                if {Path(k).name: v for k, v in run["artifact_sha256"].items()} != {
                    "model.pte": digests["model.pte"]
                }:
                    raise ValueError("Different model artifact")
                fixtures = {
                    k: v for k, v in digests.items() if k.endswith((".bin", ".shape"))
                }
                if run["fixture_sha256"] != fixtures:
                    raise ValueError("Different fixture")
            else:
                raw = summarize_run(path)
                if raw["metadata"]["model"] != manifest_paths["model.pte"] or raw[
                    "metadata"
                ]["fixture"] != str(Path(manifest_paths["input_0.bin"]).parent):
                    raise ValueError(
                        "Native model/fixture differs from checksum manifest"
                    )
                if (
                    raw["compared_elements_per_gate"] != 1245184
                    or raw["metadata"]["warmup"] != "20"
                    or raw["metadata"]["iterations"] != "100"
                ):
                    raise ValueError("Native protocol differs")
                memory = raw["memory_snapshots"][-1]
                if memory["phase"] != "after_measurement" or any(
                    memory[k] <= 0 for k in ("rss_kib", "pss_kib", "process_hwm_kib")
                ):
                    raise ValueError("Incomplete native memory snapshot")
                run = dict(
                    median_ms=raw["median_ms"],
                    p95_ms=raw["p95_ms"],
                    first_ms=raw["first_inference_ms"],
                    max_abs_error=raw["max_abs_error"],
                    rss_mib=memory["rss_kib"] / 1024,
                    pss_mib=memory["pss_kib"] / 1024,
                    hwm_mib=memory["process_hwm_kib"] / 1024,
                )
            run["run"] = path.name
            runs.append(run)
        keys = (
            "median_ms",
            "p95_ms",
            "first_ms",
            "rss_mib",
            "pss_mib",
            "hwm_mib",
            "max_abs_error",
        )
        result[mode] = {
            "runs": runs,
            "ranges": {
                k: [min(r[k] for r in runs), max(r[k] for r in runs)] for k in keys
            },
        }
    return {
        "model_sha256": digests["model.pte"],
        "scope": "fixed real-image snapshot, CPU input to owned CPU logits; three fresh processes per mode; Python in container, native on host",
        "modes": result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({k: v["ranges"] for k, v in result["modes"].items()}, indent=2))
