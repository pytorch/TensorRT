# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Validate and summarize the six checked fresh-process runs for each mode."""

import argparse
import csv
import json
import math
from pathlib import Path


def summarize(root):
    results = {}
    for mode in ("eager", "executorch"):
        records = []
        for frame in ("000", "001"):
            for repeat in range(3):
                path = root / f"{mode}-frame-{frame}-repeat-{repeat}"
                if not (path / "PASS").is_file():
                    raise ValueError(f"Missing PASS: {path}")
                report = json.loads((path / "report.json").read_text())
                checks = report["comparisons"]
                if (
                    len(checks) != 9
                    or {r["phase"] for r in checks}
                    != {"first", "after_warmup", "after_measurement"}
                    or any(
                        {r["output"] for r in checks if r["phase"] == phase}
                        != {0, 1, 2}
                        for phase in ("first", "after_warmup", "after_measurement")
                    )
                    or any(
                        r["mismatches"] or not math.isfinite(r["max_abs_error"])
                        for r in checks
                    )
                ):
                    raise ValueError(f"Incomplete/failing comparisons: {path}")
                if (
                    report["warmup"] != 20
                    or report["iterations"] != 100
                    or report["mode"] != mode
                ):
                    raise ValueError(f"Unexpected settings: {path}")
                with (path / "latency.tsv").open() as stream:
                    samples = list(csv.DictReader(stream, delimiter="\t"))
                if any(
                    sum(r["phase"] == phase for r in samples) != count
                    for phase, count in (
                        ("first", 1),
                        ("warmup", 20),
                        ("measured", 100),
                    )
                ):
                    raise ValueError(f"Incomplete latency samples: {path}")
                if any(
                    not math.isfinite(float(r["wall_ms"])) or float(r["wall_ms"]) <= 0
                    for r in samples
                ):
                    raise ValueError(f"Invalid latency sample: {path}")
                if "libtorch" not in (path / "loaded-maps.txt").read_text():
                    raise ValueError(f"Expected Torch-based Python control: {path}")
                final = report["memory"][-1]
                if final["phase"] != "after_measurement" or final["errors"]:
                    raise ValueError(f"Missing memory snapshot: {path}")
                records.append(
                    {
                        "path": path.name,
                        "frame": frame,
                        "repeat": repeat,
                        "median_ms": report["median_ms"],
                        "p95_ms": report["p95_ms"],
                        "first_ms": report["first_ms"],
                        "rss_mib": final["rss_kib"] / 1024,
                        "pss_mib": final["pss_kib"] / 1024,
                        "hwm_mib": final["process_hwm_kib"] / 1024,
                        "torch_allocated_mib": final["torch_allocated_bytes"] / 2**20,
                        "torch_reserved_mib": final["torch_reserved_bytes"] / 2**20,
                        "max_abs_error": max(c["max_abs_error"] for c in checks),
                        "versions": report["versions"],
                    }
                )
        results[mode] = {
            "runs": records,
            "ranges": {
                key: [min(r[key] for r in records), max(r[key] for r in records)]
                for key in (
                    "median_ms",
                    "p95_ms",
                    "rss_mib",
                    "pss_mib",
                    "hwm_mib",
                    "torch_allocated_mib",
                    "torch_reserved_mib",
                    "max_abs_error",
                )
            },
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(
        json.dumps({mode: value["ranges"] for mode, value in result.items()}, indent=2)
    )
