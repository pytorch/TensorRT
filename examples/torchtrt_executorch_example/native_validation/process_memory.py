# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Linux process-memory snapshots, without importing torch or CUDA.

rss_kib and process_hwm_kib match benchmark.cpp's /proc/PID/status fields.
pss_kib and smaps_rss_kib come from /proc/PID/smaps_rollup.
Unavailable measurements are None with an explicit error, never zero.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time


def parse_kib(text, key):
    """Read one exact /proc field; Linux's 'kB' here means 1024 bytes."""
    for line in text.splitlines():
        name, separator, value = line.partition(":")
        if separator and name == key:
            fields = value.split()
            if len(fields) != 2 or fields[1] != "kB":
                raise ValueError(f"Invalid unit/value for {key}: {value.strip()}")
            amount = int(fields[0])
            if amount < 0:
                raise ValueError(f"Negative value for {key}")
            return amount
    raise ValueError(f"Missing field {key}")


def snapshot(phase, *, pid=None, proc_root=Path("/proc")):
    """Read one process, not its children. No synchronization or cache clearing."""
    pid = os.getpid() if pid is None else pid
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("pid must be a positive integer")
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError("phase must be a nonempty string")

    result = {
        "phase": phase,
        "pid": pid,
        "timestamp_ns": time.time_ns(),
        "rss_kib": None,
        "process_hwm_kib": None,
        "pss_kib": None,
        "smaps_rss_kib": None,
        "errors": [],
    }
    readings = {
        "status": {"VmRSS": "rss_kib", "VmHWM": "process_hwm_kib"},
        "smaps_rollup": {"Pss": "pss_kib", "Rss": "smaps_rss_kib"},
    }
    for filename, fields in readings.items():
        path = Path(proc_root) / str(pid) / filename
        try:
            text = path.read_text()
        except OSError as error:
            result["errors"].append(f"{path}: {error}")
            continue
        for key, output_key in fields.items():
            try:
                result[output_key] = parse_kib(text, key)
            except ValueError as error:
                result["errors"].append(f"{path}: {error}")
    return result


def log_memory(phase, output=None, *, pid=None):
    """Print a snapshot and optionally append it to a single-writer JSONL file.

    Call outside latency measurements. The caller must synchronize CUDA before
    a post-inference snapshot if completion of GPU work is required.
    """
    result = snapshot(phase, pid=pid)
    if output is not None:
        with Path(output).open("a", encoding="utf-8") as file:
            file.write(json.dumps(result) + "\n")

    def mib(key):
        value = result[key]
        return "unavailable" if value is None else f"{value / 1024:.2f} MiB"

    print(
        f"{phase} (PID {result['pid']}): RSS={mib('rss_kib')}, "
        f"PSS={mib('pss_kib')}, lifetime peak RSS={mib('process_hwm_kib')}",
        flush=True,
    )
    for error in result["errors"]:
        print(f"Memory measurement unavailable: {error}", file=sys.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Required: accidentally measuring this short-lived helper would not
    # measure the user's model process.
    parser.add_argument(
        "--pid", required=True, type=int, help="PID of the live process to inspect"
    )
    parser.add_argument("--phase", default="snapshot")
    parser.add_argument(
        "--output", type=Path, help="Append to this JSONL file; parent must exist"
    )
    args = parser.parse_args()
    if args.pid <= 0:
        parser.error("--pid must be positive")
    if not args.phase.strip():
        parser.error("--phase must not be empty")
    result = log_memory(args.phase, args.output, pid=args.pid)
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
