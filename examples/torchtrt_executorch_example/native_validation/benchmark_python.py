# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Inference-only BEVFormer/SegFormer memory control; never compile here.

Fixed CPU input snapshot -> completed inference -> owned CPU outputs.
The ExecuTorch mode deliberately uses torch tensor inputs and imports torch,
so it is a Python+Torch loader control, not a Torch-free Python measurement.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time

from process_memory import log_memory
from summarize_benchmark import percentile

INPUT_SHAPES = ((1, 6, 3, 480, 800), (1, 6, 4, 4), (1, 18), (1, 2500, 256), (1, 1, 1))


def model_spec(family):
    """Keep the historical BEVFormer protocol; add the validated SegFormer shape."""
    if family == "bevformer":
        return INPUT_SHAPES, 3, "model.py"
    if family == "segformer":
        return ((1, 3, 1024, 1024),), 1, "export_segformer.py"
    raise ValueError(f"Unknown model family: {family}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "executorch"), required=True)
    parser.add_argument(
        "--model-family", choices=("bevformer", "segformer"), default="bevformer"
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.warmup <= 100000 or not 1 <= args.iterations <= 100000:
        parser.error("warmup and iterations must be in 1..100000")
    if args.mode == "executorch" and not args.model:
        parser.error("executorch mode requires --model")
    if args.mode == "executorch" and args.model_family == "bevformer" and not args.data:
        parser.error("the mixed BEVFormer artifact requires --data")
    input_shapes, output_count, model_source = model_spec(args.model_family)
    args.output.mkdir(parents=True, exist_ok=False)
    phases = {}
    memory_rows = []
    phase_start = time.perf_counter()

    def memory(phase, cuda=False):
        row = log_memory(phase)
        if row["errors"]:
            raise RuntimeError(row["errors"])
        if cuda:
            free, total = torch.cuda.mem_get_info()
            row.update(
                cuda_free_bytes=free,
                cuda_total_bytes=total,
                torch_allocated_bytes=torch.cuda.memory_allocated(),
                torch_reserved_bytes=torch.cuda.memory_reserved(),
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        with (args.output / "memory.jsonl").open("a") as file:
            file.write(json.dumps(row, allow_nan=False) + "\n")
        memory_rows.append(row)
        return row

    memory("before_torch_import")
    import numpy as np
    import torch

    phases["torch_numpy_import"] = (time.perf_counter() - phase_start) * 1000
    memory("after_torch_import")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    phase_start = time.perf_counter()
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    phases["runtime_cuda_init"] = (time.perf_counter() - phase_start) * 1000
    memory("cuda_initialized", cuda=True)
    sys.path.insert(0, str(args.source_dir.resolve()))
    phase_start = time.perf_counter()
    with torch.inference_mode(), torch.cuda.stream(stream):
        if args.mode == "eager":
            if args.model_family == "segformer":
                from export_segformer import load_model

                model, _ = load_model(args.checkpoint)
            else:
                from model import load_model

                model = load_model(args.checkpoint)
            model = model.cuda().eval()
            loaded = None
        else:
            import torch_tensorrt_executorch_runtime  # noqa: F401
            from executorch.runtime import Runtime

            data_args = {"data_path": args.data} if args.data else {}
            program = Runtime.get().load_program(args.model, **data_args)
            loaded = program.load_method("forward")
            model = None
        torch.cuda.synchronize()
        phases["model_method_load"] = (time.perf_counter() - phase_start) * 1000
        memory("method_loaded", cuda=True)
        inputs = []
        for index, shape in enumerate(input_shapes):
            recorded_shape = tuple(
                map(int, (args.case / f"input_{index}.shape").read_text().split())
            )
            if recorded_shape != shape:
                raise ValueError(
                    f"Input {index}: expected {shape}, fixture has {recorded_shape}"
                )
            raw = np.fromfile(args.case / f"input_{index}.bin", dtype="<f4")
            if not np.isfinite(raw).all():
                raise ValueError("Nonfinite input")
            inputs.append(torch.from_numpy(raw.reshape(shape)))
        expected, actual = [], []
        for index in range(output_count):
            shape = tuple(
                map(int, (args.case / f"output_{index}.shape").read_text().split())
            )
            ref = np.fromfile(args.case / f"output_{index}.bin", dtype="<f4").reshape(
                shape
            )
            if not np.isfinite(ref).all():
                raise ValueError("Nonfinite reference")
            expected.append(ref)
            actual.append(torch.empty(shape, dtype=torch.float32))
        # Reuse eager input storage instead of allocating it on every call.
        device_inputs = (
            [torch.empty_like(x, device="cuda") for x in inputs]
            if model is not None
            else []
        )
        memory("fixtures_loaded", cuda=True)
        comparisons, timings = [], []

        def run():
            start = time.perf_counter()
            if model is not None:
                for dst, src in zip(device_inputs, inputs):
                    dst.copy_(src)
                outputs = model(*device_inputs)
            else:
                outputs = loaded.execute(tuple(inputs))
            torch.cuda.synchronize()
            if len(outputs) != len(actual):
                raise ValueError("Output count differs")
            for dst, src in zip(actual, outputs):
                if not isinstance(src, torch.Tensor):
                    src = torch.from_numpy(np.asarray(src))
                if src.dtype != torch.float32 or src.shape != dst.shape:
                    raise ValueError("Output shape/dtype differs")
                dst.copy_(src)
            return (time.perf_counter() - start) * 1000

        def check(phase):
            for index, (got, want) in enumerate(zip(actual, expected)):
                values = got.numpy()
                bad = int((~np.isclose(values, want, rtol=3e-4, atol=3e-4)).sum())
                error = float(np.max(np.abs(values.astype(np.float64) - want)))
                comparisons.append(
                    dict(
                        phase=phase,
                        output=index,
                        elements=values.size,
                        max_abs_error=error,
                        mismatches=bad,
                    )
                )
                if bad or not np.isfinite(values).all():
                    raise RuntimeError(f"{phase}: output {index} has {bad} mismatches")

        timings.append(dict(phase="first", iteration=0, wall_ms=run()))
        check("first")
        memory("first_inference", cuda=True)
        for index in range(args.warmup):
            timings.append(dict(phase="warmup", iteration=index, wall_ms=run()))
        check("after_warmup")
        memory("after_warmup", cuda=True)
        for index in range(args.iterations):
            timings.append(dict(phase="measured", iteration=index, wall_ms=run()))
        check("after_measurement")
        memory("after_measurement", cuda=True)
    for name, rows in (("latency.tsv", timings), ("comparisons.tsv", comparisons)):
        with (args.output / name).open("x") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "loaded-maps.txt").write_text(Path("/proc/self/maps").read_text())
    measured = [row["wall_ms"] for row in timings if row["phase"] == "measured"]
    report = {
        "mode": args.mode,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "model_family": args.model_family,
        "cuda_build": torch.version.cuda,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "scope": "FP32 fixed CPU inputs -> synchronized forward -> owned CPU outputs; no compilation",
        "settings": {
            "tf32": False,
            "inference_mode": True,
            "threads": 2,
            "cudnn_benchmark": False,
        },
        "fixture": str(args.case),
        "checkpoint": str(args.checkpoint),
        "model": str(args.model),
        "median_ms": statistics.median(measured),
        "p95_ms": percentile(measured, 0.95),
        "first_ms": timings[0]["wall_ms"],
        "phases_ms": phases,
        "memory": memory_rows,
        "comparisons": comparisons,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_source_sha256": hashlib.sha256(
            (args.source_dir / model_source).read_bytes()
        ).hexdigest(),
        "artifact_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in ([args.model] if args.model else [])
            + ([args.data] if args.data else [])
        },
        "fixture_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.case.glob("*.bin"))
            + sorted(args.case.glob("*.shape"))
        },
        "versions": {},
    }
    for name in (
        "torch",
        "tensorrt",
        "executorch",
        "torch-tensorrt-executorch-runtime",
    ):
        try:
            report["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][name] = None
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "PASS").write_text(
        f"All {output_count} outputs match at all three gates\n"
    )
    print("PASS", args.mode, "median_ms", report["median_ms"], flush=True)


if __name__ == "__main__":
    main()
