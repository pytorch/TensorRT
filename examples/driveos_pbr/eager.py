# SPDX-License-Identifier: BSD-3-Clause
"""Inference-only PyTorch eager baseline, with the same saved fixtures as C++."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("model", choices=("bevformer", "segformer"))
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    import numpy as np
    import torch

    if torch.__version__ != "2.16.0.dev20261009+cu132":
        raise RuntimeError(
            f"Expected pinned PyTorch 2.16.0.dev20261009+cu132, got {torch.__version__}"
        )
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    memory, checks, times = [], [], []

    def snapshot(phase):
        status = dict(
            line.split(":", 1)
            for line in Path("/proc/self/status").read_text().splitlines()
        )
        smaps = dict(
            line.split(":", 1)
            for line in Path("/proc/self/smaps_rollup").read_text().splitlines()
            if ":" in line
        )
        free, total = torch.cuda.mem_get_info()
        memory.append(
            dict(
                phase=phase,
                rss_kib=int(status["VmRSS"].split()[0]),
                process_hwm_kib=int(status["VmHWM"].split()[0]),
                pss_kib=int(smaps["Pss"].split()[0]),
                cuda_free_bytes=free,
                cuda_total_bytes=total,
            )
        )

    def fixture(stem):
        shape = tuple(map(int, (args.fixture / (stem + ".shape")).read_text().split()))
        if not shape or any(x <= 0 for x in shape):
            raise ValueError("Invalid fixture shape")
        data = np.fromfile(args.fixture / (stem + ".bin"), dtype="<f4").reshape(shape)
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite fixture")
        return torch.from_numpy(data)

    snapshot("cuda_initialized")
    source = Path(__file__).resolve().parent / "models"
    sys.path.insert(
        0, str(source / "bevformer" if args.model == "bevformer" else source)
    )
    with torch.inference_mode(), torch.cuda.stream(stream):
        if args.model == "bevformer":
            from model import load_model

            model = load_model(args.model_dir / "checkpoint.pth")
        else:
            from segformer import load_model

            model = load_model(args.model_dir / "checkpoint")
        model = model.cuda().eval()
        torch.cuda.synchronize()
        snapshot("method_loaded")
        ni, no = (5, 3) if args.model == "bevformer" else (1, 1)
        inputs = [fixture(f"input_{i}") for i in range(ni)]
        expected = [fixture(f"output_{i}") for i in range(no)]
        device_inputs = [torch.empty_like(x, device="cuda") for x in inputs]
        actual = [torch.empty_like(x) for x in expected]
        snapshot("fixtures_loaded")

        def run():
            start = time.perf_counter()
            for dst, src in zip(device_inputs, inputs):
                dst.copy_(src)
            outputs = model(*device_inputs)
            torch.cuda.synchronize()
            if len(outputs) != no:
                raise RuntimeError("Output count mismatch")
            for dst, src in zip(actual, outputs):
                if src.shape != dst.shape or src.dtype != torch.float32:
                    raise RuntimeError("Output shape/type mismatch")
                dst.copy_(src)
            return (time.perf_counter() - start) * 1000

        def check(phase):
            for i, (got, want) in enumerate(zip(actual, expected)):
                bad = int((~torch.isclose(got, want, atol=3e-4, rtol=3e-4)).sum())
                checks.append(
                    dict(
                        phase=phase,
                        output=i,
                        elements=got.numel(),
                        max_abs_error=float((got.double() - want.double()).abs().max()),
                        mismatches=bad,
                    )
                )
                if bad or not torch.isfinite(got).all():
                    raise RuntimeError(f"Output {i} failed; performance result invalid")

        times.append(dict(phase="first", iteration=0, wall_ms=run()))
        check("first")
        snapshot("first_inference")
        for phase, count, end in (
            ("warmup", 20, "after_warmup"),
            ("measured", 100, "after_measurement"),
        ):
            for i in range(count):
                times.append(dict(phase=phase, iteration=i, wall_ms=run()))
            check(end)
            snapshot(end)
    for name, rows in (("memory", memory), ("comparisons", checks), ("latency", times)):
        with (args.output / f"{name}.tsv").open("x") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0], delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "loaded-maps.txt").write_text(Path("/proc/self/maps").read_text())
    (args.output / "environment.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda_build": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "python": sys.version,
            },
            indent=2,
        )
    )
    (args.output / "PASS").write_text("All outputs passed at atol=rtol=3e-4\n")


if __name__ == "__main__":
    main()
