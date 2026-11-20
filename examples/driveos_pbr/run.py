# SPDX-License-Identifier: BSD-3-Clause
"""Verify artifacts, run fresh processes, and summarize the Thor U PBR."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


def verify(bundle):
    bundle = bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["format"] != 1 or not manifest["files"]:
        raise ValueError("Invalid artifact manifest")
    for name, expected in manifest["files"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or not path.is_file():
            raise ValueError(f"Invalid artifact path: {name}")
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    return manifest


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def native_files(directory):
    """Identify the rebuilt executables and libraries without changing the bundle."""
    files = {}
    for name in (
        "bin/benchmark_et",
        "bin/benchmark_trt",
        "lib/libaoti_cuda_shims.so",
        "lib/libextension_cuda.so",
    ):
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Missing rebuilt native file: {path}")
        with path.open("rb") as f:
            files[name] = hashlib.file_digest(f, "sha256").hexdigest()
    return files


def result(directory, model):
    if not (directory / "PASS").is_file():
        raise ValueError(f"No correctness PASS in {directory}")
    times = rows(directory / "latency.tsv")
    for phase, count in (("first", 1), ("warmup", 20), ("measured", 100)):
        selected = [r for r in times if r["phase"] == phase]
        if [int(r["iteration"]) for r in selected] != list(range(count)):
            raise ValueError(f"Incomplete {phase} calls in {directory}")
        if any(
            not math.isfinite(float(r["wall_ms"])) or float(r["wall_ms"]) <= 0
            for r in selected
        ):
            raise ValueError("Invalid latency")
    checks = rows(directory / "comparisons.tsv")
    no = 3 if model == "bevformer" else 1
    expected_checks = {
        (p, i)
        for p in ("first", "after_warmup", "after_measurement")
        for i in range(no)
    }
    if (
        len(checks) != len(expected_checks)
        or {(r["phase"], int(r["output"])) for r in checks} != expected_checks
    ):
        raise ValueError("Missing output checks")
    if any(int(r["mismatches"]) != 0 or int(r["elements"]) <= 0 for r in checks):
        raise ValueError("Output mismatch")
    memory = {r["phase"]: r for r in rows(directory / "memory.tsv")}
    start, load, end = (
        memory[p] for p in ("cuda_initialized", "method_loaded", "after_measurement")
    )
    return {
        "latency_ms": statistics.median(
            float(r["wall_ms"]) for r in times if r["phase"] == "measured"
        ),
        "loading_peak_rss_mib": int(load["process_hwm_kib"]) / 1024,
        "retained_rss_mib": int(end["rss_kib"]) / 1024,
        "retained_pss_mib": int(end["pss_kib"]) / 1024,
        "lifetime_peak_rss_mib": int(end["process_hwm_kib"]) / 1024,
        "cuda_free_decrease_mib": (
            int(start["cuda_free_bytes"]) - int(end["cuda_free_bytes"])
        )
        / 1048576,
    }


def summarize(output, models, modes):
    summary = []
    for model, mode in itertools.product(models, modes):
        runs = [
            result(output / f"{model}-{mode}-case{case}-launch{launch}", model)
            for case in range(2)
            for launch in range(3)
        ]
        for metric in runs[0]:
            values = [r[metric] for r in runs]
            summary.append(
                dict(
                    model=model,
                    mode=mode,
                    metric=metric,
                    processes=6,
                    mean=statistics.mean(values),
                    minimum=min(values),
                    maximum=max(values),
                )
            )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--native-dir",
        type=Path,
        help="Installed source build with bin/ and lib/; overrides bundled native runtime",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("bevformer", "segformer"),
        default=["bevformer", "segformer"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("eager", "onnx", "executorch"),
        default=["eager", "onnx", "executorch"],
    )
    parser.add_argument(
        "--container", help="Configured authoring container for eager only"
    )
    parser.add_argument(
        "--container-root",
        type=Path,
        help="Mount of the handoff directory inside the container",
    )
    parser.add_argument(
        "--python",
        default="/opt/pbr-venv/bin/python",
        help="Python interpreter inside the container",
    )
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    verify(bundle)
    if args.verify_only:
        print("Artifact checksums passed")
        return
    if not args.output:
        parser.error("--output is required for a benchmark")
    root = bundle.parent
    native = args.native_dir.resolve() if args.native_dir else bundle
    rebuilt = native_files(native) if args.native_dir else None
    output = args.output.resolve()
    source = Path(__file__).resolve().parent
    if "eager" in args.modes and (not args.container or not args.container_root):
        parser.error("Eager needs --container and --container-root")
    if "eager" in args.modes and not all(
        p.is_relative_to(root) for p in (source, output)
    ):
        parser.error(
            "Source and output must be under the handoff directory mounted in the container"
        )
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = (
        str(native / "lib") + ":" + environment.get("LD_LIBRARY_PATH", "")
    )
    environment["OMP_NUM_THREADS"] = "2"
    environment["MKL_NUM_THREADS"] = "2"
    (output / "host.txt").write_text(
        subprocess.check_output(["uname", "-a"], text=True)
    )
    for name in ("nv_tegra_release", "os-release"):
        path = Path("/etc") / name
        if path.exists():
            (output / name).write_text(path.read_text())
    (output / "configuration.json").write_text(
        json.dumps(
            {
                "models": args.models,
                "modes": args.modes,
                "warmup": 20,
                "measured": 100,
                "native_directory": str(native),
                "rebuilt_native_sha256": rebuilt,
                "native_source_revisions": (
                    (native / "source-revisions.txt").read_text()
                    if args.native_dir and (native / "source-revisions.txt").is_file()
                    else None
                ),
                "artifact_manifest_sha256": hashlib.sha256(
                    (bundle / "manifest.json").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
    orders = list(itertools.permutations(args.modes))
    for model in args.models:
        for trial in range(6):
            case, launch = trial % 2, trial // 2
            for mode in orders[trial % len(orders)]:
                name = f"{model}-{mode}-case{case}-launch{launch}"
                destination = output / name
                arguments = [
                    bundle / model,
                    bundle / model / f"case{case}",
                    destination,
                ]
                if mode == "eager":

                    def inside(path):
                        return str(args.container_root / path.relative_to(root))

                    command = [
                        "docker",
                        "exec",
                        "-e",
                        "OMP_NUM_THREADS=2",
                        "-e",
                        "MKL_NUM_THREADS=2",
                        "-e",
                        f"LD_LIBRARY_PATH={inside(bundle / 'authoring/lib')}",
                        args.container,
                        args.python,
                        inside(source / "eager.py"),
                        *map(inside, arguments),
                        model,
                    ]
                else:
                    command = [
                        str(
                            native
                            / "bin"
                            / ("benchmark_trt" if mode == "onnx" else "benchmark_et")
                        ),
                        *map(str, arguments),
                        model,
                    ]
                print(name, flush=True)
                (output / f"{name}.command.json").write_text(json.dumps(command))
                with (output / f"{name}.log").open("w") as log:
                    subprocess.run(
                        command,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                result(destination, model)
    summarize(output, args.models, args.modes)
    print(f"All runs passed. Results: {output / 'summary.csv'}")


if __name__ == "__main__":
    main()
