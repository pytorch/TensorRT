# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect instructions and replay saved region-generation tasks."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch_tensorrt.region import KernelGenerator


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _run_directory(path: Path) -> Path:
    directory = path.resolve() if path.is_dir() else path.resolve().parent
    for parent in (directory, *directory.parents):
        if (parent / "task.json").is_file() and (parent / "reference.pt").is_file():
            return parent
    if path.is_file():
        manifest = _read(path)
        if "run" in manifest:
            return _run_directory(Path(manifest["run"]))
        state = manifest.get("state")
        if isinstance(state, dict) and isinstance(state.get("run"), str):
            return _run_directory(Path(state["run"]))
    raise ValueError(
        "PATH must identify a saved generation run, task, attempt, or accepted manifest"
    )


def _evaluate(args: argparse.Namespace, run: Path) -> dict[str, Any]:
    from filelock import FileLock

    from torch_tensorrt.region import KernelGenerator

    from ._factory import _evaluate as evaluate
    from ._factory import _meets_objective, _write_json

    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    deadline = time.monotonic() + args.timeout
    task = _read(run / "task.json")
    proposal = _read(args.candidate)
    candidate = proposal.get("candidate", proposal)
    request = {
        "reference_path": str(run / "reference.pt"),
        "backend": task["backend"],
        "inputs": task["inputs"],
        "outputs": task["outputs"],
        "objective": task.get("objective", "candidate"),
        "validation_phase": "holdout",
        "diagnostics": task.get("diagnostics", False),
        **task["validation"],
        "candidate": candidate,
    }
    for name in ("backend_version", "proposal_schema_sha256"):
        if name in task:
            request[name] = task[name]
    directory = Path(tempfile.mkdtemp(prefix="evaluation-", dir=run))
    device = task["inputs"][0]["device"].split(":")[-1]
    with FileLock(str(run.parent.parent / f"gpu-{device}.lock"), timeout=args.timeout):
        report = evaluate(request, directory, max(0, deadline - time.monotonic()))
    report["meets_objective"] = _meets_objective(
        report,
        KernelGenerator(
            objective=task.get("objective", "candidate"),
            min_speedup=task.get("min_speedup", 1.0),
        ),
    )
    _write_json(directory / "feedback.json", report)
    return {"directory": str(directory), "report": report}


def _tuning(value: str) -> dict[str, tuple[Any, ...]]:
    try:
        domains = json.loads(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("tuning must be a JSON object") from error
    if not isinstance(domains, dict) or any(
        not isinstance(domain, list) for domain in domains.values()
    ):
        raise argparse.ArgumentTypeError("tuning must map knob names to JSON arrays")
    return {name: tuple(domain) for name, domain in domains.items()}


def _generation_config(
    args: argparse.Namespace, task: dict[str, Any], run: Path
) -> KernelGenerator:
    from torch_tensorrt.region import KernelGenerator, SearchPolicy

    guidance = args.instructions
    if (
        guidance is None
        and not args.instruction_file
        and (run / "instructions.json").is_file()
    ):
        bundle = _read(run / "instructions.json")
        guidance = (
            "\n\n".join(
                entry["text"]
                for entry in bundle["resources"]
                if entry["origin"] in ("file", "inline")
            )
            or None
        )
    saved_search = task.get("search", {})
    if not isinstance(saved_search, dict):
        raise ValueError("Saved task search policy must be a JSON object")
    search = dict(saved_search)
    for name in (
        "batch_size",
        "max_candidates",
        "max_no_improvement",
        "max_variants",
        "finalists",
        "resume",
        "tuning",
    ):
        value = getattr(args, name, None)
        if value is not None:
            search[name] = value
    diagnostics = getattr(args, "diagnostics", None)
    policy = SearchPolicy(**search)
    cache_dir = args.cache_dir
    if cache_dir is None and policy.resume:
        cache_dir = run.parent.parent
    return KernelGenerator(
        provider=args.provider,
        backend=args.backend or task["backend"],
        model=args.model,
        instructions=guidance,
        instruction_files=tuple(args.instruction_file),
        max_attempts=args.attempts,
        timeout_s=args.timeout,
        cache_dir=cache_dir,
        objective=args.objective or task.get("objective", "candidate"),
        min_speedup=(
            args.min_speedup
            if args.min_speedup is not None
            else task.get("min_speedup", 1.0)
        ),
        search=policy,
        diagnostics=(
            diagnostics if diagnostics is not None else task.get("diagnostics", False)
        ),
        **task["validation"],
    )


def _generate(args: argparse.Namespace, run: Path) -> dict[str, Any]:
    import torch

    from ._factory import _cache_root
    from ._factory import _generate as generate
    from ._worker import _dtype, _outputs, _restore_metadata

    task = _read(run / "task.json")
    instruction_bundle = _saved_instruction_bundle(args, task, run)
    config = _generation_config(args, task, run)
    device = torch.device(task["inputs"][0]["device"])
    with torch.cuda.device(device), torch.inference_mode():
        reference = torch.load(
            run / "reference.pt", map_location=device, weights_only=False
        )
        if not isinstance(reference, torch.fx.GraphModule):
            raise ValueError("Saved reference must be a GraphModule")
        inputs = tuple(
            torch.randn(spec["shape"], dtype=_dtype(spec["dtype"]), device=device)
            for spec in task["inputs"]
        )
        _restore_metadata(reference, inputs, _outputs(reference(*inputs)))
        _, directory = generate(
            reference,
            config,
            region_id="interactive",
            instruction_bundle=instruction_bundle,
        )
    accepted = directory / "accepted.json"
    report = (
        _read(accepted)["validation"]
        if accepted.is_file()
        else _read(directory / "cache_check.json")
    )
    return {
        "status": "passed",
        "backend": config.backend,
        "provider": config.provider,
        "cache_dir": str(_cache_root(config)),
        "directory": str(directory),
        "report": report,
    }


def _saved_instruction_bundle(
    args: argparse.Namespace, task: dict[str, Any], run: Path
) -> dict[str, Any] | None:
    from ._instructions import validate_instructions

    if (
        args.instructions is not None
        or args.instruction_file
        or (args.backend is not None and args.backend != task["backend"])
    ):
        return None
    path = run / "instructions.json"
    if not path.is_file():
        if "instructions_sha256" in task:
            raise ValueError(
                "Saved instructions.json is missing; exact replay requires the "
                "instruction bundle recorded by the task"
            )
        return None
    bundle = validate_instructions(_read(path), task["backend"])
    if task.get("instructions_sha256") != bundle["sha256"]:
        raise ValueError("Saved instructions do not match the task instruction digest")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    instructions = commands.add_parser(
        "instructions", help="Print packaged author guidance without a model call"
    )
    instructions.add_argument("--backend", default="triton")
    inspect = commands.add_parser("inspect", help="Print a saved task and its results")
    inspect.add_argument("path", type=Path)
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate a proposal using fresh inputs and the real worker"
    )
    evaluate.add_argument("path", type=Path)
    evaluate.add_argument("--candidate", required=True, type=Path)
    evaluate.add_argument("--timeout", default=120.0, type=float)
    generate = commands.add_parser(
        "generate", help="Run the same generation controller on a saved reference"
    )
    generate.add_argument("path", type=Path)
    generate.add_argument("--provider", default="claude")
    generate.add_argument("--backend")
    generate.add_argument("--model")
    generate.add_argument("--instructions")
    generate.add_argument("--instruction-file", action="append", default=[], type=Path)
    generate.add_argument("--attempts", default=4, type=int)
    generate.add_argument("--timeout", default=600.0, type=float)
    generate.add_argument("--cache-dir", type=Path)
    generate.add_argument("--objective", choices=("candidate", "eager", "tensorrt"))
    generate.add_argument("--min-speedup", type=float)
    generate.add_argument("--batch-size", type=int, help="Proposals per author round")
    generate.add_argument(
        "--max-candidates",
        type=int,
        help="Search evaluations, including tuning and resume",
    )
    generate.add_argument(
        "--max-no-improvement",
        type=int,
        help="Stop after this many author rounds without improvement",
    )
    generate.add_argument(
        "--max-variants",
        type=int,
        help="Maximum tuning variants per authored candidate",
    )
    generate.add_argument(
        "--finalists", type=int, help="Maximum candidates to try in final validation"
    )
    generate.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Restore compatible search evidence and remeasure retained candidates",
    )
    generate.add_argument(
        "--diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Collect compiler and execution facts from the evaluator",
    )
    generate.add_argument(
        "--tuning", type=_tuning, help='JSON knob domains, e.g. {"num_warps":[4,8]}'
    )
    args = parser.parse_args()
    try:
        if args.command == "instructions":
            from ._backends import get_backend
            from ._instructions import load_instructions

            backend = get_backend(args.backend)
            print(
                load_instructions(args.backend, backend.instruction_resources)["text"]
            )
            return
        run = _run_directory(args.path)
        if args.command == "inspect":
            names = (
                "task.json",
                "instructions.json",
                "accepted.json",
                "cache_check.json",
                "history.json",
                "search.json",
            )
            result = {
                "directory": str(run),
                **{
                    name: json.loads((run / name).read_text(encoding="utf-8"))
                    for name in names
                    if (run / name).is_file()
                },
            }
            checkpoint = run.parent / "checkpoint.json"
            if checkpoint.is_file():
                result["checkpoint.json"] = _read(checkpoint)
        elif args.command == "evaluate":
            result = _evaluate(args, run)
        else:
            result = _generate(args, run)
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.command == "evaluate" and not result["report"].get(
            "meets_objective", False
        ):
            raise SystemExit(1)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
