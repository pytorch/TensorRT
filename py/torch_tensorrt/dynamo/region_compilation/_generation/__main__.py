# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect instructions and replay saved region-generation tasks."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import sys
import tempfile
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from ._reference import FILENAME as REFERENCE_FILENAME
from ._reference import load_reference
from ._samples import FILENAME as SAMPLES_FILENAME
from ._util import gpu_lock, read_json_object, write_json

if TYPE_CHECKING:
    from torch_tensorrt.region import KernelGenerator

_MANIFESTS = ("accepted.json", "checkpoint.json")
_INSPECTED = (
    "task.json",
    "instructions.json",
    "accepted.json",
    "cache_check.json",
    "history.json",
    "search.json",
)
_SEARCH_OPTIONS = (
    "batch_size",
    "max_candidates",
    "max_no_improvement",
    "max_variants",
    "finalists",
    "resume",
    "tuning",
    "screening",
)
_PATH_HELP = (
    "A saved run, a path inside one, or a task cache directory (the reported "
    "'Artifacts:' path) or one of its manifests"
)


def _is_run(directory: Path) -> bool:
    return all(
        (directory / name).is_file() for name in ("task.json", REFERENCE_FILENAME)
    )


def _manifest_run(manifest: Path) -> Path | None:
    """Return the run named by a task cache's accepted.json or checkpoint.json."""
    if manifest.name not in _MANIFESTS or not manifest.is_file():
        return None
    value = read_json_object(manifest)
    state = value.get("state")
    run = value.get("run", state.get("run") if isinstance(state, dict) else None)
    # Runs are children of the task cache directory, which may have moved.
    return manifest.parent / Path(run).name if isinstance(run, str) else None


def _run_directory(path: Path) -> Path:
    """Resolve a run, a path inside one, or a task cache directory."""
    path = path.resolve()
    directory = path if path.is_dir() else path.parent
    for parent in (directory, *directory.parents):
        if _is_run(parent):
            return parent
    # A task cache directory holds runs and the manifests that name them; an
    # explicitly named manifest takes precedence.
    for manifest in (path, *(directory / name for name in _MANIFESTS)):
        run = _manifest_run(manifest)
        if run is not None and _is_run(run):
            return run
    runs = [run for run in directory.glob("run-*") if _is_run(run)]
    if not runs:
        raise ValueError(
            "PATH must identify a saved generation run, a path inside one, or a "
            "task cache directory"
        )
    # Without a task-level manifest (use_cache=False), a run that accepted a
    # kernel keeps its samples, unlike a later run that only reused it.
    accepted = [run for run in runs if (run / "accepted.json").is_file()]
    return max(accepted or runs, key=lambda run: run.stat().st_mtime)


def _read_task(run: Path) -> dict[str, Any]:
    from ._factory import _VERSION

    task = read_json_object(run / "task.json")
    if task.get("version") != _VERSION:
        raise ValueError(
            f"{run} records a version {task.get('version')!r} task, but this release "
            f"replays version {_VERSION}; compile the region again to record a new run"
        )
    return task


def _inspect(run: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"directory": str(run)}
    for name in _INSPECTED:
        if (run / name).is_file():
            result[name] = json.loads((run / name).read_text(encoding="utf-8"))
    # The task cache keeps one checkpoint, describing its most recent search.
    checkpoint = run.parent / "checkpoint.json"
    if _manifest_run(checkpoint) == run:
        result["checkpoint.json"] = read_json_object(checkpoint)
    return result


def _evaluate(args: argparse.Namespace, run: Path) -> dict[str, Any]:
    from filelock import Timeout

    from torch_tensorrt.region import KernelGenerator

    from ._factory import _evaluate as evaluate
    from ._factory import _meets_objective, _useful

    task = _read_task(run)
    timeout = (
        task["budget"]["candidate_timeout_s"] if args.timeout is None else args.timeout
    )
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    deadline = time.monotonic() + timeout
    proposal = read_json_object(args.candidate)
    request = {
        **read_json_object(run / "evaluation-template.json"),
        # The run may have moved since the template was written.
        "reference_path": str(run / REFERENCE_FILENAME),
        "validation_seed": secrets.randbits(63),
        "validation_phase": "holdout",
        "candidate": proposal.get("candidate", proposal),
        # As in replay, only the samples that the task records are checked.
        "samples_path": (
            None if _saved_samples(task, run) is None else str(run / SAMPLES_FILENAME)
        ),
    }
    directory = Path(tempfile.mkdtemp(prefix="evaluation-", dir=run))
    report: dict[str, Any] = {
        "status": "error",
        "stage": "budget",
        "error": f"No time remained of the {timeout:g}s budget after waiting for the GPU",
    }
    with (
        suppress(Timeout),
        gpu_lock(run.parent.parent, request["inputs"][0]["device"]).acquire(
            timeout=timeout
        ),
    ):
        remaining = deadline - time.monotonic()
        if _useful(remaining, timeout):
            report = evaluate(request, directory, remaining)
            if report.get("stage") == "timeout" and remaining < timeout:
                # As in the controller: the GPU wait, not the candidate, cut
                # this evaluation short.
                report = {
                    "status": "error",
                    "stage": "budget",
                    "error": (
                        f"Waiting for the GPU left {remaining:.3g}s of the "
                        f"{timeout:g}s budget, and the evaluation did not finish"
                    ),
                }
    report["meets_objective"] = _meets_objective(
        report,
        KernelGenerator(objective=task["objective"], min_speedup=task["min_speedup"]),
    )
    write_json(directory / "feedback.json", report)
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


def _given(args: argparse.Namespace, names: Iterable[str]) -> dict[str, Any]:
    """Return the options set on the command line; the task supplies the rest."""
    return {
        name: getattr(args, name)
        for name in names
        if getattr(args, name, None) is not None
    }


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
        bundle = read_json_object(run / "instructions.json")
        guidance = (
            "\n\n".join(
                entry["text"]
                for entry in bundle["resources"]
                if entry["origin"] in ("file", "inline")
            )
            or None
        )
    saved = {
        "backend": task["backend"],
        "objective": task["objective"],
        "min_speedup": task["min_speedup"],
        "diagnostics": task["diagnostics"],
        **task["budget"],
        **task["validation"],
    }
    try:
        search = SearchPolicy(**{**task["search"], **_given(args, _SEARCH_OPTIONS)})
        # A run lives in <cache root>/<task key>/, so replay reuses its kernels
        # and checkpoint unless --cache-dir points elsewhere.
        cache_dir = run.parent.parent if args.cache_dir is None else args.cache_dir
        return KernelGenerator(
            provider=args.provider,
            model=args.model,
            instructions=guidance,
            instruction_files=tuple(args.instruction_file),
            cache_dir=cache_dir,
            search=search,
            **{**saved, **_given(args, saved)},
        )
    except TypeError as error:
        raise ValueError(f"Invalid generation settings for {run}: {error}") from None


def _saved_instruction_bundle(
    args: argparse.Namespace, task: dict[str, Any], run: Path
) -> dict[str, Any] | None:
    """Return the run's exact guidance unless the command line replaces it."""
    from ._instructions import validate_instructions

    if (
        args.instructions is not None
        or args.instruction_file
        or args.backend not in (None, task["backend"])
    ):
        return None
    path = run / "instructions.json"
    if not path.is_file():
        raise ValueError(
            "Saved instructions.json is missing; exact replay requires the "
            "instruction bundle recorded by the task"
        )
    bundle = validate_instructions(read_json_object(path), task["backend"])
    if bundle["sha256"] != task["instructions_sha256"]:
        raise ValueError("Saved instructions do not match the task instruction digest")
    return bundle


def _saved_samples(task: dict[str, Any], run: Path) -> list[Any] | None:
    """Return the run's validation samples once they match the task's record."""
    from ._samples import load_samples, samples_digest

    if task.get("samples") is None:
        return None
    path = run / SAMPLES_FILENAME
    if not path.is_file():
        raise ValueError(
            f"Saved {SAMPLES_FILENAME} is missing; the task records validation "
            "samples, which every evaluation checks. A run that reused a cached "
            "kernel deletes them after its check: pass the task cache directory "
            "to use the run that accepted the kernel"
        )
    samples = load_samples(path)
    if samples_digest(samples) != task["samples"]["sha256"]:
        raise ValueError("Saved validation samples do not match the task digest")
    return samples


def _generate(args: argparse.Namespace, run: Path) -> dict[str, Any]:
    import torch

    from torch_tensorrt.dynamo import CompilationSettings

    from ._factory import _cache_root
    from ._factory import _generate as generate
    from ._worker import _outputs, _restore_metadata

    task = _read_task(run)
    instruction_bundle = _saved_instruction_bundle(args, task, run)
    samples = _saved_samples(task, run)
    config = _generation_config(args, task, run)
    device = torch.device(task["inputs"][0]["device"])
    with torch.cuda.device(device), torch.inference_mode():
        reference = load_reference(run / REFERENCE_FILENAME)
        inputs = tuple(
            torch.randn(
                spec["shape"], dtype=getattr(torch, spec["dtype"]), device=device
            )
            for spec in task["inputs"]
        )
        _restore_metadata(reference, inputs, _outputs(reference(*inputs)))
        _, directory = generate(
            reference,
            config,
            region_id="interactive",
            settings=CompilationSettings(**task["compile_options"]),
            instruction_bundle=instruction_bundle,
            samples=samples,
        )
    # A cache hit revalidates the accepted kernel and runs no search.
    searched = (directory / "search.json").is_file()
    cache_hit = not searched and (directory / "cache_check.json").is_file()
    report = (
        read_json_object(directory / "cache_check.json")
        if cache_hit
        else read_json_object(directory / "accepted.json")["validation"]
    )
    return {
        "status": "passed",
        "cache_hit": cache_hit,
        "backend": config.backend,
        "provider": config.provider,
        "cache_dir": str(_cache_root(config)),
        "directory": str(directory),
        "report": report,
    }


def _instructions(backend_name: str) -> str:
    from ._backends import get_backend
    from ._instructions import load_instructions

    backend = get_backend(backend_name)
    guidance = load_instructions(backend_name, backend.instruction_resources)["text"]
    schema = json.dumps(backend.proposal_schema, indent=2)
    return f"{guidance}\n\nPROPOSAL SCHEMA:\n{schema}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Also write the JSON result to this file; TensorRT can print "
        "warnings to stdout while torch_tensorrt is imported",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    instructions = commands.add_parser(
        "instructions",
        help="Print packaged author guidance and the proposal schema without a model call",
    )
    instructions.add_argument("--backend", default="triton")
    inspect = commands.add_parser("inspect", help="Print a saved task and its results")
    inspect.add_argument("path", type=Path, help=_PATH_HELP)
    evaluate = commands.add_parser(
        "evaluate",
        help="Evaluate a proposal on fresh holdout inputs using the real worker",
    )
    evaluate.add_argument("path", type=Path, help=_PATH_HELP)
    evaluate.add_argument(
        "--candidate",
        required=True,
        type=Path,
        help="A candidate JSON object, or a manifest with a 'candidate' field",
    )
    evaluate.add_argument(
        "--timeout",
        type=float,
        help="Seconds, including any wait for the GPU (default: the task's "
        "candidate timeout)",
    )
    generate = commands.add_parser(
        "generate", help="Run the same generation controller on a saved reference"
    )
    generate.add_argument("path", type=Path, help=_PATH_HELP)
    generate.add_argument("--provider", default="claude")
    generate.add_argument("--backend")
    generate.add_argument("--model")
    generate.add_argument("--instructions")
    generate.add_argument("--instruction-file", action="append", default=[], type=Path)
    generate.add_argument(
        "--cache-dir", type=Path, help="Cache root; defaults to the run's own"
    )
    generate.add_argument("--objective", choices=("candidate", "eager", "tensorrt"))
    generate.add_argument("--min-speedup", type=float)
    # Omitted budget and search options keep the values recorded by the task.
    generate.add_argument(
        "--attempts", dest="max_attempts", type=int, help="Maximum author rounds"
    )
    generate.add_argument(
        "--timeout", dest="timeout_s", type=float, help="Total generation seconds"
    )
    generate.add_argument(
        "--candidate-timeout",
        dest="candidate_timeout_s",
        type=float,
        help="Seconds per candidate evaluation",
    )
    generate.add_argument(
        "--author-timeout",
        dest="author_timeout_s",
        type=float,
        help="Seconds per author round, including evidence reads",
    )
    generate.add_argument(
        "--max-context-requests",
        type=int,
        help="Evidence reads and diagnoses per author round",
    )
    generate.add_argument(
        "--cache",
        dest="use_cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reuse a revalidated cached kernel; --no-cache searches again",
    )
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
    generate.add_argument(
        "--screening",
        choices=("launch", "engine"),
        help="Run search evaluations as direct launches or trial engines; "
        "finalists always run in an engine",
    )
    return parser


@contextmanager
def _stdout_to_stderr() -> Iterator[None]:
    """Keep stdout for the JSON result while TensorRT and plugins load.

    Native libraries print warnings straight to file descriptor 1.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "instructions":
            print(_instructions(args.backend))
            return
        run = _run_directory(args.path)
        with _stdout_to_stderr():
            if args.command == "inspect":
                result = _inspect(run)
            elif args.command == "evaluate":
                result = _evaluate(args, run)
            else:
                result = _generate(args, run)
        if args.output is not None:
            write_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.command == "evaluate" and not result["report"]["meets_objective"]:
            raise SystemExit(1)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
