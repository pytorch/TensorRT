# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded author/evaluator loop, persisted evidence, and kernel cache."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import secrets
import sys
import tempfile
import time
from dataclasses import asdict
from functools import partial
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable

import torch
from filelock import FileLock, Timeout

from torch_tensorrt.kernels import KernelSpec
from torch_tensorrt.region_annotations import KernelGenerator, RegionError

from .._plugins import _boundaries
from . import _providers
from ._backends import candidate_variants, get_backend, validate_tuning_space
from ._candidate import make_registration, validate_candidate
from ._instructions import EvidenceCatalog, load_instructions, validate_instructions

_LOGGER = logging.getLogger(__name__)
_VERSION = 4


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _task_digest(task: dict[str, Any]) -> str:
    # Scheduling and diagnostics do not change kernel compatibility. Tuning
    # domains do: a cached candidate must satisfy the current user's constraints.
    identity = {k: v for k, v in task.items() if k not in ("search", "diagnostics")}
    identity["tuning"] = task.get("search", {}).get("tuning", {})
    return _digest(identity)


def _write_json(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _reference(
    child: torch.fx.GraphModule,
) -> tuple[torch.fx.GraphModule, dict[str, Any]]:
    graph = torch.fx.Graph()
    nodes: dict[torch.fx.Node, torch.fx.Node] = {}
    constants = {}
    input_count = 0
    for node in child.graph.nodes:
        if node.op == "placeholder":
            nodes[node] = graph.placeholder(f"arg{input_count}")
            input_count += 1
        else:
            if node.op not in ("get_attr", "call_function", "output"):
                raise RegionError(
                    f"Agent generation cannot export reference node {node.op!r}"
                )
            if node.op == "call_function":
                target = node.target
                if isinstance(target, torch._ops.OpOverload):
                    if target.namespace not in ("aten", "prims"):
                        raise RegionError(
                            "Agent generation requires an ATen reference graph"
                        )
                    if torch.Tag.nondeterministic_seeded in target.tags:
                        raise RegionError(
                            "Agent generation requires a deterministic reference"
                        )
            if node.op == "get_attr":
                value = child
                for part in str(node.target).split("."):
                    value = getattr(value, part)
                if isinstance(value, torch.Tensor):
                    raise RegionError(
                        "Agent-generated regions require tensor constants to be boundary inputs; "
                        f"captured attribute {node.target!r} is not an input"
                    )
                else:
                    constants[str(node.target)] = value
            nodes[node] = graph.node_copy(node, lambda old: nodes[old])
            if node.op != "output":
                # GraphModule deserialization can rename nodes. Cache identity
                # depends on operations and dependencies, not incidental names.
                nodes[node].name = f"value{len(nodes)}"
        nodes[node].meta = {}
    reference = torch.fx.GraphModule(child, graph)
    reference.meta = {}
    return reference, constants


def _target(device: torch.device, backend: str) -> dict[str, Any]:
    return get_backend(backend).target_metadata(device)


def _prompt_history(
    history: list[dict[str, Any]], best: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Keep recent feedback and the incumbent without resending every kernel."""
    indices = set(range(max(0, len(history) - 3), len(history)))
    passing = [
        index
        for index, attempt in enumerate(history)
        if attempt["result"].get("status") == "passed"
    ]
    if best is not None:
        incumbent = next(
            (i for i in reversed(passing) if history[i]["candidate"] == best), None
        )
        if incumbent is not None:
            indices.add(incumbent)
    elif passing:
        indices.add(min(passing, key=lambda i: history[i]["result"]["latency_ms"]))
    return [history[index] for index in sorted(indices)]


def _prompt(
    task: dict[str, Any],
    history: list[dict[str, Any]],
    remaining_s: float,
    instructions: dict[str, Any],
    evidence: EvidenceCatalog,
    best: dict[str, Any] | None = None,
    action: str = "author",
) -> str:
    return (
        str(instructions["text"])
        + "\nRESPONSE PROTOCOL: Return action, candidate, candidates, resource, offset. "
        "For action=propose, supply either one candidate or a bounded candidates array, "
        "following the backend schema; unused fields are null. "
        "For action=read, resource/offset identify a page from the evidence catalog. "
        "For action=diagnose, request a measured diagnostic of the current best candidate; "
        "this requires diagnostics=true and shares the evaluation and context budgets. "
        "Use action=stop when further authoring is not useful; Python still validates the finalist. "
        "Only catalog resources are readable. The controller runs all compilation and evaluation. "
        "Use feedback to change source or launch settings; unchanged passing candidates reuse their prior evaluation."
        + f"\nSEARCH STEP: {action}. Remaining search time: {max(0, int(remaining_s))} seconds."
        + "\nTASK:\n"
        + json.dumps(task, indent=2)
        + "\nRECENT ATTEMPTS AND BEST CANDIDATE:\n"
        + json.dumps(_prompt_history(history, best), indent=2)
        + "\nEVIDENCE CATALOG:\n"
        + json.dumps(evidence.catalog(), indent=2)
    )


def _response_schema(
    candidate_schema: dict[str, Any], batch_size: int = 1
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": ["propose", "read", "diagnose", "stop"],
            },
            "candidate": {"anyOf": [candidate_schema, {"type": "null"}]},
            "candidates": {
                "anyOf": [
                    {
                        "type": "array",
                        "items": candidate_schema,
                        "minItems": 1,
                        "maxItems": batch_size,
                    },
                    {"type": "null"},
                ]
            },
            "resource": {"type": ["string", "null"]},
            "offset": {"type": ["integer", "null"], "minimum": 0},
        },
        "required": ["action", "candidate", "candidates", "resource", "offset"],
    }


def _author(
    config: KernelGenerator,
    prompt: str,
    schema: dict[str, Any],
    directory: Path,
    deadline: float,
    evidence: EvidenceCatalog,
    diagnose: Callable[[Path, float], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    # Reserve time for evaluation, including final validation; evidence reads share
    # this author deadline rather than receiving a fresh timeout on every turn.
    remaining = deadline - time.monotonic()
    reserve = min(2 * config.candidate_timeout_s, remaining / 2)
    author_deadline = min(
        deadline - reserve, time.monotonic() + config.author_timeout_s
    )
    for turn in range(config.max_context_requests + 1):
        timeout = author_deadline - time.monotonic()
        if timeout <= 0:
            raise TimeoutError(
                "Authoring budget exhausted; evaluation time is reserved"
            )
        turn_dir = directory if turn == 0 else directory / f"read-{turn}"
        turn_dir.mkdir(exist_ok=True)
        turn_prompt = prompt + (
            f"\nAUTHOR BUDGET: {config.max_context_requests - turn} evidence reads "
            f"remaining; {max(0, int(timeout))} seconds. "
            "Reads and diagnostics share this allowance. "
            "When no reads remain, return action=propose or action=stop."
        )
        (turn_dir / "prompt.txt").write_text(turn_prompt, encoding="utf-8")
        reply = _providers.propose(
            provider=config.provider,
            model=config.model,
            prompt=turn_prompt,
            workdir=turn_dir,
            timeout_s=timeout,
            schema=schema,
            evidence=(
                {"catalog": evidence.catalog()} if ":" in config.provider else None
            ),
        )
        # Imported providers may retain the original direct-candidate interface.
        if "action" not in reply:
            return [reply]
        if reply.get("action") == "stop":
            return []
        if reply.get("action") == "propose":
            candidate, candidates = reply.get("candidate"), reply.get("candidates")
            if candidate is not None and candidates is not None:
                raise ValueError(
                    "Proposal must supply candidate or candidates, not both"
                )
            batch = [candidate] if candidate is not None else candidates
            if (
                not isinstance(batch, list)
                or not 1 <= len(batch) <= config.search.batch_size
                or any(not isinstance(item, dict) for item in batch)
            ):
                raise ValueError(
                    "Proposal must contain a bounded batch of candidate objects"
                )
            return batch
        if reply.get("action") not in ("read", "diagnose"):
            raise ValueError("Unknown author response action")
        if turn == config.max_context_requests:
            raise ValueError("Evidence request limit reached; return a candidate")
        try:
            if reply["action"] == "diagnose":
                if diagnose is None or not config.diagnostics:
                    raise ValueError("Diagnostics are not enabled for this search")
                page = diagnose(turn_dir, author_deadline)
            else:
                resource = reply.get("resource")
                if not isinstance(resource, str):
                    raise ValueError("Evidence resource must be a catalog name")
                offset = reply.get("offset")
                page = evidence.read(resource, 0 if offset is None else offset)
        except (ValueError, KeyError, TypeError) as error:
            page = {"error": str(error)}
        _write_json(turn_dir / "evidence.json", {"request": reply, "result": page})
        prompt += "\nEVIDENCE RESULT:\n" + json.dumps(page)
    raise AssertionError("Author request loop did not return")


def _meets_objective(report: dict[str, Any], config: KernelGenerator) -> bool:
    if report.get("status") != "passed":
        return False
    if config.objective == "candidate":
        return True
    baseline = report.get("baseline_latency_ms")
    if (
        isinstance(baseline, bool)
        or not isinstance(baseline, (int, float))
        or not math.isfinite(baseline)
        or baseline <= 0
    ):
        return False
    measurements = report.get("benchmark", {})
    baseline_noise = measurements.get("baseline", {}).get("mad_ms", 0)
    candidate_noise = measurements.get("candidate", {}).get("mad_ms", 0)
    # A noise allowance, not a statistical confidence bound.
    return bool(
        baseline - baseline_noise
        >= config.min_speedup * (report["latency_ms"] + candidate_noise)
    )


def _improves(report: dict[str, Any], best_report: dict[str, Any] | None) -> bool:
    if best_report is None:
        return True
    incumbent = report.get("incumbent_latency_ms", best_report["latency_ms"])
    # Within measured noise, retain the incumbent instead of rewarding jitter.
    measurements = report.get("benchmark", {})
    noise = sum(
        measurements.get(arm, {}).get("mad_ms", 0) for arm in ("candidate", "incumbent")
    )
    return bool(incumbent - report["latency_ms"] > noise)


def _next_action(
    history: list[dict[str, Any]],
    best: dict[str, Any] | None,
    stale: int,
    config: KernelGenerator,
) -> str:
    """Choose the next experiment from measured outcomes, not agent verdicts."""
    if (
        config.search.max_no_improvement is not None
        and stale >= config.search.max_no_improvement
    ):
        return "finalize"
    if history:
        report = history[-1]["result"]
        if report.get("status") == "error":
            stage = report.get("stage")
            if stage in ("provider", "baseline", "budget"):
                return "finalize"
            if stage in ("source", "compile"):
                return "repair_compile"
            if stage == "correctness":
                return "repair_correctness"
            return "repair"
        if stale or best is None:
            return "rewrite"
    return "optimize" if best is not None else "author"


def _evaluate(
    request: dict[str, Any], directory: Path, timeout_s: float
) -> dict[str, Any]:
    request_path = directory / "evaluation.json"
    result_path = directory / "result.json"
    _write_json(request_path, request)
    command = [
        sys.executable,
        "-m",
        "torch_tensorrt.dynamo.region_compilation._generation._worker",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]
    try:
        completed = _providers.run_process(
            command, input="", cwd=directory, timeout_s=timeout_s
        )
        (directory / "worker.log").write_text(
            completed.stdout + "\n" + completed.stderr
        )
        if completed.returncode != 0:
            return {
                "status": "error",
                "error": f"Evaluator exited {completed.returncode}: {completed.stderr[-3000:]}",
            }
        report = json.loads(result_path.read_text())
        if not isinstance(report, dict) or report.get("status") not in (
            "passed",
            "error",
        ):
            return {"status": "error", "error": "Evaluator returned an invalid report"}
        latency = report.get("latency_ms")
        if report.get("status") == "passed" and (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
        ):
            return {"status": "error", "error": "Evaluator returned an invalid timing"}
        return report
    except (OSError, ValueError, TimeoutError, RuntimeError) as error:
        return {"status": "error", "error": str(error)[-3000:]}


def _cache_root(config: KernelGenerator) -> Path:
    if config.cache_dir is not None:
        return Path(config.cache_dir).resolve()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "torch_tensorrt" / "region_kernels").resolve()


def generate_registration(
    child: torch.fx.GraphModule, config: KernelGenerator, *, region_id: str
) -> KernelSpec | partial[None]:
    """Generate a kernel after capture and return the existing registration form."""
    registration, _ = _generate(child, config, region_id=region_id)
    return registration


def _generate(
    child: torch.fx.GraphModule,
    config: KernelGenerator,
    *,
    region_id: str,
    instruction_bundle: dict[str, Any] | None = None,
) -> tuple[KernelSpec | partial[None], Path]:
    """Return a registration and the exact run containing its validation evidence."""
    input_specs, output_specs = _boundaries(child, region_id)
    if any(
        spec.dtype not in (torch.float16, torch.float32, torch.bfloat16)
        for spec in input_specs + output_specs
    ):
        raise RegionError(
            "Agent-generated regions support float16, float32, and bfloat16 boundaries"
        )
    if any(math.prod(spec.shape) == 0 for spec in input_specs + output_specs):
        raise RegionError("Agent-generated regions require nonempty tensors")
    reference, constants = _reference(child)
    backend = get_backend(config.backend)
    instructions = (
        validate_instructions(instruction_bundle, config.backend)
        if instruction_bundle is not None
        else load_instructions(
            config.backend,
            backend.instruction_resources,
            config.instructions,
            config.instruction_files,
        )
    )
    search = asdict(config.search)
    search["tuning"] = validate_tuning_space(config.backend, config.search.tuning)

    def describe(spec: Any) -> dict[str, Any]:
        return {
            "shape": list(spec.shape),
            "dtype": str(spec.dtype).removeprefix("torch."),
            "device": str(spec.device),
        }

    task = {
        "version": _VERSION,
        "reference": reference.code,
        "constants": constants,
        "inputs": [describe(s) for s in input_specs],
        "outputs": [describe(s) for s in output_specs],
        "backend": config.backend,
        "backend_version": backend.version,
        "instructions_sha256": instructions["sha256"],
        "proposal_schema_sha256": _digest(backend.proposal_schema),
        "objective": config.objective,
        "min_speedup": config.min_speedup,
        "search": search,
        "diagnostics": config.diagnostics,
        "analysis": {
            "operations": [
                str(node.target)
                for node in reference.graph.nodes
                if node.op == "call_function"
            ],
            "boundary_bytes": sum(
                math.prod(spec.shape) * spec.dtype.itemsize
                for spec in input_specs + output_specs
            ),
        },
        "target": _target(input_specs[0].device, config.backend),
        "validation": {
            "num_tests": config.num_tests,
            "benchmark_iters": config.benchmark_iters,
            "benchmark_repeats": config.benchmark_repeats,
            "check_nonfinite": config.check_nonfinite,
            "rtol": config.rtol,
            "atol": config.atol,
        },
    }
    key = _task_digest(task)
    root = _cache_root(config)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / key
    cache.mkdir(exist_ok=True)
    deadline = time.monotonic() + config.timeout_s
    try:
        with FileLock(str(cache / "generation.lock"), timeout=config.timeout_s):
            return _generate_locked(
                reference,
                task,
                config,
                cache,
                deadline,
                input_specs,
                output_specs,
                region_id,
                instructions,
            )
    except RegionError:
        raise
    except Exception as error:
        raise RegionError(
            f"Kernel generation for {region_id!r} failed: {error}. Artifacts: {cache}"
        ) from error


def _generate_locked(
    reference: torch.fx.GraphModule,
    task: dict[str, Any],
    config: KernelGenerator,
    cache: Path,
    deadline: float,
    input_specs: list[Any],
    output_specs: list[Any],
    region_id: str,
    instructions: dict[str, Any],
) -> tuple[KernelSpec | partial[None], Path]:
    run = Path(tempfile.mkdtemp(prefix="run-", dir=cache))
    _write_json(run / "task.json", task)
    _write_json(run / "instructions.json", instructions)
    policy = config.search
    schema = _response_schema(
        get_backend(config.backend).proposal_schema, policy.batch_size
    )
    _write_json(run / "proposal.schema.json", schema)
    reference_path = run / "reference.pt"
    torch.save(reference, reference_path)
    evaluation = {
        "reference_path": str(reference_path),
        "backend": config.backend,
        "backend_version": task["backend_version"],
        "proposal_schema_sha256": task["proposal_schema_sha256"],
        "inputs": task["inputs"],
        "outputs": task["outputs"],
        "objective": config.objective,
        "diagnostics": config.diagnostics,
        "validation_seed": secrets.randbits(63),
        **task["validation"],
    }
    _write_json(run / "evaluation-template.json", evaluation)
    evidence = EvidenceCatalog()
    for resource in instructions["resources"]:
        evidence.add(resource["name"], resource["text"])
    evidence.add("task.json", json.dumps(task, indent=2))
    # Keep a final evaluator slice independent of author and search timeouts.
    search_deadline = deadline - min(
        config.candidate_timeout_s, max(0, deadline - time.monotonic()) / 3
    )
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_report: dict[str, Any] | None = None
    contenders: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    used = 0
    stale = 0
    max_candidates = policy.max_candidates or config.max_attempts * policy.batch_size
    manifest = cache / "accepted.json"
    checkpoint = cache / "checkpoint.json"
    task_digest = _task_digest(task)

    def evaluate(
        candidate: dict[str, Any],
        directory: Path,
        *,
        holdout: bool = False,
        incumbent: dict[str, Any] | None = None,
        until: float | None = None,
    ) -> dict[str, Any]:
        nonlocal used
        evaluation_deadline = deadline if holdout else search_deadline
        if until is not None:
            evaluation_deadline = min(evaluation_deadline, until)
        remaining = evaluation_deadline - time.monotonic()
        if remaining <= 0 or (not holdout and used >= max_candidates):
            return {
                "status": "error",
                "stage": "budget",
                "error": "Generation evaluation or wall-time budget exhausted",
            }
        gpu_lock = FileLock(
            str(cache.parent / f"gpu-{input_specs[0].device.index}.lock")
        )
        try:
            with gpu_lock.acquire(timeout=remaining):
                remaining = evaluation_deadline - time.monotonic()
                if remaining <= 0:
                    return {
                        "status": "error",
                        "stage": "budget",
                        "error": "Generation wall-time budget exhausted waiting for GPU",
                    }
                request = {**evaluation, "candidate": candidate}
                if holdout:
                    request.update(
                        validation_seed=secrets.randbits(63), validation_phase="holdout"
                    )
                elif incumbent is not None:
                    request["incumbent"] = incumbent
                if not holdout:
                    used += 1
                return _evaluate(
                    request, directory, min(config.candidate_timeout_s, remaining)
                )
        except Timeout:
            return {
                "status": "error",
                "stage": "budget",
                "error": "GPU evaluation lock timed out",
            }

    if config.use_cache and manifest.exists():
        try:
            stored = json.loads(manifest.read_text(encoding="utf-8"))
            candidate = stored["candidate"]
            if stored["task_sha256"] != task_digest or stored[
                "candidate_sha256"
            ] != _digest(candidate):
                raise ValueError("Cache identity or content digest mismatch")
            candidate = validate_candidate(
                candidate, config.backend, len(input_specs), len(output_specs)
            )
            report = evaluate(candidate, run, holdout=True)
            _write_json(run / "cache_check.json", report)
            if _meets_objective(report, config):
                _LOGGER.info("Agent region %s: cache hit (%s)", region_id, cache)
                return (
                    make_registration(
                        candidate, config.backend, input_specs, output_specs, run
                    ),
                    run,
                )
            _LOGGER.warning("Agent region cache revalidation failed: %s", report)
        except (OSError, KeyError, ValueError, TypeError) as error:
            _LOGGER.warning("Ignoring invalid agent region cache: %s", error)

    def ranked() -> list[dict[str, Any]]:
        selected = contenders.get(_digest(best)) if best is not None else None
        others = sorted(
            (entry for entry in contenders.values() if entry is not selected),
            key=lambda entry: entry["report"]["latency_ms"],
        )
        return ([selected] if selected is not None else []) + others

    def save_checkpoint() -> None:
        state = {
            "task_sha256": task_digest,
            "run": str(run),
            "history": _prompt_history(history, best),
            "finalists": [entry["candidate"] for entry in ranked()[: policy.finalists]],
            "pending": pending,
        }
        _write_json(checkpoint, {"state": state, "sha256": _digest(state)})

    def register_logs(directory: Path) -> None:
        for log in (
            directory / "worker.log",
            directory / "provider.log",
            *sorted(directory.glob("read-*/provider.log")),
        ):
            if log.is_file():
                evidence.add(
                    str(log.relative_to(run)),
                    log.read_text(encoding="utf-8", errors="replace"),
                )

    def record(
        candidate: dict[str, Any] | None,
        report: dict[str, Any],
        directory: Path,
        action: str,
    ) -> bool:
        nonlocal best, best_report
        report = dict(report)
        report["meets_objective"] = _meets_objective(report, config)
        feedback_path = directory / "feedback.json"
        _write_json(feedback_path, report)
        history.append({"candidate": candidate, "result": report, "action": action})
        prefix = str(directory.relative_to(run))
        evidence.add(f"{prefix}/feedback.json", json.dumps(report, indent=2))
        improved = False
        if candidate is not None:
            _write_json(directory / "candidate.json", candidate)
            evidence.add(f"{prefix}/candidate.json", json.dumps(candidate, indent=2))
            if isinstance(candidate.get("source"), str):
                evidence.add(f"{prefix}/source", candidate["source"])
            candidate_digest = _digest(candidate)
            if report.get("status") == "passed" and "reused_from" not in report:
                contenders[candidate_digest] = {
                    "candidate": candidate,
                    "report": report,
                    "evaluation_path": str(feedback_path),
                }
                if _improves(report, best_report):
                    best, best_report = candidate, report
                    improved = True
        register_logs(directory)
        _write_json(run / "history.json", history)
        _LOGGER.info("Agent region %s: %s", region_id, report)
        return improved

    def run_pending(directories: list[Path], action: str) -> bool:
        improved = False
        for directory in directories:
            if (
                not pending
                or used >= max_candidates
                or time.monotonic() >= search_deadline
            ):
                break
            candidate = pending[0]
            previous = contenders.get(_digest(candidate))
            if previous is None:
                report = evaluate(candidate, directory, incumbent=best)
            else:
                report = {
                    **previous["report"],
                    "reused_from": previous["evaluation_path"],
                }
            improved = record(candidate, report, directory, action) or improved
            if report.get("stage") == "budget":
                save_checkpoint()
                break
            pending.pop(0)
            save_checkpoint()
        return improved

    # A checkpoint carries proposals and historical context, never current proof.
    # Every retained candidate is validated and measured again in this invocation.
    if policy.resume and checkpoint.is_file():
        try:
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            state = saved["state"]
            if state["task_sha256"] != task_digest or saved["sha256"] != _digest(state):
                raise ValueError("Checkpoint identity or content digest mismatch")
            old_history = state["history"]
            if not isinstance(old_history, list) or any(
                not isinstance(entry, dict)
                or not isinstance(entry.get("result"), dict)
                or not (
                    entry.get("candidate") is None
                    or isinstance(entry["candidate"], dict)
                )
                for entry in old_history
            ):
                raise ValueError("Invalid checkpoint history")
            restored = []
            seen = set()
            for item in [*state["finalists"][: policy.finalists], *state["pending"]]:
                candidate = validate_candidate(
                    item, config.backend, len(input_specs), len(output_specs)
                )
                candidate_digest = _digest(candidate)
                if candidate_digest not in seen:
                    seen.add(candidate_digest)
                    restored.append(candidate)
            history.extend({**entry, "historical": True} for entry in old_history)
            pending.extend(restored)
            evidence.add("previous-history.json", json.dumps(history, indent=2))
            _write_json(
                run / "resume.json", {"from": state["run"], "candidates": len(pending)}
            )
        except (OSError, KeyError, ValueError, TypeError) as error:
            _LOGGER.warning("Ignoring invalid generation checkpoint: %s", error)
        directories = []
        for index in range(len(pending)):
            directory = run / f"resume-{index + 1}"
            directory.mkdir()
            directories.append(directory)
        save_checkpoint()
        run_pending(directories, "resume")

    def diagnose(directory: Path, until: float) -> dict[str, Any]:
        if best is None:
            raise ValueError("Diagnostics require a current verified candidate")
        diagnostic_dir = directory / "diagnostics"
        diagnostic_dir.mkdir()
        report = evaluate(best, diagnostic_dir, until=until)
        _write_json(diagnostic_dir / "feedback.json", report)
        evidence.add(
            str((diagnostic_dir / "feedback.json").relative_to(run)),
            json.dumps(report, indent=2),
        )
        return report

    stop_reason = "attempt_limit"
    for attempt in range(config.max_attempts):
        remaining = search_deadline - time.monotonic()
        if remaining <= 0 or used >= max_candidates:
            stop_reason = "budget"
            break
        action = _next_action(
            [entry for entry in history if not entry.get("historical")],
            best,
            stale,
            config,
        )
        if action == "finalize":
            stop_reason = "policy"
            break
        directory = run / f"attempt-{attempt + 1}"
        directory.mkdir()
        prompt = _prompt(task, history, remaining, instructions, evidence, best, action)
        _LOGGER.info(
            "Agent region %s: %s attempt %d/%d with %s (%s)",
            region_id,
            action,
            attempt + 1,
            config.max_attempts,
            config.provider,
            config.backend,
        )
        try:
            proposals = _author(
                config, prompt, schema, directory, search_deadline, evidence, diagnose
            )
        except (OSError, ValueError, TypeError, TimeoutError, RuntimeError) as error:
            provider_failed = isinstance(error, _providers.ProviderError)
            record(
                None,
                {
                    "status": "error",
                    "stage": "provider" if provider_failed else "author",
                    "error": str(error)[-3000:],
                },
                directory,
                action,
            )
            stale += 1
            save_checkpoint()
            if provider_failed:
                stop_reason = "provider"
                break
            continue
        register_logs(directory)
        if not proposals:
            stop_reason = "author_stop"
            save_checkpoint()
            break

        # Round-robin across sources so tuning the first proposal cannot consume
        # the entire batch before a different implementation receives a trial.
        groups = []
        failures = []
        for proposal in proposals:
            try:
                groups.append(
                    candidate_variants(
                        proposal,
                        config.backend,
                        task["search"]["tuning"],
                        policy.max_variants,
                    )
                )
            except (ValueError, TypeError) as error:
                failures.append(str(error))
        options = [
            candidate
            for group in zip_longest(*groups)
            for candidate in group
            if candidate is not None
        ]
        total = len(options) + len(failures)
        directories = []
        for index, option in enumerate([*options, *failures]):
            candidate_directory = (
                directory if total == 1 else directory / f"candidate-{index + 1}"
            )
            candidate_directory.mkdir(exist_ok=True)
            try:
                if isinstance(option, str):
                    raise ValueError(option)
                candidate = validate_candidate(
                    option, config.backend, len(input_specs), len(output_specs)
                )
            except (ValueError, TypeError) as error:
                record(
                    None,
                    {"status": "error", "stage": "source", "error": str(error)[-3000:]},
                    candidate_directory,
                    action,
                )
                continue
            pending.append(candidate)
            directories.append(candidate_directory)
        save_checkpoint()
        improved = run_pending(directories, "tune" if policy.tuning else action)
        stale = 0 if improved else stale + 1

    _write_json(run / "history.json", history)
    _write_json(
        run / "search.json",
        {
            "stop_reason": stop_reason,
            "evaluations": used,
            "max_candidates": max_candidates,
            "stale_rounds": stale,
        },
    )
    save_checkpoint()
    choices = [entry for entry in ranked() if entry["report"]["meets_objective"]][
        : policy.finalists
    ]
    if not choices:
        detail = (
            history[-1]["result"].get(
                "error", "No candidate met the performance objective"
            )
            if history
            else "Generation budget exhausted"
        )
        raise RegionError(
            f"Agent generated no valid kernel for region {region_id!r}: {detail}. Artifacts: {run}"
        )

    # Keep the selection winner first. Alternatives are independently rechecked,
    # with a time slice reserved for each, before any cache entry is published.
    final_report: dict[str, Any] = {}
    for index, entry in enumerate(choices):
        final_directory = run / ("holdout" if index == 0 else f"holdout-{index + 1}")
        final_directory.mkdir()
        remaining = deadline - time.monotonic()
        final_report = evaluate(
            entry["candidate"],
            final_directory,
            holdout=True,
            until=time.monotonic() + max(0, remaining) / (len(choices) - index),
        )
        _write_json(final_directory / "feedback.json", final_report)
        if not _meets_objective(final_report, config):
            contenders.pop(_digest(entry["candidate"]), None)
            history.append(
                {
                    "candidate": entry["candidate"],
                    "result": final_report,
                    "action": "holdout",
                }
            )
            _write_json(run / "history.json", history)
            save_checkpoint()
            continue
        # Construct registration before publishing a successful cache entry.
        registration = make_registration(
            entry["candidate"], config.backend, input_specs, output_specs, run
        )
        accepted = {
            "task_sha256": task_digest,
            "candidate_sha256": _digest(entry["candidate"]),
            "candidate": entry["candidate"],
            "report": entry["report"],
            "validation": final_report,
            "evaluation_path": entry["evaluation_path"],
            "provider": config.provider,
            "model": config.model,
            "instructions_sha256": instructions["sha256"],
            "run": str(run),
        }
        _write_json(run / "accepted.json", accepted)
        if config.use_cache:
            _write_json(manifest, accepted)
        checkpoint.unlink(missing_ok=True)
        _LOGGER.info("Agent region %s: selected kernel; artifacts: %s", region_id, run)
        return registration, run
    _write_json(run / "history.json", history)
    raise RegionError(
        f"Generated kernel for region {region_id!r} failed final validation: "
        f"{final_report.get('error', 'Performance objective not met')}. Artifacts: {run}"
    )
