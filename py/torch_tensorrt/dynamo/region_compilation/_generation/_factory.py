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
from functools import partial
from pathlib import Path
from typing import Any

import torch
from filelock import FileLock, Timeout

from torch_tensorrt.kernels import KernelSpec
from torch_tensorrt.region_annotations import KernelGenerator, RegionError

from .._plugins import _boundaries
from . import _providers
from ._backends import get_backend
from ._candidate import make_registration, validate_candidate
from ._instructions import EvidenceCatalog, load_instructions

_LOGGER = logging.getLogger(__name__)
_VERSION = 3


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


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
) -> str:
    mode = "author"
    if history:
        mode = (
            "optimize" if history[-1]["result"].get("status") == "passed" else "repair"
        )
    return (
        str(instructions["text"])
        + "\nRESPONSE PROTOCOL: Return a JSON object with action, candidate, resource, offset. "
        "For action=propose, candidate follows the supplied backend schema and resource/offset are null. "
        "For action=read, candidate is null and resource/offset identify a page from the evidence catalog. "
        "Only catalog resources are readable. The controller runs all compilation and evaluation. "
        "Use feedback to change source or launch settings; unchanged passing candidates reuse their prior evaluation."
        + f"\nSEARCH STEP: {mode}. Remaining search time: {max(0, int(remaining_s))} seconds."
        + "\nTASK:\n"
        + json.dumps(task, indent=2)
        + "\nRECENT ATTEMPTS AND BEST CANDIDATE:\n"
        + json.dumps(_prompt_history(history, best), indent=2)
        + "\nEVIDENCE CATALOG:\n"
        + json.dumps(evidence.catalog(), indent=2)
    )


def _response_schema(candidate_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["propose", "read"]},
            "candidate": {"anyOf": [candidate_schema, {"type": "null"}]},
            "resource": {"type": ["string", "null"]},
            "offset": {"type": ["integer", "null"], "minimum": 0},
        },
        "required": ["action", "candidate", "resource", "offset"],
    }


def _author(
    config: KernelGenerator,
    prompt: str,
    schema: dict[str, Any],
    directory: Path,
    deadline: float,
    evidence: EvidenceCatalog,
) -> dict[str, Any]:
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
            "When no reads remain, return action=propose."
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
            return reply
        if reply.get("action") == "propose":
            if not isinstance(reply.get("candidate"), dict):
                raise ValueError("Proposal must contain a candidate object")
            return dict(reply["candidate"])
        if reply.get("action") != "read":
            raise ValueError("Unknown author response action")
        if turn == config.max_context_requests:
            raise ValueError("Evidence request limit reached; return a candidate")
        try:
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
    child: torch.fx.GraphModule, config: KernelGenerator, *, region_id: str
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
    instructions = load_instructions(
        config.backend,
        backend.instruction_resources,
        config.instructions,
        config.instruction_files,
    )

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
    key = _digest(task)
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
    schema = _response_schema(get_backend(config.backend).proposal_schema)
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
    best_evaluation_path: Path | None = None
    passing_evaluations: dict[str, tuple[dict[str, Any], Path]] = {}
    manifest = cache / "accepted.json"

    def evaluate(
        candidate: dict[str, Any],
        directory: Path,
        *,
        holdout: bool = False,
        incumbent: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evaluation_deadline = deadline if holdout else search_deadline
        remaining = evaluation_deadline - time.monotonic()
        if remaining <= 0:
            return {
                "status": "error",
                "stage": "budget",
                "error": "Generation wall-time budget exhausted",
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
            if stored["task_sha256"] != _digest(task) or stored[
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

    for attempt in range(config.max_attempts):
        remaining = search_deadline - time.monotonic()
        if remaining <= 0:
            break
        directory = run / f"attempt-{attempt + 1}"
        directory.mkdir()
        prompt = _prompt(task, history, remaining, instructions, evidence, best)
        _LOGGER.info(
            "Agent region %s: attempt %d/%d with %s (%s)",
            region_id,
            attempt + 1,
            config.max_attempts,
            config.provider,
            config.backend,
        )
        candidate = None
        evaluation_path = directory / "feedback.json"
        provider_failed = False
        stage = "author"
        try:
            proposal = _author(
                config, prompt, schema, directory, search_deadline, evidence
            )
            stage = "source"
            candidate = validate_candidate(
                proposal, config.backend, len(input_specs), len(output_specs)
            )
            _write_json(directory / "candidate.json", candidate)
            candidate_digest = _digest(candidate)
            if candidate_digest in passing_evaluations:
                previous_report, evaluation_path = passing_evaluations[candidate_digest]
                report = {**previous_report, "reused_from": str(evaluation_path)}
            else:
                report = evaluate(candidate, directory, incumbent=best)
                if report.get("status") == "passed":
                    passing_evaluations[candidate_digest] = (report, evaluation_path)
        except _providers.ProviderError as error:
            report = {
                "status": "error",
                "stage": "provider",
                "error": str(error)[-3000:],
            }
            provider_failed = True
        except (OSError, ValueError, TypeError, TimeoutError, RuntimeError) as error:
            report = {"status": "error", "stage": stage, "error": str(error)[-3000:]}
        report["meets_objective"] = _meets_objective(report, config)
        _write_json(directory / "feedback.json", report)
        history.append({"candidate": candidate, "result": report})
        evidence.add(
            f"attempt-{attempt + 1}/feedback.json", json.dumps(report, indent=2)
        )
        if candidate is not None:
            evidence.add(
                f"attempt-{attempt + 1}/candidate.json", json.dumps(candidate, indent=2)
            )
            if isinstance(candidate.get("source"), str):
                evidence.add(f"attempt-{attempt + 1}/source", candidate["source"])
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
        if (
            report["meets_objective"]
            and "reused_from" not in report
            and _improves(report, best_report)
        ):
            best, best_report, best_evaluation_path = candidate, report, evaluation_path
        _LOGGER.info("Agent region %s: %s", region_id, report)
        if provider_failed:
            break

    _write_json(run / "history.json", history)
    if best is None:
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
    final_directory = run / "holdout"
    final_directory.mkdir()
    final_report = evaluate(best, final_directory, holdout=True)
    _write_json(final_directory / "feedback.json", final_report)
    if not _meets_objective(final_report, config):
        raise RegionError(
            f"Generated kernel for region {region_id!r} failed final validation: "
            f"{final_report.get('error', 'Performance objective not met')}. Artifacts: {run}"
        )
    accepted = {
        "task_sha256": _digest(task),
        "candidate_sha256": _digest(best),
        "candidate": best,
        "report": best_report,
        "validation": final_report,
        "evaluation_path": str(best_evaluation_path),
        "provider": config.provider,
        "model": config.model,
        "instructions_sha256": instructions["sha256"],
        "run": str(run),
    }
    # Construct the final registration before publishing a successful cache entry.
    registration = make_registration(
        best, config.backend, input_specs, output_specs, run
    )
    _write_json(run / "accepted.json", accepted)
    if config.use_cache:
        _write_json(manifest, accepted)
    _LOGGER.info("Agent region %s: selected kernel; artifacts: %s", region_id, run)
    return registration, run
