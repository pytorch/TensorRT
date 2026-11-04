# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded author/evaluator loop, persisted evidence, and kernel cache."""

from __future__ import annotations

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
from typing import TYPE_CHECKING, Any, Callable

import torch
from filelock import FileLock, Timeout

from torch_tensorrt.kernels import KernelSpec
from torch_tensorrt.region import KernelGenerator, RegionError

from .._plugins import _boundaries
from . import _providers
from ._backends import (
    GenerationBackend,
    candidate_variants,
    get_backend,
    validate_tuning_space,
)
from ._instructions import EvidenceCatalog, load_instructions, validate_instructions
from ._util import digest, gpu_lock, read_json_object, scrub, strip_control, write_json

if TYPE_CHECKING:
    from torch_tensorrt.dynamo._settings import CompilationSettings

_LOGGER = logging.getLogger(__name__)
_VERSION = 5
# Starting the worker or an author CLI alone takes about a second, so a shorter
# time slice can only expire; the controller stops instead of launching it.
_MIN_SLICE_S = 1.0
# Prompts list only the newest evidence; the "catalog" resource lists it all.
_PROMPT_EVIDENCE = 20
# User settings that change how TensorRT builds the trial and baseline
# engines. Settings that regions reject never reach generation, and lowering
# settings such as decompose_attention already shaped the captured reference.
_COMPILE_OPTIONS = (
    "disable_tf32",
    "optimization_level",
    "use_fp32_acc",
    "tiling_optimization_level",
    "l2_limit_for_tiling",
    "workspace_size",
    "max_aux_streams",
    "build_route",
    "version_compatible",
)
_BUDGET = (
    "max_attempts",
    "timeout_s",
    "candidate_timeout_s",
    "author_timeout_s",
    "max_context_requests",
    "use_cache",
)
# No proposal can fix these, so the search stops and validates what it has.
_TERMINAL_STAGES = frozenset({"provider", "setup", "baseline", "budget"})
# These reproduce for an unchanged candidate on the run's fixed search inputs.
_DETERMINISTIC_STAGES = frozenset({"source", "compile", "engine", "correctness"})
_PROTOCOL = (
    "\nRESPONSE PROTOCOL: Return action, candidates, resource, and offset; unused "
    "fields are null. For action=propose, candidates holds 1 to {batch_size} "
    "proposals following the backend schema. For action=read, resource names a "
    "readable evidence resource and offset is a character offset into it. "
    "For action=diagnose, request a measured diagnostic of the current best candidate; "
    "this requires diagnostics=true and shares the evaluation and context budgets. "
    "Use action=stop when further authoring is not useful; Python still validates the finalist. "
    "Only catalog resources are readable. The controller runs all compilation and evaluation. "
    "Use feedback to change source or launch settings: an unchanged candidate reuses "
    "its earlier pass or source, compile, engine, or correctness failure."
)


def _task_digest(task: dict[str, Any]) -> str:
    # Scheduling, budgets, diagnostics, and logical device indices do not change
    # kernel compatibility; "target" identifies the GPU. Tuning domains do: a
    # cached candidate must satisfy the current user's constraints.
    identity = {
        key: value
        for key, value in task.items()
        if key not in ("search", "budget", "diagnostics")
    }
    identity["tuning"] = task["search"]["tuning"]
    for side in ("inputs", "outputs"):
        identity[side] = [
            {**spec, "device": spec["device"].partition(":")[0]} for spec in task[side]
        ]
    return digest(identity)


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


def _describe(spec: Any) -> dict[str, Any]:
    return {
        "shape": list(spec.shape),
        "dtype": str(spec.dtype).removeprefix("torch."),
        "device": str(spec.device),
    }


def _compile_options(settings: CompilationSettings | None) -> dict[str, Any]:
    if settings is None:
        from torch_tensorrt.dynamo._settings import CompilationSettings

        settings = CompilationSettings()
    return {name: getattr(settings, name) for name in _COMPILE_OPTIONS}


def _target(backend: GenerationBackend, device: torch.device) -> dict[str, Any]:
    return backend.target_metadata(device)


def make_registration(
    candidate: dict[str, Any],
    backend: GenerationBackend,
    input_specs: list[Any],
    output_specs: list[Any],
    directory: Path,
) -> KernelSpec | partial[None]:
    """Rebuild the backend registration of a validated, serializable proposal."""
    return backend.make_registration(candidate, input_specs, output_specs, directory)


def _error(stage: str, message: str) -> dict[str, Any]:
    return {"status": "error", "stage": stage, "error": message}


def _positive(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


def _useful(slice_s: float, limit_s: float) -> bool:
    return slice_s >= min(_MIN_SLICE_S, limit_s)


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


def _listing(entries: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{entry['resource']} ({entry['chars']} chars)" for entry in entries
    )


def _response_schema(
    candidate_schema: dict[str, Any], batch_size: int
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": ["propose", "read", "diagnose", "stop"],
            },
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
        "required": ["action", "candidates", "resource", "offset"],
    }


def _author(
    config: KernelGenerator,
    prompt: str,
    schema: dict[str, Any],
    directory: Path,
    deadline: float,
    evidence: EvidenceCatalog,
    diagnose: Callable[[Path, float], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the author's proposals, or an empty batch when it stops.

    Evidence reads and diagnostics share ``deadline`` rather than receiving a
    fresh timeout on every turn.
    """
    for turn in range(config.max_context_requests + 1):
        timeout = deadline - time.monotonic()
        if not _useful(timeout, config.author_timeout_s):
            raise TimeoutError("Authoring time ran out; evaluation time is reserved")
        turn_dir = directory if turn == 0 else directory / f"read-{turn}"
        turn_dir.mkdir(exist_ok=True)
        turn_prompt = prompt + (
            f"\nAUTHOR BUDGET: {config.max_context_requests - turn} evidence reads "
            f"remaining; {int(timeout)} seconds. "
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
        action = reply.get("action")
        if action == "stop":
            return []
        if action == "propose":
            batch = reply.get("candidates")
            if (
                not isinstance(batch, list)
                or not 1 <= len(batch) <= config.search.batch_size
                or any(not isinstance(item, dict) for item in batch)
            ):
                raise ValueError(
                    "Proposal candidates must be a bounded batch of 1 to "
                    f"{config.search.batch_size} candidate objects"
                )
            return batch
        if action not in ("read", "diagnose"):
            raise ValueError(
                "Author response action must be propose, read, diagnose, or stop"
            )
        if turn == config.max_context_requests:
            raise ValueError("Evidence request limit reached; return a candidate")
        try:
            if action == "diagnose":
                if not config.diagnostics:
                    raise ValueError("Diagnostics are not enabled for this search")
                page = diagnose(turn_dir, deadline)
            else:
                resource, offset = reply.get("resource"), reply.get("offset")
                if not isinstance(resource, str):
                    raise ValueError("Evidence resource must be a catalog name")
                source = evidence
                if resource == "catalog":
                    # The listing grows during a search, so page a snapshot.
                    source = EvidenceCatalog()
                    source.add("catalog", _listing(evidence.catalog()))
                page = source.read(resource, 0 if offset is None else offset)
        except (ValueError, KeyError, TypeError) as error:
            page = {"error": str(error)}
        write_json(turn_dir / "evidence.json", {"request": reply, "result": page})
        prompt += "\nEVIDENCE RESULT:\n" + json.dumps(page)
    raise AssertionError("Author request loop did not return")


def _meets_objective(report: dict[str, Any], config: KernelGenerator) -> bool:
    if report.get("status") != "passed":
        return False
    if config.objective == "candidate":
        return True
    baseline = report.get("baseline_latency_ms")
    if not _positive(baseline):
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
    entries: list[dict[str, Any]],
    improved: bool,
    stale: int,
    config: KernelGenerator,
) -> str:
    """Choose the next experiment from the last round's outcomes, not agent verdicts.

    A round's evaluation order depends on batch and tuning interleaving, so the
    most severe outcome of the whole round decides.
    """
    limit = config.search.max_no_improvement
    if limit is not None and stale >= limit:
        return "finalize"
    stages = {
        entry["result"].get("stage")
        for entry in entries
        if entry["result"].get("status") == "error"
    }
    if stages & _TERMINAL_STAGES:
        return "finalize"
    if improved:
        return "optimize"
    if stages & {"source", "compile", "engine"}:
        return "repair_compile"
    if "correctness" in stages:
        return "repair_correctness"
    if stages:
        return "repair"
    return "rewrite" if entries else "author"


def _worker_failure(
    stage: str, summary: str, stderr: str, directory: Path
) -> dict[str, Any]:
    tail = scrub(stderr, directory).strip()[-3000:]
    return _error(stage, f"{summary}: {tail}" if tail else summary)


def _evaluate(
    request: dict[str, Any], directory: Path, timeout_s: float
) -> dict[str, Any]:
    request_path = directory / "evaluation.json"
    result_path = directory / "result.json"
    log_path = directory / "worker.log"
    write_json(request_path, request)
    result_path.unlink(missing_ok=True)
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
    except _providers.ProcessTimeoutError as error:
        # The partial output shows the author where the worker was stuck.
        log_path.write_text(
            strip_control(error.stdout + "\n" + error.stderr), encoding="utf-8"
        )
        return _worker_failure(
            "timeout",
            f"Evaluator exceeded its {timeout_s:g}s timeout",
            error.stderr,
            directory,
        )
    except OSError as error:
        return _error("setup", f"Evaluator could not start: {error}")
    log_path.write_text(
        strip_control(completed.stdout + "\n" + completed.stderr), encoding="utf-8"
    )
    try:
        report: dict[str, Any] = read_json_object(result_path)
    except (OSError, ValueError):
        report = {}
    if report.get("status") in ("passed", "error"):
        # The worker publishes its report before exiting; anything after that
        # cannot change the outcome.
        if report["status"] == "passed" and not _positive(report.get("latency_ms")):
            return _error("setup", "Evaluator returned an invalid timing")
        return report
    if completed.returncode < 0:
        # The worker reports Python failures itself, so a signal means native
        # code crashed, typically the candidate's kernel or plugin.
        return _worker_failure(
            "crash",
            f"Evaluator was killed by signal {-completed.returncode}",
            completed.stderr,
            directory,
        )
    if completed.returncode != 0:
        return _worker_failure(
            "setup",
            f"Evaluator exited {completed.returncode}",
            completed.stderr,
            directory,
        )
    return _error("setup", "Evaluator returned an invalid report")


def _cache_root(config: KernelGenerator) -> Path:
    if config.cache_dir is not None:
        return Path(config.cache_dir).resolve()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "torch_tensorrt" / "region_kernels").resolve()


def generate_registration(
    child: torch.fx.GraphModule,
    config: KernelGenerator,
    *,
    region_id: str,
    settings: CompilationSettings | None = None,
    memo: dict[str, Any] | None = None,
) -> KernelSpec | partial[None]:
    """Generate a kernel after capture and return the existing registration form.

    ``settings`` are the user's compilation settings; their engine options apply
    to evaluation engines. ``memo`` belongs to one compilation: identical regions
    in it reuse one validated kernel without another search or check.
    """
    registration, _ = _generate(
        child, config, region_id=region_id, settings=settings, memo=memo
    )
    return registration


def _generate(
    child: torch.fx.GraphModule,
    config: KernelGenerator,
    *,
    region_id: str,
    instruction_bundle: dict[str, Any] | None = None,
    settings: CompilationSettings | None = None,
    memo: dict[str, Any] | None = None,
) -> tuple[KernelSpec | partial[None], Path]:
    """Return a registration and the exact run containing its validation evidence."""
    cache: Path | None = None
    try:
        if settings is not None and settings.hardware_compatible:
            raise RegionError(
                f"Kernel generation for {region_id!r} compiles for the build GPU, so it "
                "cannot run in a hardware_compatible=True engine; compile without it "
                "or supply the kernel through fused_kernel(kernel=...)"
            )
        input_specs, output_specs = _boundaries(child, region_id)
        specs = input_specs + output_specs
        if any(
            spec.dtype not in (torch.float16, torch.float32, torch.bfloat16)
            for spec in specs
        ):
            raise RegionError(
                "Agent-generated regions support float16, float32, and bfloat16 boundaries"
            )
        if any(math.prod(spec.shape) == 0 for spec in specs):
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
        policy = asdict(config.search)
        policy["tuning"] = validate_tuning_space(config.backend, config.search.tuning)
        task = {
            "version": _VERSION,
            "reference": reference.code,
            "constants": constants,
            "inputs": [_describe(spec) for spec in input_specs],
            "outputs": [_describe(spec) for spec in output_specs],
            "backend": config.backend,
            "backend_version": backend.version,
            "instructions_sha256": instructions["sha256"],
            "proposal_schema_sha256": digest(backend.proposal_schema),
            "objective": config.objective,
            "min_speedup": config.min_speedup,
            "search": policy,
            "budget": {name: getattr(config, name) for name in _BUDGET},
            "diagnostics": config.diagnostics,
            "compile_options": _compile_options(settings),
            "analysis": {
                "operations": [
                    str(node.target)
                    for node in reference.graph.nodes
                    if node.op == "call_function"
                ],
                "boundary_bytes": sum(
                    math.prod(spec.shape) * spec.dtype.itemsize for spec in specs
                ),
            },
            "target": _target(backend, input_specs[0].device),
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
        if memo is not None and key in memo:
            candidate, run = memo[key]
            return (
                make_registration(candidate, backend, input_specs, output_specs, run),
                run,
            )
        cache = _cache_root(config) / key
        cache.mkdir(parents=True, exist_ok=True)
        # timeout_s separately bounds the wait for a concurrent generation of
        # this task and the search, whose budget starts once the lock is held,
        # so a waiter can still revalidate what its peer published. The worst
        # case wall time is therefore about 2 * timeout_s.
        with FileLock(str(cache / "generation.lock"), timeout=config.timeout_s):
            search = _Search(
                reference=reference,
                task=task,
                key=key,
                config=config,
                backend=backend,
                instructions=instructions,
                input_specs=input_specs,
                output_specs=output_specs,
                region_id=region_id,
                cache=cache,
            )
            candidate, registration = search.generate()
        if memo is not None:
            memo[key] = (candidate, search.run)
        return registration, search.run
    except RegionError:
        raise
    except Timeout as error:
        raise RegionError(
            f"Kernel generation for {region_id!r} waited {config.timeout_s:g}s for "
            f"another process generating the same kernel. Artifacts: {cache}"
        ) from error
    except Exception as error:
        artifacts = "" if cache is None else f". Artifacts: {cache}"
        raise RegionError(
            f"Kernel generation for {region_id!r} failed: {error}{artifacts}"
        ) from error


class _Search:
    """One locked generation run: cache reuse, resume, author rounds, selection."""

    def __init__(
        self,
        *,
        reference: torch.fx.GraphModule,
        task: dict[str, Any],
        key: str,
        config: KernelGenerator,
        backend: GenerationBackend,
        instructions: dict[str, Any],
        input_specs: list[Any],
        output_specs: list[Any],
        region_id: str,
        cache: Path,
    ) -> None:
        self.deadline = time.monotonic() + config.timeout_s
        self.task, self.key, self.config, self.policy = task, key, config, config.search
        self.backend, self.instructions = backend, instructions
        self.input_specs, self.output_specs = input_specs, output_specs
        self.region_id, self.cache = region_id, cache
        self.run = Path(tempfile.mkdtemp(prefix="run-", dir=cache))
        self.schema = _response_schema(backend.proposal_schema, self.policy.batch_size)
        reference_path = self.run / "reference.pt"
        torch.save(reference, reference_path)
        self.template = {
            "reference_path": str(reference_path),
            "backend": config.backend,
            "backend_version": task["backend_version"],
            "proposal_schema_sha256": task["proposal_schema_sha256"],
            "inputs": task["inputs"],
            "outputs": task["outputs"],
            "objective": config.objective,
            "diagnostics": config.diagnostics,
            **task["validation"],
            "validation_seed": secrets.randbits(63),
            "validation_phase": "search",
            "compile_options": task["compile_options"],
        }
        for name, value in (
            ("task.json", task),
            ("instructions.json", instructions),
            ("proposal.schema.json", self.schema),
            ("evaluation-template.json", self.template),
        ):
            write_json(self.run / name, value)
        self.evidence = EvidenceCatalog()
        for resource in instructions["resources"]:
            self.evidence.add(resource["name"], resource["text"])
        self.evidence.add("task.json", json.dumps(task, indent=2))
        # Final validation keeps a slice for every finalist, but never more than
        # a third of the budget.
        self.search_deadline = self.deadline - min(
            self.policy.finalists * config.candidate_timeout_s, config.timeout_s / 3
        )
        variants = min(
            self.policy.max_variants,
            math.prod(len(choices) for choices in task["search"]["tuning"].values()),
        )
        self.max_candidates = self.policy.max_candidates or (
            config.max_attempts * self.policy.batch_size * variants
        )
        self.gpu = gpu_lock(cache.parent, input_specs[0].device)
        self.history: list[dict[str, Any]] = []
        # The latest round's entries and whether it produced a new best.
        self.round: list[dict[str, Any]] = []
        self.improved = False
        # Reusable evaluations by candidate digest: passes and failures at
        # deterministic stages. Passing entries are the selection contenders.
        self.measured: dict[str, dict[str, Any]] = {}
        self.best_key: str | None = None
        self.pending: list[dict[str, Any]] = []
        self.used = 0
        self.stale = 0
        self.stop_reason = "attempt_limit"

    @property
    def best(self) -> dict[str, Any] | None:
        return None if self.best_key is None else self.measured.get(self.best_key)

    @property
    def incumbent(self) -> dict[str, Any] | None:
        best = self.best
        return None if best is None else best["candidate"]

    def generate(self) -> tuple[dict[str, Any], KernelSpec | partial[None]]:
        """Return the accepted candidate and its registration, or raise RegionError."""
        cached = self.from_cache()
        if cached is not None:
            return cached
        if self.policy.resume:
            self.resume()
        for attempt in range(self.config.max_attempts):
            if not self.author_round(attempt):
                break
        return self.finalize()

    def validate(self, proposal: Any) -> dict[str, Any]:
        return self.backend.validate_candidate(
            proposal, len(self.input_specs), len(self.output_specs)
        )

    def register(self, candidate: dict[str, Any]) -> KernelSpec | partial[None]:
        return make_registration(
            candidate, self.backend, self.input_specs, self.output_specs, self.run
        )

    def from_cache(self) -> tuple[dict[str, Any], KernelSpec | partial[None]] | None:
        manifest = self.cache / "accepted.json"
        if not self.config.use_cache or not manifest.is_file():
            return None
        try:
            stored = read_json_object(manifest)
            if stored["task_sha256"] != self.key or stored[
                "candidate_sha256"
            ] != digest(stored["candidate"]):
                raise ValueError("Cache identity or content digest mismatch")
            candidate = self.validate(stored["candidate"])
        except (OSError, KeyError, ValueError, TypeError) as error:
            _LOGGER.warning("Ignoring invalid agent region cache: %s", error)
            return None
        report = self.evaluate(candidate, self.run, self.deadline, holdout=True)
        write_json(self.run / "cache_check.json", report)
        if not _meets_objective(report, self.config):
            _LOGGER.warning("Agent region cache revalidation failed: %s", report)
            return None
        _LOGGER.info("Agent region %s: cache hit (%s)", self.region_id, self.cache)
        return candidate, self.register(candidate)

    def resume(self) -> None:
        """Remeasure retained candidates; a checkpoint never carries current proof."""
        checkpoint = self.cache / "checkpoint.json"
        if not checkpoint.is_file():
            return
        try:
            saved = read_json_object(checkpoint)
            state = saved["state"]
            if state["task_sha256"] != self.key or saved["sha256"] != digest(state):
                raise ValueError("Checkpoint identity or content digest mismatch")
            history = state["history"]
            if not isinstance(history, list) or any(
                not isinstance(entry, dict)
                or not isinstance(entry.get("result"), dict)
                or not (
                    entry.get("candidate") is None
                    or isinstance(entry["candidate"], dict)
                )
                for entry in history
            ):
                raise ValueError("Invalid checkpoint history")
            restored: dict[str, dict[str, Any]] = {}
            for item in [
                *state["finalists"][: self.policy.finalists],
                *state["pending"],
            ]:
                candidate = self.validate(item)
                restored.setdefault(digest(candidate), candidate)
        except (OSError, KeyError, ValueError, TypeError) as error:
            _LOGGER.warning("Ignoring invalid generation checkpoint: %s", error)
            return
        self.history.extend({**entry, "historical": True} for entry in history)
        self.pending.extend(restored.values())
        self.evidence.add("previous-history.json", json.dumps(self.history, indent=2))
        write_json(
            self.run / "resume.json",
            {"from": state["run"], "candidates": len(self.pending)},
        )
        directories = []
        for index in range(len(self.pending)):
            directory = self.run / f"resume-{index + 1}"
            directory.mkdir()
            directories.append(directory)
        self.save_checkpoint()
        self.run_pending(directories, "resume")

    def author_round(self, attempt: int) -> bool:
        """Request, expand, and measure one batch; return False to stop searching."""
        config = self.config
        now = time.monotonic()
        remaining = self.search_deadline - now
        # Keep evaluation time for this round's proposals.
        author_deadline = min(
            self.search_deadline - min(2 * config.candidate_timeout_s, remaining / 2),
            now + config.author_timeout_s,
        )
        shortened = author_deadline < now + config.author_timeout_s
        if self.used >= self.max_candidates or not _useful(
            author_deadline - now, config.author_timeout_s
        ):
            self.stop_reason = "budget"
            return False
        action = _next_action(self.round, self.improved, self.stale, config)
        if action == "finalize":
            self.stop_reason = "policy"
            return False
        directory = self.run / f"attempt-{attempt + 1}"
        directory.mkdir()
        prompt = self.prompt(action, remaining, attempt)
        self.round, self.improved = [], False
        _LOGGER.info(
            "Agent region %s: %s attempt %d/%d with %s (%s)",
            self.region_id,
            action,
            attempt + 1,
            config.max_attempts,
            config.provider,
            config.backend,
        )
        try:
            proposals = _author(
                config,
                prompt,
                self.schema,
                directory,
                author_deadline,
                self.evidence,
                self.diagnose,
            )
        except (OSError, ValueError, TypeError, TimeoutError, RuntimeError) as error:
            # A slower turn on a slice the deadline shortened cannot do better
            # in the even shorter slice that would follow.
            stage = (
                "provider"
                if isinstance(error, _providers.ProviderError)
                else (
                    "budget"
                    if shortened and isinstance(error, TimeoutError)
                    else "author"
                )
            )
            self.record(None, _error(stage, str(error)[-3000:]), directory, action)
            self.stale += 1
            self.save_checkpoint()
            if stage == "author":
                return True
            self.stop_reason = stage
            return False
        self.register_logs(directory)
        if not proposals:
            self.stop_reason = "author_stop"
            self.save_checkpoint()
            return False
        directories = self.enqueue(proposals, directory, action)
        self.save_checkpoint()
        self.run_pending(directories, "tune" if self.policy.tuning else action)
        self.stale = 0 if self.improved else self.stale + 1
        return True

    def enqueue(
        self, proposals: list[dict[str, Any]], directory: Path, action: str
    ) -> list[Path]:
        """Queue each valid variant and return the directories that will hold them."""
        groups = []
        failures = []
        for proposal in proposals:
            try:
                groups.append(
                    candidate_variants(
                        proposal,
                        self.config.backend,
                        self.task["search"]["tuning"],
                        self.policy.max_variants,
                    )
                )
            except (ValueError, TypeError) as error:
                failures.append(str(error))
        # Round-robin across sources so tuning the first proposal cannot consume
        # the entire batch before a different implementation receives a trial.
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
                candidate = self.validate(option)
            except (ValueError, TypeError) as error:
                self.record(
                    None,
                    _error("source", str(error)[-3000:]),
                    candidate_directory,
                    action,
                )
                continue
            self.pending.append(candidate)
            directories.append(candidate_directory)
        return directories

    def run_pending(self, directories: list[Path], action: str) -> None:
        for directory in directories:
            if not self.pending:
                break
            # Reuse costs no evaluation; otherwise evaluate() reports an
            # exhausted budget, which is recorded and keeps the rest pending.
            candidate = self.pending[0]
            previous = self.measured.get(digest(candidate))
            if previous is None:
                report = self.evaluate(
                    candidate,
                    directory,
                    self.search_deadline,
                    incumbent=self.incumbent,
                )
            else:
                report = {
                    **previous["report"],
                    "reused_from": previous["evaluation_path"],
                }
            self.record(candidate, report, directory, action)
            # A candidate the budget did not let finish stays pending for resume.
            interrupted = report.get("stage") == "budget"
            if not interrupted:
                self.pending.pop(0)
            self.save_checkpoint()
            if interrupted:
                self.stop_reason = "budget"
                break

    def evaluate(
        self,
        candidate: dict[str, Any],
        directory: Path,
        deadline: float,
        *,
        holdout: bool = False,
        incumbent: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Measure one candidate before ``deadline``; stage "budget" means not now."""
        limit = self.config.candidate_timeout_s
        if not holdout and self.used >= self.max_candidates:
            return _error(
                "budget", f"All {self.max_candidates} candidate evaluations are used"
            )
        remaining = deadline - time.monotonic()
        if not _useful(remaining, limit):
            return _error("budget", "Too little generation time remains to evaluate")
        try:
            with self.gpu.acquire(timeout=remaining):
                granted = min(limit, deadline - time.monotonic())
                if not _useful(granted, limit):
                    return _error(
                        "budget", "Generation time ran out waiting for the GPU"
                    )
                request = {**self.template, "candidate": candidate}
                if holdout:
                    request.update(
                        validation_seed=secrets.randbits(63),
                        validation_phase="holdout",
                    )
                else:
                    self.used += 1
                    if incumbent is not None:
                        request["incumbent"] = incumbent
                report = _evaluate(request, directory, granted)
        except Timeout:
            return _error("budget", "GPU evaluation lock timed out")
        if report.get("stage") == "timeout" and granted < limit:
            # The deadline, not the candidate's own time limit, ended this run.
            return _error(
                "budget",
                f"Generation time ran out {granted:.3g}s into this evaluation",
            )
        return report

    def record(
        self,
        candidate: dict[str, Any] | None,
        report: dict[str, Any],
        directory: Path,
        action: str,
    ) -> None:
        report = {**report, "meets_objective": _meets_objective(report, self.config)}
        feedback = directory / "feedback.json"
        write_json(feedback, report)
        entry = {"candidate": candidate, "result": report, "action": action}
        self.history.append(entry)
        self.round.append(entry)
        prefix = directory.relative_to(self.run).as_posix()
        self.evidence.add(f"{prefix}/feedback.json", json.dumps(report, indent=2))
        if candidate is not None:
            write_json(directory / "candidate.json", candidate)
            self.evidence.add(
                f"{prefix}/candidate.json", json.dumps(candidate, indent=2)
            )
            if isinstance(candidate.get("source"), str):
                self.evidence.add(f"{prefix}/source", candidate["source"])
            if "reused_from" not in report:
                self.learn(candidate, report, feedback)
        self.register_logs(directory)
        write_json(self.run / "history.json", self.history)
        _LOGGER.info("Agent region %s: %s", self.region_id, report)

    def learn(
        self, candidate: dict[str, Any], report: dict[str, Any], feedback: Path
    ) -> None:
        # With "incumbent_error", only the incumbent's timing arm failed: both keep
        # their own measurements, and _improves compares them unpaired.
        passed = report.get("status") == "passed"
        if not passed and report.get("stage") not in _DETERMINISTIC_STAGES:
            return
        key = digest(candidate)
        self.measured[key] = {
            "candidate": candidate,
            "report": report,
            "evaluation_path": feedback.relative_to(self.run).as_posix(),
        }
        best = self.best
        if passed and _improves(report, None if best is None else best["report"]):
            self.best_key = key
            self.improved = True

    def ranked(self) -> list[dict[str, Any]]:
        """Passing candidates: the paired-comparison winner, then by latency."""
        best = self.best
        others = sorted(
            (
                entry
                for entry in self.measured.values()
                if entry["report"].get("status") == "passed" and entry is not best
            ),
            key=lambda entry: entry["report"]["latency_ms"],
        )
        return ([best] if best is not None else []) + others

    def save_checkpoint(self) -> None:
        state = {
            "task_sha256": self.key,
            "run": str(self.run),
            "history": _prompt_history(self.history, self.incumbent),
            "finalists": [
                entry["candidate"] for entry in self.ranked()[: self.policy.finalists]
            ],
            "pending": self.pending,
        }
        write_json(
            self.cache / "checkpoint.json", {"state": state, "sha256": digest(state)}
        )

    def register_logs(self, directory: Path) -> None:
        for log in (
            directory / "worker.log",
            directory / "provider.log",
            *sorted(directory.glob("read-*/provider.log")),
        ):
            if log.is_file():
                self.evidence.add(
                    log.relative_to(self.run).as_posix(),
                    scrub(
                        log.read_text(encoding="utf-8", errors="replace"),
                        self.run,
                        "<run>",
                    ),
                )

    def diagnose(self, directory: Path, until: float) -> dict[str, Any]:
        incumbent = self.incumbent
        if incumbent is None:
            raise ValueError("Diagnostics require a current verified candidate")
        if self.max_candidates - self.used < 2:
            raise ValueError(
                "Diagnostics would use the last candidate evaluation; propose instead"
            )
        diagnostics = directory / "diagnostics"
        diagnostics.mkdir()
        report = self.evaluate(incumbent, diagnostics, until)
        write_json(diagnostics / "feedback.json", report)
        self.evidence.add(
            (diagnostics / "feedback.json").relative_to(self.run).as_posix(),
            json.dumps(report, indent=2),
        )
        return report

    def prompt(self, action: str, remaining_s: float, attempt: int) -> str:
        catalog = self.evidence.catalog()
        inlined = {resource["name"] for resource in self.instructions["resources"]}
        recent = [entry for entry in catalog if entry["resource"] not in inlined]
        return (
            str(self.instructions["text"])
            + _PROTOCOL.format(batch_size=self.policy.batch_size)
            + f"\nSEARCH STEP: {action}. Remaining search seconds: "
            f"{max(0, int(remaining_s))}; candidate evaluations: "
            f"{self.max_candidates - self.used}; author rounds including this one: "
            f"{self.config.max_attempts - attempt}."
            + "\nTASK:\n"
            + json.dumps(self.task, indent=2)
            + "\nRECENT ATTEMPTS AND BEST CANDIDATE:\n"
            + json.dumps(_prompt_history(self.history, self.incumbent), indent=2)
            + f"\nEVIDENCE: {len(catalog)} readable resources, including the "
            "instruction resources above. The newest follow; read resource "
            '"catalog" for the full listing.\n' + _listing(recent[-_PROMPT_EVIDENCE:])
        )

    def finalize(self) -> tuple[dict[str, Any], KernelSpec | partial[None]]:
        """Recheck finalists on fresh inputs in rank order, then publish one."""
        write_json(self.run / "history.json", self.history)
        write_json(
            self.run / "search.json",
            {
                "stop_reason": self.stop_reason,
                "evaluations": self.used,
                "max_candidates": self.max_candidates,
                "stale_rounds": self.stale,
            },
        )
        self.save_checkpoint()
        choices = [
            entry for entry in self.ranked() if entry["report"]["meets_objective"]
        ][: self.policy.finalists]
        if not choices:
            raise RegionError(self.failure())
        report: dict[str, Any] = {}
        for index, entry in enumerate(choices):
            directory = self.run / ("holdout" if index == 0 else f"holdout-{index + 1}")
            directory.mkdir()
            # Each finalist gets a full slice of what remains, so alternatives
            # never shorten the winner's check.
            report = self.evaluate(
                entry["candidate"], directory, self.deadline, holdout=True
            )
            write_json(directory / "feedback.json", report)
            if _meets_objective(report, self.config):
                return entry["candidate"], self.accept(entry, report)
            self.history.append(
                {"candidate": entry["candidate"], "result": report, "action": "holdout"}
            )
            write_json(self.run / "history.json", self.history)
            if report.get("status") == "passed" or report.get("stage") in (
                "correctness",
                "crash",
            ):
                # Fresh inputs disproved this finalist. An inconclusive check,
                # such as a timeout, keeps it in the checkpoint for resume.
                self.measured.pop(digest(entry["candidate"]), None)
            self.save_checkpoint()
        raise RegionError(
            f"Generated kernel for region {self.region_id!r} failed final validation: "
            f"{report.get('error', 'Performance objective not met')}. Artifacts: {self.run}"
        )

    def accept(
        self, entry: dict[str, Any], validation: dict[str, Any]
    ) -> KernelSpec | partial[None]:
        # Construct the registration before publishing a cache entry.
        registration = self.register(entry["candidate"])
        accepted = {
            "task_sha256": self.key,
            "candidate_sha256": digest(entry["candidate"]),
            "candidate": entry["candidate"],
            "report": entry["report"],
            "validation": validation,
            "evaluation_path": entry["evaluation_path"],
            "provider": self.config.provider,
            "model": self.config.model,
            "instructions_sha256": self.instructions["sha256"],
            "run": str(self.run),
        }
        write_json(self.run / "accepted.json", accepted)
        if self.config.use_cache:
            write_json(self.cache / "accepted.json", accepted)
        (self.cache / "checkpoint.json").unlink(missing_ok=True)
        _LOGGER.info(
            "Agent region %s: selected kernel; artifacts: %s", self.region_id, self.run
        )
        return registration

    def failure(self) -> str:
        """Explain why this invocation found no kernel, ignoring resumed history."""
        results = [
            entry["result"] for entry in self.history if not entry.get("historical")
        ]
        limit = self.policy.max_no_improvement
        reason = {
            "attempt_limit": f"all {self.config.max_attempts} author rounds were used",
            "budget": (
                "all candidate evaluations were used"
                if self.used >= self.max_candidates
                else "the search time ran out"
            ),
            "policy": (
                f"{limit} author rounds in a row brought no improvement"
                if limit is not None and self.stale >= limit
                else "an evaluation failed in a way no proposal can fix"
            ),
            "author_stop": "the author stopped",
            "provider": "the provider failed",
        }[self.stop_reason]
        if any(result.get("status") == "passed" for result in results):
            detail = "no passing candidate met the performance objective"
        elif results:
            detail = f"last failure: {results[-1].get('error', 'unknown error')}"
        else:
            detail = "no candidate was evaluated"
        return (
            f"Agent generated no valid kernel for region {self.region_id!r}: "
            f"{reason} ({self.used} of {self.max_candidates} evaluations used); "
            f"{detail}. Artifacts: {self.run}"
        )
