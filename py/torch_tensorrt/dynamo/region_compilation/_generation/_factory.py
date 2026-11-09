# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded author/evaluator loop, persisted evidence, and kernel cache."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import secrets
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch
from filelock import FileLock, Timeout

from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.kernels import KernelSpec
from torch_tensorrt.region import KernelGenerator, RegionError

from .._capture import _SAMPLING_SWITCHES, _draws_random_numbers
from .._plugins import _boundaries
from . import _providers
from ._backends import (
    GenerationBackend,
    candidate_variants,
    get_backend,
    validate_tuning_space,
    with_options,
)
from ._instructions import EvidenceCatalog, load_instructions, validate_instructions
from ._reference import FILENAME as REFERENCE_FILENAME
from ._reference import encode_reference, save_reference
from ._samples import FILENAME as SAMPLES_FILENAME
from ._samples import (
    Coverage,
    check_samples,
    coverage,
    covers,
    merge_coverage,
    samples_digest,
    save_samples,
)
from ._util import (
    digest,
    gpu_lock,
    private_directory,
    read_json_object,
    scrub,
    write_json,
)

if TYPE_CHECKING:
    from torch_tensorrt.dynamo._settings import CompilationSettings

_LOGGER = logging.getLogger(__name__)
_VERSION = 5
# "0" forbids generation for every KernelGenerator, as allow_generation=False
# does for one, so CI can use only cached kernels without code changes.
_GENERATION_ENV = "TORCH_TENSORRT_REGION_GENERATION"
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
    "max_protocol_retries",
    "use_cache",
)
# No proposal can fix these, so the search stops and validates what it has.
_TERMINAL_STAGES = frozenset({"provider", "setup", "baseline", "budget"})
# These reproduce for an unchanged candidate on the run's fixed search inputs.
_DETERMINISTIC_STAGES = frozenset({"source", "compile", "engine", "correctness"})
# A worker reaches these only after building the TensorRT baseline.
_AFTER_BASELINE = frozenset(
    {"compile", "resources", "engine", "correctness", "benchmark"}
)
# A revalidation that fails here has not shown that the cached kernel is wrong.
_INCONCLUSIVE_STAGES = frozenset(
    {"budget", "setup", "timeout", "baseline", "resources"}
)
# Seconds to wait before evaluating again after a setup failure, which device
# memory that other work briefly held can cause.
_SETUP_RETRY_WAIT_S = 2.0
# Waits before asking a temporarily unavailable model API again, such as after
# a rate limit (429) or an overloaded (529) or failing (5xx) server.
_UNAVAILABLE_BACKOFF_S = (5.0, 20.0, 60.0)
# Rejected author output kept in the history is cut to this many characters.
_REJECTED_CHARS = 131072
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
    # cached candidate must satisfy the current user's constraints. Validation
    # samples do not: they depend on model weights and on upstream numerics. A
    # kernel from the persistent cache is checked on the current samples before
    # every reuse. Within one compilation, an identical region reuses the shared
    # kernel unchecked only when it has no samples, when the kernel already
    # passed samples with the same digest, or when its samples come from the
    # example inputs and covers() finds every input within what earlier checks
    # covered; otherwise it is checked on its own samples (see _generate).
    identity = {
        key: value
        for key, value in task.items()
        if key not in ("search", "budget", "diagnostics", "samples")
    }
    # Tensor factories in the reference name the region's logical device.
    identity["reference"] = re.sub(
        r"device\(type='cuda', index=\d+\)", "device(type='cuda')", task["reference"]
    )
    identity["tuning"] = task["search"]["tuning"]
    for side in ("inputs", "outputs"):
        identity[side] = [
            {**spec, "device": spec["device"].partition(":")[0]} for spec in task[side]
        ]
    return digest(identity)


@dataclass
class _Shared:
    """The kernel that identical regions of one compilation share."""

    candidate: dict[str, Any]
    # The run whose copy of the kernel source these regions load.
    run: Path
    # Digests of the regions' samples that the kernel passed, and those
    # samples, which a new search for an identical region must also pass.
    checked: set[str]
    samples: list[tuple[torch.Tensor, ...]]
    # The ids of the regions that use it, which final_registration rebuilds.
    regions: set[str] = field(default_factory=set)
    # Each input's values that the kernel's checks on those samples covered,
    # the union of the samples' own ranges (see coverage); None without samples.
    covered: list[Coverage] | None = None


def _random_operation(target: torch._ops.OpOverload) -> str:
    """Explain why a reference cannot call ``target`` as captured."""
    from torch_tensorrt.dynamo.lowering._decompositions import (
        ATTENTION_DECOMPOSITION_OPS,
    )

    switches = _SAMPLING_SWITCHES.get(target.overloadpacket)
    if switches is None:
        reason = f"PyTorch tags {target} as random"
    else:
        reason = (
            f"{target} draws them here; it draws none when its "
            f"{' or '.join(switches)} argument is 0 or False"
        )
    message = (
        "Agent generation requires a reference that draws no random numbers, "
        f"but {reason}"
    )
    if target in ATTENTION_DECOMPOSITION_OPS:
        message += (
            ". With decompose_attention=True, compilation captures attention "
            "as matrix multiplications and a softmax instead"
        )
    return message


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
                    # As in region capture: PyTorch tags attention, dropout
                    # and RNNs as random even when they draw no random numbers.
                    if _draws_random_numbers(node):
                        raise RegionError(_random_operation(target))
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
    try:
        # Runs save the reference as data, never as a pickle; fail before any author call.
        encode_reference(reference)
    except TypeError as error:
        raise RegionError(
            f"Agent generation cannot save the reference: {error}"
        ) from None
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


def _codegen(config: KernelGenerator) -> dict[str, Any]:
    """Options that change the code a backend generates for a candidate.

    They are part of the cache key, and both the controller's and the
    worker's registrations apply them.
    """
    return {"aligned_io": config.aligned_io}


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
    candidate_schema: dict[str, Any], batch_size: int, propose_only: bool = False
) -> dict[str, Any]:
    """The author's reply schema; ``anyOf`` keeps it valid in strict validators.

    A propose-only turn, with no reads left or repairing a malformed reply,
    requires a nonempty ``candidates`` array, which ``stop`` ignores: a
    nullable field let replies drop the kernel they had written.
    """
    candidates = {
        "type": "array",
        "items": candidate_schema,
        "minItems": 1,
        "maxItems": batch_size,
    }
    properties: dict[str, Any] = {
        "action": {
            "type": "string",
            "enum": ["propose", "read", "diagnose", "stop"],
        },
        "candidates": {"anyOf": [candidates, {"type": "null"}]},
        "resource": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "offset": {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]},
    }
    if propose_only:
        properties.update(
            action={"type": "string", "enum": ["propose", "stop"]},
            candidates=candidates,
            resource={"type": "null"},
            offset={"type": "null"},
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": ["action", "candidates", "resource", "offset"],
    }


class _AuthorTimeout(TimeoutError):
    """An author turn ran out of time; ``shortened`` if the search deadline cut it."""

    def __init__(self, message: str, shortened: bool) -> None:
        super().__init__(message)
        self.shortened = shortened


def _batch(reply: dict[str, Any], batch_size: int) -> list[dict[str, Any]]:
    """Return a proposing reply's candidates, naming any defect of the batch."""
    batch = reply.get("candidates")
    if (
        isinstance(batch, list)
        and 1 <= len(batch) <= batch_size
        and all(isinstance(item, dict) for item in batch)
    ):
        return batch
    if "candidates" not in reply:
        defect = "candidates is missing"
    elif batch is None:
        defect = "candidates is null"
    elif not isinstance(batch, list):
        defect = f"candidates is a {type(batch).__name__}, not an array"
    elif not batch:
        defect = "candidates is empty"
    elif len(batch) > batch_size:
        defect = f"candidates holds {len(batch)} proposals"
    else:
        index = next(i for i, item in enumerate(batch) if not isinstance(item, dict))
        kind = type(batch[index]).__name__
        defect = f"candidates[{index}] is a {kind}, not an object"
    raise _providers.ProtocolError(
        f"Proposal candidates must be a bounded batch of 1 to {batch_size} "
        f"candidate objects, but {defect}"
    )


def _bounded(value: Any) -> Any:
    """A JSON copy of rejected author output, its long strings cut short."""
    if isinstance(value, str):
        if len(value) <= _REJECTED_CHARS:
            return value
        return value[:_REJECTED_CHARS] + "\n[truncated]"
    if isinstance(value, dict):
        return {str(key): _bounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bounded(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value if not isinstance(value, float) or math.isfinite(value) else None
    return _bounded(repr(value))


def _author(
    config: KernelGenerator,
    prompt: str,
    schemas: tuple[dict[str, Any], dict[str, Any]],
    directory: Path,
    limit: float,
    evidence: EvidenceCatalog,
    diagnose: Callable[[Path, float], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the author's proposals, or an empty batch when it stops.

    Every turn has its own ``author_timeout_s``, ending by ``limit``, which
    keeps time to evaluate the round's proposals: the first request, each
    evidence read or diagnosis, each retry after a reply that broke the
    protocol, and each retry after the model API was temporarily unavailable.
    ``schemas`` are the full reply schema and the propose-only one that turns
    without reads left and protocol retries use; a read or diagnosis requested
    on such a turn breaks the protocol.
    """
    reads = retries = waits = 0
    turn_dir = directory
    while True:
        now = time.monotonic()
        deadline = min(limit, now + config.author_timeout_s)
        timeout = deadline - now
        shortened = deadline < now + config.author_timeout_s
        if not _useful(timeout, config.author_timeout_s):
            raise _AuthorTimeout(
                "Authoring time ran out; evaluation time is reserved", True
            )
        turn_dir.mkdir(exist_ok=True)
        propose_only = retries > 0 or reads == config.max_context_requests
        turn_prompt = prompt + (
            f"\nAUTHOR BUDGET: {config.max_context_requests - reads} evidence reads "
            f"remaining; {int(timeout)} seconds for this turn. Each read or "
            "diagnosis gets a new turn. "
            + (
                "This turn must return action=propose, with candidates holding "
                "the proposals, or action=stop."
                if propose_only
                else "When no reads remain, return action=propose or action=stop."
            )
        )
        (turn_dir / "prompt.txt").write_text(turn_prompt, encoding="utf-8")
        reply: Any = None
        try:
            reply = _providers.propose(
                provider=config.provider,
                model=config.model,
                prompt=turn_prompt,
                workdir=turn_dir,
                timeout_s=timeout,
                schema=schemas[1] if propose_only else schemas[0],
                evidence=(
                    {"catalog": evidence.catalog()} if ":" in config.provider else None
                ),
            )
            action = reply.get("action")
            if action == "stop":
                return []
            if action == "propose":
                return _batch(reply, config.search.batch_size)
            if action not in ("read", "diagnose"):
                raise _providers.ProtocolError(
                    "Author response action must be propose, read, diagnose, or "
                    f"stop, but it is {action!r}"
                )
            if propose_only:
                # The turn's schema allowed only propose and stop.
                turn = (
                    "when no evidence reads remain"
                    if reads == config.max_context_requests
                    else "on a protocol retry"
                )
                raise _providers.ProtocolError(
                    f"Author response action must be propose or stop {turn}, "
                    f"but it is {action!r}"
                )
        except _providers.ProtocolError as error:
            # The reply is kept so the next round can show it to the author.
            error.reply = reply  # type: ignore[attr-defined]
            if retries >= config.max_protocol_retries:
                raise
            retries += 1
            _LOGGER.info("Author reply broke the protocol, asking again: %s", error)
            turn_dir = directory / f"retry-{retries}"
            prompt += (
                f"\nPROTOCOL ERROR: {error}. That reply was not used. Reply with "
                "action=propose and candidates holding 1 to "
                f"{config.search.batch_size} proposals that follow the backend "
                "schema, or with action=stop."
            )
            continue
        except _providers.ProviderUnavailableError as error:
            wait = _UNAVAILABLE_BACKOFF_S[min(waits, len(_UNAVAILABLE_BACKOFF_S) - 1)]
            if waits >= len(_UNAVAILABLE_BACKOFF_S) or not _useful(
                limit - time.monotonic() - wait, config.author_timeout_s
            ):
                raise
            waits += 1
            _LOGGER.warning("%s; asking again in %gs", error, wait)
            time.sleep(wait)
            turn_dir = directory / f"unavailable-{waits}"
            continue
        except TimeoutError as error:
            raise _AuthorTimeout(str(error), shortened) from error
        reads += 1
        try:
            if action == "diagnose":
                if not config.diagnostics:
                    raise ValueError("Diagnostics are not enabled for this search")
                page = diagnose(turn_dir, limit)
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
        turn_dir = directory / f"read-{reads}"


def _meets_objective(report: dict[str, Any], config: KernelGenerator) -> bool | None:
    """Whether a report meets the objective; None when only an engine can tell.

    A launch screen builds no engine, so it cannot decide a speedup over the
    PyTorch or TensorRT baseline; final validation measures that in an engine.
    """
    if report.get("status") != "passed":
        return False
    if config.objective == "candidate":
        return True
    if report.get("execution") == "launch":
        return None
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


def _speedup_shortfall(report: dict[str, Any], config: KernelGenerator) -> str | None:
    """Give the measured speedup of a correct kernel that misses the objective.

    None when the report lacks the baseline or the kernel latency.
    """
    baseline, latency = report.get("baseline_latency_ms"), report.get("latency_ms")
    if not (_positive(baseline) and _positive(latency)):
        return None
    measurements = report.get("benchmark", {})
    baseline_noise = measurements.get("baseline", {}).get("mad_ms", 0)
    latency_noise = measurements.get("candidate", {}).get("mad_ms", 0)
    # The ratio that _meets_objective compares with min_speedup.
    speedup = (baseline - baseline_noise) / (latency + latency_noise)
    return (
        f"speedup over the {config.objective} baseline after allowing for timing "
        f"noise, {speedup:.3g}x (baseline {baseline:.4g} ± {baseline_noise:.2g} ms, "
        f"kernel {latency:.4g} ± {latency_noise:.2g} ms), is below "
        f"min_speedup={config.min_speedup:g}"
    )


def _slower_than_eager(report: dict[str, Any], config: KernelGenerator) -> str | None:
    """Why an accepted kernel deserves a warning, if it does.

    ``objective="candidate"`` accepts the fastest correct kernel even when it
    is slower than the eager PyTorch region it replaces.
    """
    latency: Any = report.get("latency_ms")
    eager: Any = report.get("reference_latency_ms")
    if (
        config.objective != "candidate"
        or not (_positive(latency) and _positive(eager))
        or latency <= eager
    ):
        return None
    return (
        f"the selected kernel is slower than eager PyTorch on the validation "
        f"inputs ({latency:.3g} ms vs {eager:.3g} ms); objective='candidate' "
        "accepts the fastest correct kernel, objective='eager' requires a speedup"
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
    # Redacted whole, so the tail cannot keep part of a credential.
    tail = scrub(_providers._redact(stderr), directory).strip()[-3000:]
    return _error(stage, f"{summary}: {tail}" if tail else summary)


def _redacted(value: Any) -> Any:
    """A worker report without credentials, as provider logs are redacted.

    The fitted candidate is the author's own source with new constants, which
    the search adopts only if its source is unchanged.
    """
    if isinstance(value, str):
        return _providers._redact(value)
    if isinstance(value, list):
        return [_redacted(item) for item in value]
    if isinstance(value, dict):
        return {
            key: item if key == "fitted_candidate" else _redacted(item)
            for key, item in value.items()
        }
    return value


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
        # Keep the candidate directory, the working directory, off sys.path.
        "-P",
        "-m",
        "torch_tensorrt.dynamo.region_compilation._generation._worker",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
        # The worker ends itself if this process dies without killing it.
        "--timeout",
        str(timeout_s),
        "--parent",
        str(os.getpid()),
    ]
    try:
        completed = _providers.run_process(
            command, input="", cwd=directory, timeout_s=timeout_s
        )
    except _providers.ProcessTimeoutError as error:
        # The partial output shows the author where the worker was stuck.
        # Candidates can print anything, and the author can read this log.
        log_path.write_text(
            _providers._redact(error.stdout + "\n" + error.stderr), encoding="utf-8"
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
        _providers._redact(completed.stdout + "\n" + completed.stderr),
        encoding="utf-8",
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
        # Its messages quote compiler and candidate output to the author.
        redacted: dict[str, Any] = _redacted(report)
        return redacted
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


def _generation_block(config: KernelGenerator) -> str | None:
    """Name the setting that forbids a search, or None when one may run."""
    switch = os.environ.get(_GENERATION_ENV, "1")
    # Reject other spellings, so that a build meant to be cache-only never
    # calls an agent because of a typo.
    if switch not in ("0", "1"):
        raise RegionError(f"{_GENERATION_ENV} must be 0 or 1; got {switch!r}")
    if switch == "0":
        return f"{_GENERATION_ENV}=0"
    return None if config.allow_generation else "allow_generation=False"


def _cache_root(config: KernelGenerator) -> Path:
    if config.cache_dir is not None:
        return Path(config.cache_dir).resolve()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "torch_tensorrt" / "region_kernels").resolve()


def _contract(
    child: torch.fx.GraphModule,
    config: KernelGenerator,
    *,
    region_id: str,
    settings: CompilationSettings | None,
    instruction_bundle: dict[str, Any] | None = None,
) -> tuple[
    str | None,
    list[Any],
    list[Any],
    torch.fx.GraphModule,
    dict[str, Any],
    GenerationBackend,
    dict[str, Any],
    dict[str, Any],
    dict[str, list[Any]],
]:
    """Check what a search requires and no agent can change.

    Return the setting that forbids a search, if any, the boundary specs, the
    reference and its constants, the backend, the target it reports, the
    instructions (``instruction_bundle``, a replayed run's, when given) and
    the tuning choices.
    """
    blocked = _generation_block(config)
    if blocked is not None and not config.use_cache:
        raise RegionError(
            f"Kernel generation for {region_id!r} is disabled by {blocked}, "
            "and use_cache=False leaves no cached kernel to use"
        )
    if settings is not None and settings.hardware_compatible:
        raise RegionError(
            f"Kernel generation for {region_id!r} compiles for the build GPU, so it "
            "cannot run in a hardware_compatible=True engine; compile without it "
            "or supply the kernel through fused_kernel(kernel=...)"
        )
    if not ENABLED_FEATURES.qdp_plugin:
        # Trial engines and the compiled engine run the kernel as a plugin.
        raise RegionError(
            f"Kernel generation for {region_id!r} embeds its kernel in the "
            "TensorRT engine as a Quick Deployable Plugin (QDP), which this "
            "TensorRT build does not support; QDP plugins require TensorRT 10.7 "
            "or later, other than 10.14.x, and are unavailable with TensorRT-RTX"
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
    # KernelGenerator checks only that its instruction files exist, and only a
    # built-in backend's tuning choices.
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
    tuning = validate_tuning_space(config.backend, config.search.tuning)
    target = _target(backend, input_specs[0].device)
    root = _cache_root(config)
    if blocked is not None and not root.is_dir():
        # Creating a mistyped or unrestored cache_dir would only hide why
        # no kernel was found there.
        raise RegionError(
            f"Kernel generation for {region_id!r} is disabled by {blocked}, "
            f"and the kernel cache directory {root} does not exist; check "
            "cache_dir, or compile once with generation allowed to fill it"
        )
    return (
        blocked,
        input_specs,
        output_specs,
        reference,
        constants,
        backend,
        target,
        instructions,
        tuning,
    )


def check_generation(
    child: torch.fx.GraphModule,
    config: KernelGenerator,
    *,
    region_id: str,
    settings: CompilationSettings | None = None,
) -> None:
    """Raise the error that would end a search for this region before it starts.

    materialize_regions checks every generated region before the first search,
    so that an invalid region fails at once rather than after the searches of
    the regions before it; a dry run logs these errors instead. They name the
    region.
    """
    try:
        _contract(child, config, region_id=region_id, settings=settings)
    except RegionError as error:
        if repr(region_id) in str(error):
            raise
        # Keep the cause, such as the compiler error that failed a toolchain probe.
        raise RegionError(
            f"Kernel generation for {region_id!r}: {error}"
        ) from error.__cause__
    except Exception as error:
        raise RegionError(
            f"Kernel generation for {region_id!r} failed: {error}"
        ) from error


def generate_registration(
    child: torch.fx.GraphModule,
    config: KernelGenerator,
    *,
    region_id: str,
    settings: CompilationSettings | None = None,
    memo: dict[str, Any] | None = None,
    samples: Sequence[tuple[torch.Tensor, ...]] | None = None,
    implicit_samples: bool = False,
) -> KernelSpec | partial[None]:
    """Generate a kernel after capture and return the existing registration form.

    ``settings`` are the user's compilation settings; their engine options apply
    to evaluation engines. ``memo`` belongs to one compilation: identical regions
    in it share one validated kernel, which a region reuses after a correctness
    check on its samples if they are new; a region they fail searches again,
    also on the samples of the identical regions before it, and replaces the
    shared kernel. ``samples`` are region inputs captured from the generator's
    ``validation_inputs``; every evaluation also checks them.
    ``implicit_samples`` marks samples taken from the compilation's example
    inputs rather than ``validation_inputs``: they validate the first of a group
    of identical regions, and a later one reuses its kernel without a recheck
    when the kernel's checks so far covered each of its inputs (see covers):
    the input's finite values lie within the finite range of that input in one
    sample the kernel passed, or within overlapping such ranges but never in a
    gap between them; ranges of integer-valued samples cover integers only;
    and an input with at most two distinct finite values, such as a mask,
    equals one that was checked. The magnitude cases do not widen the ranges.
    Otherwise its samples are checked as new ones are, and once the kernel
    passes them their ranges are covered too.
    """
    registration, _ = _generate(
        child,
        config,
        region_id=region_id,
        settings=settings,
        memo=memo,
        samples=samples,
        implicit_samples=implicit_samples,
    )
    return registration


def final_registration(
    child: torch.fx.GraphModule,
    config: KernelGenerator,
    *,
    region_id: str,
    memo: dict[str, Any],
) -> KernelSpec | partial[None] | None:
    """Rebuild a generated region's registration from its group's final kernel.

    A later identical region's search may have replaced the kernel this region
    was generated with by one that also passed this region's samples. The
    lookup uses the task each region had when it was generated, so it never
    searches, whatever changed since.
    """
    for shared in memo.values():
        if region_id in shared.regions:
            input_specs, output_specs = _boundaries(child, region_id)
            return make_registration(
                shared.candidate,
                with_options(get_backend(config.backend), _codegen(config)),
                input_specs,
                output_specs,
                shared.run,
            )
    return None


def _generate(
    child: torch.fx.GraphModule,
    config: KernelGenerator,
    *,
    region_id: str,
    instruction_bundle: dict[str, Any] | None = None,
    settings: CompilationSettings | None = None,
    memo: dict[str, Any] | None = None,
    samples: Sequence[tuple[torch.Tensor, ...]] | None = None,
    implicit_samples: bool = False,
) -> tuple[KernelSpec | partial[None], Path]:
    """Return a registration and the exact run containing its validation evidence."""
    cache: Path | None = None
    try:
        (
            blocked,
            input_specs,
            output_specs,
            reference,
            constants,
            backend,
            target,
            instructions,
            tuning,
        ) = _contract(
            child,
            config,
            region_id=region_id,
            settings=settings,
            instruction_bundle=instruction_bundle,
        )
        specs = input_specs + output_specs
        if samples is not None:
            try:
                check_samples(samples, input_specs)
            except ValueError as error:
                raise RegionError(
                    f"KernelGenerator validation_inputs do not fit region "
                    f"{region_id!r}: {error}"
                ) from None
        policy = asdict(config.search)
        policy["tuning"] = tuning
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
            "target": target,
            "codegen": _codegen(config),
            "validation": {
                "num_tests": config.num_tests,
                "benchmark_iters": config.benchmark_iters,
                "benchmark_repeats": config.benchmark_repeats,
                "check_nonfinite": config.check_nonfinite,
                "rtol": config.rtol,
                "atol": config.atol,
                "fp32_matmul_precision": config.fp32_matmul_precision,
            },
        }
        sampled = None
        if samples:
            # Replay checks a run's samples.pt against this record. The cache
            # key leaves it out, so that new weights or nondeterministic
            # upstream values reuse a kernel after a check on these samples.
            sampled = samples_digest(samples)
            task["samples"] = {"count": len(samples), "sha256": sampled}
        key = _task_digest(task)
        backend = with_options(backend, task["codegen"])
        shared = None if memo is None else memo.get(key)
        if shared is not None and (
            sampled is None
            or sampled in shared.checked
            or (implicit_samples and covers(shared.covered, samples or ()))
        ):
            shared.regions.add(region_id)
            return (
                make_registration(
                    shared.candidate, backend, input_specs, output_specs, shared.run
                ),
                shared.run,
            )
        root = _cache_root(config)
        cache = private_directory(root) / key
        private_directory(cache)
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
                samples=samples,
                blocked=blocked,
                shared=shared,
            )
            candidate, registration = search.generate()
        if memo is not None:
            # The kernel passed this region's samples. It is the shared one, or
            # a search also checked it on every identical region's samples.
            previous = shared or _Shared(candidate, search.run, set(), [])
            kept = candidate == previous.candidate
            everything = [*previous.samples, *(samples or ())]
            memo[key] = _Shared(
                candidate=candidate,
                run=previous.run if kept else search.run,
                checked=previous.checked | ({sampled} if sampled else set()),
                samples=everything,
                regions=previous.regions | {region_id},
                # A kept kernel was checked on this region's samples alone;
                # a new one on all of them at once.
                covered=(
                    merge_coverage(previous.covered, coverage(samples or ()))
                    if kept
                    else coverage(everything)
                ),
            )
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
        samples: Sequence[tuple[torch.Tensor, ...]] | None = None,
        blocked: str | None = None,
        shared: _Shared | None = None,
    ) -> None:
        self.deadline = time.monotonic() + config.timeout_s
        # What forbids a search on a cache miss, why a cached kernel was not
        # used, and whether its revalidation was inconclusive.
        self.blocked = blocked
        self.shared, self.samples = shared, list(samples or ())
        self.cache_miss: str | None = None
        self.inconclusive = self.slow = False
        # Whether evaluating that cached kernel again could change the outcome.
        self.retryable = True
        self.task, self.key, self.config, self.policy = task, key, config, config.search
        self.backend, self.instructions = backend, instructions
        self.input_specs, self.output_specs = input_specs, output_specs
        self.region_id, self.cache = region_id, cache
        self.run = Path(tempfile.mkdtemp(prefix="run-", dir=cache))
        self.schema = _response_schema(backend.proposal_schema, self.policy.batch_size)
        self.propose_only_schema = _response_schema(
            backend.proposal_schema, self.policy.batch_size, propose_only=True
        )
        reference_path = self.run / REFERENCE_FILENAME
        save_reference(reference, reference_path)
        samples_path = None
        if samples:
            samples_path = self.run / SAMPLES_FILENAME
            save_samples(samples, samples_path)
        self.template = {
            "reference_path": str(reference_path),
            "samples_path": None if samples_path is None else str(samples_path),
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
            "screening": self.policy.screening,
            # Screens also build the TensorRT baseline until one search
            # evaluation gets past it, so a baseline that cannot be built ends
            # the search at once, as it does under engine screening.
            "check_baseline": config.objective == "tensorrt",
            "compile_options": task["compile_options"],
            # The worker's registrations apply the same code-generation options.
            "codegen": task["codegen"],
            # Holdout and cache checks never fit: they judge the exact kernel.
            "auto_fit": self.policy.auto_fit,
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
        if self.blocked is not None:
            generate = (
                f"compile once with generation allowed (allow_generation=True and "
                f"{_GENERATION_ENV} unset or 1), the same cache_dir, GPU and "
                "settings"
            )
            if self.inconclusive and not self.retryable:
                # Evaluating again cannot change the outcome.
                advice = "Resolve the cause that the check names, then compile again"
            elif self.inconclusive:
                advice = (
                    "Retry once the cause is resolved, for example with a less busy "
                    "GPU or larger timeout_s and candidate_timeout_s"
                )
            elif self.slow:
                # The kernel is correct; a busy GPU is the likeliest reason.
                advice = (
                    "Retry on a less busy GPU; if the speedup stays below "
                    f"min_speedup, {generate}, to search for a faster kernel"
                )
            else:
                advice = (
                    f"{generate[0].upper()}{generate[1:]}, to validate and cache a "
                    "kernel for this region; later compilations reuse it without "
                    "an agent"
                )
            raise RegionError(
                f"Kernel generation for region {self.region_id!r} is disabled by "
                f"{self.blocked}, and {self.cache_miss or self.uncached()}. "
                f"{advice}. Artifacts: {self.run}"
            )
        self.prepare_search()
        if self.policy.resume:
            self.resume()
        for attempt in range(self.config.max_attempts):
            if not self.author_round(attempt):
                break
        try:
            return self.finalize()
        except RegionError as error:
            # Nothing else runs the region, so the compilation fails.
            relax = (
                ""
                if self.config.objective == "candidate"
                else ", a lower min_speedup or objective='candidate'"
            )
            raise RegionError(
                f"{error}. Compilation has no fallback for the region: compile "
                "again with SearchPolicy(resume=True) to continue this search, or "
                f"with a larger budget (max_attempts, timeout_s){relax}; supply a "
                "kernel through fused_kernel(kernel=...); or remove the annotation"
            ) from None

    def validate(self, proposal: Any) -> dict[str, Any]:
        return self.backend.validate_candidate(
            proposal, len(self.input_specs), len(self.output_specs)
        )

    def register(self, candidate: dict[str, Any]) -> KernelSpec | partial[None]:
        return make_registration(
            candidate, self.backend, self.input_specs, self.output_specs, self.run
        )

    def from_cache(self) -> tuple[dict[str, Any], KernelSpec | partial[None]] | None:
        """Reuse an accepted kernel that passes a holdout check on this run.

        Identical regions of one compilation share a kernel, also with
        ``use_cache=False``, and load one copy of it; materialize_regions
        registers the final one for all of them.
        """
        if self.shared is not None:
            # It met the objective on this GPU in this compilation; only its
            # correctness on this region's samples is new.
            result = self.recheck(
                self.shared.candidate, self.shared.run, correctness_only=True
            )
            if result is None and self.inconclusive and self.retryable:
                # Once more before a search or a cache-only refusal: a busy GPU
                # is no reason for either, but a kernel that hangs on these
                # samples is.
                result = self.recheck(
                    self.shared.candidate, self.shared.run, correctness_only=True
                )
            return result
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
            self.cache_miss = f"its cached kernel entry is invalid ({error})"
            return None
        return self.recheck(candidate, self.run)

    def recheck(
        self, candidate: dict[str, Any], directory: Path, correctness_only: bool = False
    ) -> tuple[dict[str, Any], KernelSpec | partial[None]] | None:
        """Return the candidate, registered from ``directory``, if it passes."""
        report = self.evaluate(
            candidate,
            self.run,
            self.deadline,
            holdout=True,
            objective="candidate" if correctness_only else None,
        )
        write_json(self.run / "cache_check.json", report)
        passed = (
            report.get("status") == "passed"
            if correctness_only
            else _meets_objective(report, self.config)
        )
        if not passed:
            _LOGGER.warning("Agent region cache revalidation failed: %s", report)
            self.cache_miss = self.revalidation_failure(report)
            self.inconclusive = report.get("stage") in _INCONCLUSIVE_STAGES
            self.retryable = report.get("retryable", True) is not False
            # Speed advice needs the remeasured speedup that cache_miss reports.
            self.slow = report.get("status") == "passed" and all(
                _positive(report.get(name))
                for name in ("baseline_latency_ms", "latency_ms")
            )
            return None
        _LOGGER.info("Agent region %s: cache hit (%s)", self.region_id, self.cache)
        # Runs that reuse a kernel accumulate with every compilation, so they
        # drop their copy of the samples once checked; the accepting run keeps
        # its own.
        (self.run / SAMPLES_FILENAME).unlink(missing_ok=True)
        return candidate, make_registration(
            candidate, self.backend, self.input_specs, self.output_specs, directory
        )

    def prepare_search(self) -> None:
        """Add identical regions' samples to this run's, then publish its task."""
        if self.shared is not None and self.shared.samples:
            # A new kernel must also pass the samples of this compilation's
            # identical regions, or their searches would keep replacing each
            # other's cached kernel.
            self.samples = [*self.shared.samples, *self.samples]
            save_samples(self.samples, self.run / SAMPLES_FILENAME)
            self.task["samples"] = {
                "count": len(self.samples),
                "sha256": samples_digest(self.samples),
            }
            write_json(self.run / "task.json", self.task)
        self.evidence.add("task.json", json.dumps(self.task, indent=2))

    def revalidation_failure(self, report: dict[str, Any]) -> str:
        """Say why a cached kernel's revalidation report rules it out."""
        objective = self.config.objective
        if report.get("status") == "passed":
            shortfall = _speedup_shortfall(report, self.config)
            if shortfall is None:
                return (
                    "its cached kernel passed revalidation without a measured "
                    f"{objective} baseline (see cache_check.json)"
                )
            return (
                "its cached kernel passed the correctness checks, but its "
                f"remeasured {shortfall} (see cache_check.json)"
            )
        stage = report.get("stage")
        lines = str(report.get("error", "")).strip().splitlines()
        outcome = (
            "could not be revalidated"
            if stage in _INCONCLUSIVE_STAGES
            else "failed revalidation"
        )
        return (
            f"its cached kernel {outcome} ({stage}: "
            f"{lines[-1][:300] if lines else 'no error message'}; "
            "see cache_check.json)"
        )

    def uncached(self) -> str:
        """Explain a cache miss by a kernel cached for the same computation."""
        root = self.cache.parent
        manifests = sorted(root.glob("*/accepted.json"))
        closest: tuple[list[str], Path] | None = None
        for manifest in manifests:
            try:
                run = manifest.parent / Path(read_json_object(manifest)["run"]).name
                other = read_json_object(run / "task.json")
            except (OSError, KeyError, TypeError, ValueError):
                continue
            if self.changes_key("reference", other):
                continue
            fields = [
                # Of the search policy, only the tuning choices are keyed.
                "search.tuning" if name == "search" else name
                for name in sorted(self.task.keys() | other.keys())
                if self.changes_key(name, other)
            ]
            if fields and (closest is None or len(fields) < len(closest[0])):
                closest = fields, run
        if closest is None:
            return (
                "no accepted kernel is cached for this task, and none of the "
                f"{len(manifests)} kernels cached in {root} is for this captured "
                "computation"
            )
        fields, run = closest
        return (
            "no accepted kernel is cached for this task; a kernel is cached for the "
            f"same captured computation, but its task.json differs in "
            f"{', '.join(fields)} (compare {run / 'task.json'} with this run's)"
        )

    def changes_key(self, name: str, other: dict[str, Any]) -> bool:
        """Whether taking field ``name`` from another task changes this cache key."""
        task = {key: value for key, value in self.task.items() if key != name}
        if name in other:
            task[name] = other[name]
        try:
            return _task_digest(task) != self.key
        except (KeyError, TypeError, AttributeError):
            return True

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

    def author_limit(self, now: float) -> float:
        """When a round that starts at ``now`` must stop authoring.

        Evaluation time for the round's proposals stays reserved; every author
        turn gets its own slice, all ending by this limit.
        """
        remaining = self.search_deadline - now
        return self.search_deadline - min(
            2 * self.config.candidate_timeout_s, remaining / 2
        )

    def author_round(self, attempt: int) -> bool:
        """Request, expand, and measure one batch; return False to stop searching."""
        config = self.config
        now = time.monotonic()
        remaining = self.search_deadline - now
        author_limit = self.author_limit(now)
        if self.used >= self.max_candidates or not _useful(
            min(author_limit, now + config.author_timeout_s) - now,
            config.author_timeout_s,
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
                (self.schema, self.propose_only_schema),
                directory,
                author_limit,
                self.evidence,
                self.diagnose,
            )
        except (OSError, ValueError, TypeError, TimeoutError, RuntimeError) as error:
            if isinstance(
                error,
                # Unavailable only once waiting and asking again failed too.
                (_providers.ProviderError, _providers.ProviderUnavailableError),
            ):
                stage = "provider"
            elif isinstance(error, _AuthorTimeout) and error.shortened:
                # A turn that the deadline shortened ran out of time. Another
                # round is worth it only with a full turn, and the time to
                # evaluate what it proposes, still ahead.
                later = time.monotonic()
                full = self.author_limit(later) - later >= config.author_timeout_s
                stage = "author" if full else "budget"
            else:
                stage = "author"
            self.record(
                None,
                _error(stage, str(error)[-3000:]),
                directory,
                action,
                rejected=getattr(error, "reply", None),
            )
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
                failures.append((str(error), proposal))
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
                if isinstance(option, tuple):
                    raise ValueError(option[0])
                candidate = self.validate(option)
            except (ValueError, TypeError) as error:
                # The author sees what it sent, so it can repair that exact text.
                self.record(
                    None,
                    _error("source", str(error)[-3000:]),
                    candidate_directory,
                    action,
                    rejected=option[1] if isinstance(option, tuple) else option,
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
                candidate, report = self.adopt_fit(candidate, report)
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

    def adopt_fit(
        self, candidate: dict[str, Any], report: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the variant a shared-memory fit evaluated in place of ``candidate``.

        The evaluator measured that variant, so it is the one recorded and,
        if selected, accepted. It may differ only in launch settings and
        constant values.
        """
        fitted = report.get("fitted_candidate")
        if fitted is None:
            return candidate, report
        try:
            variant = self.validate(fitted)
            same = ("source", "kernel_name", "grid")
            if any(variant[key] != candidate[key] for key in same) or [
                item["name"] for item in variant["constants"]
            ] != [item["name"] for item in candidate["constants"]]:
                raise ValueError("it changes more than launch settings and constants")
        except (ValueError, TypeError, KeyError) as error:
            return candidate, _error(
                "setup", f"Evaluator returned an invalid fitted candidate: {error}"
            )
        _LOGGER.info(
            "Agent region %s: evaluated a shared-memory fit: %s",
            self.region_id,
            "; ".join(report.get("fit_log", ())),
        )
        # The history records the variant as the candidate; keep what changed.
        report = {
            key: value for key, value in report.items() if key != "fitted_candidate"
        }
        report["fitted_from"] = {
            key: copy.deepcopy(candidate[key])
            for key in ("num_stages", "constants")
            if key in candidate
        }
        return variant, report

    def evaluate(
        self,
        candidate: dict[str, Any],
        directory: Path,
        deadline: float,
        *,
        holdout: bool = False,
        incumbent: dict[str, Any] | None = None,
        objective: str | None = None,
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
                if objective is not None:
                    request["objective"] = objective
                if holdout:
                    request.update(
                        validation_seed=secrets.randbits(63),
                        validation_phase="holdout",
                        auto_fit=False,
                    )
                else:
                    self.used += 1
                    if incumbent is not None:
                        request["incumbent"] = incumbent
                report = _evaluate(request, directory, granted)
        except Timeout:
            return _error("budget", "GPU evaluation lock timed out")
        if report.get("stage") == "setup" and report.get("retryable", True):
            # Device memory or a device that other work held can free up;
            # only a repeated setup failure stands. The wait releases the GPU
            # to the evaluations of other processes.
            write_json(directory / "setup_failure.json", report)
            time.sleep(_SETUP_RETRY_WAIT_S)
            remaining = deadline - time.monotonic()
            if _useful(remaining, limit):
                try:
                    with self.gpu.acquire(timeout=remaining):
                        retry = min(limit, deadline - time.monotonic())
                        if _useful(retry, limit):
                            granted = retry
                            report = _evaluate(request, directory, granted)
                except Timeout:
                    # The GPU stayed busy; the first setup failure stands.
                    pass
        if not holdout and (
            report.get("status") == "passed" or report.get("stage") in _AFTER_BASELINE
        ):
            self.template["check_baseline"] = False
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
        rejected: Any = None,
    ) -> None:
        """Record an outcome; ``rejected`` is author output that was never evaluated.

        Rejected proposals and protocol-breaking replies stay in the history,
        and so in the next prompt, so the author can repair what it sent.
        """
        report = {**report, "meets_objective": _meets_objective(report, self.config)}
        feedback = directory / "feedback.json"
        write_json(feedback, report)
        entry = {"candidate": candidate, "result": report, "action": action}
        prefix = directory.relative_to(self.run).as_posix()
        if isinstance(rejected, dict):
            reply = "action" in rejected
            key = "rejected_reply" if reply else "rejected_proposal"
            entry[key] = _bounded(rejected)
            proposals = rejected.get("candidates") if reply else [rejected]
            sources = [
                item["source"]
                for item in (proposals if isinstance(proposals, list) else [])
                if isinstance(item, dict) and isinstance(item.get("source"), str)
            ]
            for index, source in enumerate(sources):
                suffix = "" if len(sources) == 1 else f"-{index + 1}"
                self.evidence.add(f"{prefix}/rejected_source{suffix}", source)
        self.history.append(entry)
        self.round.append(entry)
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
            *sorted(directory.glob("retry-*/provider.log")),
            *sorted(directory.glob("unavailable-*/provider.log")),
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

    def provider_cost(self) -> dict[str, Any]:
        """Total and log what author turns cost, as far as the provider said."""
        costs = []
        for path in self.run.rglob(_providers.SPEND_FILENAME):
            try:
                costs.append(float(read_json_object(path)["total_cost_usd"]))
            except (OSError, KeyError, TypeError, ValueError, OverflowError):
                continue
        if not costs:
            return {}
        total = math.fsum(costs)
        _LOGGER.info(
            "Agent region %s: the provider reported a cost of $%.4f for this "
            "search (author turns priced: %d)",
            self.region_id,
            total,
            len(costs),
        )
        return {"provider_cost_usd": total, "priced_author_turns": len(costs)}

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
                **self.provider_cost(),
            },
        )
        self.save_checkpoint()
        # Screened candidates whose speedup is undecided remain eligible.
        choices = [
            entry
            for entry in self.ranked()
            if entry["report"]["meets_objective"] is not False
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
            stage = report.get("stage")
            # Drop a disproved finalist; an inconclusive check, such as a
            # timeout, keeps it in the checkpoint for resume. A screened
            # finalist met its first engine here, so a compile or engine failure
            # reproduces, as in the search, rather than being a flaky rebuild.
            if (
                report.get("status") == "passed"
                or stage in ("correctness", "crash")
                or (
                    entry["report"].get("execution") == "launch"
                    and stage in ("compile", "engine")
                )
            ):
                self.measured.pop(digest(entry["candidate"]), None)
            self.save_checkpoint()
        if report.get("status") == "passed":
            # Correct, but too slow or without a baseline to compare with.
            shortfall = _speedup_shortfall(report, self.config)
            reason = (
                "Performance objective not met: "
                + (
                    f"its {shortfall}"
                    if shortfall is not None
                    else f"no {self.config.objective} baseline was measured"
                )
                + f" (see {directory.name}/feedback.json)"
            )
        else:
            reason = report.get("error", "unknown error")
        raise RegionError(
            f"Generated kernel for region {self.region_id!r} failed final validation: "
            f"{reason}. Artifacts: {self.run}"
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
        slower = _slower_than_eager(validation, self.config)
        if slower:
            _LOGGER.warning("Agent region %s: %s", self.region_id, slower)
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
