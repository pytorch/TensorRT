# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded author/evaluator loop, persisted evidence, and kernel cache."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import os
import sys
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from filelock import FileLock, Timeout

from torch_tensorrt.region import KernelGenerator, RegionError

from .._plugins import _boundaries
from . import _providers
from ._candidate import make_registration, validate_candidate

_LOGGER = logging.getLogger(__name__)
_VERSION = 2


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
        nodes[node].meta = {}
    reference = torch.fx.GraphModule(child, graph)
    reference.meta = {}
    return reference, constants


def _target(device: torch.device, backend: str) -> dict[str, Any]:
    import tensorrt

    import torch_tensorrt

    properties = torch.cuda.get_device_properties(device)
    versions = {
        "torch": torch.__version__,
        "torch_tensorrt": torch_tensorrt.__version__,
        "tensorrt": tensorrt.__version__,
        "cuda": torch.version.cuda,
    }
    if backend == "cutile":
        from torch_tensorrt.kernels._cutile import (
            _cutile_driver_version,
            validate_cutile_toolchain,
        )

        versions["driver"] = list(_cutile_driver_version())
        versions["cutile_compiler"] = list(
            validate_cutile_toolchain("agent_generation")
        )
    else:
        from torch_tensorrt.kernels._triton import _driver_max_ptx_version

        versions["driver_ptx"] = _driver_max_ptx_version()
    package = "triton" if backend == "triton" else "cuda-tile"
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        raise RegionError(f"Kernel generation backend requires {package}") from None
    return {
        "name": properties.name,
        "capability": [properties.major, properties.minor],
        "multiprocessors": properties.multi_processor_count,
        "versions": versions,
    }


def _prompt_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep recent feedback and the incumbent without resending every kernel."""
    indices = set(range(max(0, len(history) - 3), len(history)))
    passing = [
        index
        for index, attempt in enumerate(history)
        if attempt["result"].get("status") == "passed"
    ]
    if passing:
        indices.add(min(passing, key=lambda i: history[i]["result"]["latency_ms"]))
    return [history[index] for index in sorted(indices)]


def _prompt(
    task: dict[str, Any], history: list[dict[str, Any]], remaining_s: float
) -> str:
    backend = task["backend"]
    syntax = (
        "Import triton and triton.language as tl. Define exactly one @triton.jit function. "
        "Tensor parameters have no annotations; constant parameters use tl.constexpr. "
        "Mask all boundary loads/stores, disable floating-point reassociation when needed, "
        "and preserve intermediate rounding required by the PyTorch reference. "
        "For float16/bfloat16, cast each reference operation's result back to that dtype "
        "before continuing. For exact float32 rounding, you may import "
        "triton.language.extra.cuda.libdevice as libdevice and use add_rn/mul_rn; "
        "there is no enable_fp_fusion compiler option in this interface."
        if backend == "triton"
        else "Import cuda.tile as ct. Define exactly one @ct.kernel function. Tensor parameters "
        "have no annotations; constants use ct.Constant[int], ct.Constant[float], or "
        "ct.Constant[bool]. Arrays are flattened rank-one views. Use ct.bid, ct.load, "
        "ct.store and tile operations; ct.load(array, index=(block,), shape=(BLOCK,)). "
        "num_warps and num_stages must be null."
    )
    return (
        "Author a fast, correct GPU kernel for the exact captured region below. "
        "Return ONLY a JSON object matching the supplied output schema. Do not use tools, "
        "run programs, write files, change tests, or claim measured results. The executor "
        "compiles your code into a TensorRT QDP plugin and reports actual validation/timing. "
        "Implement the entire reference, including all outputs, with one kernel. "
        "Parameters must be arg0, arg1, ... for inputs, then out0, out1, ... for outputs, "
        "then constants in the exact order of the constants array. No runtime scalar "
        "parameters. Use literal JSON grid dimensions (1 to 3 positive integers). "
        "Only backend imports and this single kernel definition are allowed at module "
        "scope: no helpers, assignments, allocations, PyTorch fallback or host wrapper. "
        "All dimensions are static. Choose and tune constants/grid/compiler options. "
        "Use prior feedback to change the source or launch configuration; unchanged "
        "passing candidates reuse their prior evaluation. "
        "Source must use only the selected DSL and safe arithmetic; no host introspection, "
        "filesystem, network, inline assembly or dynamic Python execution. "
        + syntax
        + "\nTASK:\n"
        + json.dumps(task, indent=2)
        + "\nRECENT ATTEMPTS AND BEST CANDIDATE (executor results, fastest passing candidate is retained):\n"
        + json.dumps(_prompt_history(history), indent=2)
        + f"\nRemaining search time: {max(0, int(remaining_s))} seconds."
    )


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
) -> partial[None]:
    """Generate a kernel after capture and return the existing registration form."""
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
        "target": _target(input_specs[0].device, config.backend),
        "validation": {
            "num_tests": config.num_tests,
            "benchmark_iters": config.benchmark_iters,
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
) -> partial[None]:
    run = Path(tempfile.mkdtemp(prefix="run-", dir=cache))
    _write_json(run / "task.json", task)
    reference_path = run / "reference.pt"
    torch.save(reference, reference_path)
    evaluation = {
        "reference_path": str(reference_path),
        "backend": config.backend,
        "inputs": task["inputs"],
        "outputs": task["outputs"],
        **task["validation"],
    }
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_report: dict[str, Any] | None = None
    best_evaluation_path: Path | None = None
    passing_evaluations: dict[str, tuple[dict[str, Any], Path]] = {}
    manifest = cache / "accepted.json"

    def evaluate(candidate: dict[str, Any], directory: Path) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"status": "error", "error": "Generation wall-time budget exhausted"}
        gpu_lock = FileLock(
            str(cache.parent / f"gpu-{input_specs[0].device.index}.lock")
        )
        try:
            with gpu_lock.acquire(timeout=remaining):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {
                        "status": "error",
                        "error": "Generation wall-time budget exhausted waiting for GPU",
                    }
                return _evaluate(
                    {**evaluation, "candidate": candidate},
                    directory,
                    min(config.candidate_timeout_s, remaining),
                )
        except Timeout:
            return {"status": "error", "error": "GPU evaluation lock timed out"}

    if config.use_cache and manifest.exists():
        try:
            stored = json.loads(manifest.read_text())
            candidate = stored["candidate"]
            if stored["task_sha256"] != _digest(task) or stored[
                "candidate_sha256"
            ] != _digest(candidate):
                raise ValueError("Cache identity or content digest mismatch")
            candidate = validate_candidate(
                candidate, config.backend, len(input_specs), len(output_specs)
            )
            report = evaluate(candidate, run)
            _write_json(run / "cache_check.json", report)
            if report.get("status") == "passed":
                _LOGGER.info("Agent region %s: cache hit (%s)", region_id, cache)
                cached_registration: partial[None] = make_registration(
                    candidate, config.backend, input_specs, output_specs, run
                )
                return cached_registration
            _LOGGER.warning(
                "Agent region cache revalidation failed: %s", report.get("error")
            )
        except (OSError, KeyError, ValueError, TypeError) as error:
            _LOGGER.warning("Ignoring invalid agent region cache: %s", error)

    for attempt in range(config.max_attempts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        directory = run / f"attempt-{attempt + 1}"
        directory.mkdir()
        prompt = _prompt(task, history, remaining)
        (directory / "prompt.txt").write_text(prompt)
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
        try:
            proposal = _providers.propose(
                provider=config.provider,
                model=config.model,
                prompt=prompt,
                workdir=directory,
                timeout_s=remaining,
            )
            candidate = validate_candidate(
                proposal, config.backend, len(input_specs), len(output_specs)
            )
            _write_json(directory / "candidate.json", candidate)
            candidate_digest = _digest(candidate)
            if candidate_digest in passing_evaluations:
                previous_report, evaluation_path = passing_evaluations[candidate_digest]
                report = {**previous_report, "reused_from": str(evaluation_path)}
            else:
                report = evaluate(candidate, directory)
                if report.get("status") == "passed":
                    passing_evaluations[candidate_digest] = (report, evaluation_path)
        except _providers.ProviderError as error:
            report = {"status": "error", "error": str(error)[-3000:]}
            provider_failed = True
        except (OSError, ValueError, TypeError, TimeoutError, RuntimeError) as error:
            report = {"status": "error", "error": str(error)[-3000:]}
        _write_json(directory / "feedback.json", report)
        history.append({"candidate": candidate, "result": report})
        if report.get("status") == "passed" and (
            best_report is None or report["latency_ms"] < best_report["latency_ms"]
        ):
            best, best_report = candidate, report
            best_evaluation_path = evaluation_path
        _LOGGER.info("Agent region %s: %s", region_id, report)
        if provider_failed:
            break

    _write_json(run / "history.json", history)
    if best is None:
        detail = (
            history[-1]["result"].get("error", "No passing candidate")
            if history
            else "Generation budget exhausted"
        )
        raise RegionError(
            f"Agent generated no valid kernel for region {region_id!r}: {detail}. Artifacts: {run}"
        )
    accepted = {
        "task_sha256": _digest(task),
        "candidate_sha256": _digest(best),
        "candidate": best,
        "report": best_report,
        "evaluation_path": str(best_evaluation_path),
        "provider": config.provider,
        "model": config.model,
        "run": str(run),
    }
    _write_json(run / "accepted.json", accepted)
    if config.use_cache:
        _write_json(manifest, accepted)
    _LOGGER.info("Agent region %s: selected kernel; artifacts: %s", region_id, run)
    registration: partial[None] = make_registration(
        best, config.backend, input_specs, output_specs, run
    )
    return registration
