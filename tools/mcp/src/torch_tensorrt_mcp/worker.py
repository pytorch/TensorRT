"""Execution-only code. Heavy imports occur after spawn and log redirection."""

from __future__ import annotations

import os
import random
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from .compat import check_core_compatibility
from .policy import Policy, PolicyError, confined_path, file_hash
from .storage import read_json, write_json


def _samples(value: Any) -> list[Any]:
    from torch_tensorrt.verification import InputSample

    if isinstance(value, InputSample):
        return [value]
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, InputSample) for item in value)
    ):
        return value
    if isinstance(value, tuple):
        return [InputSample(args=value)]
    if isinstance(value, dict) and set(value) <= {
        "args",
        "kwargs",
        "profile",
        "sample_id",
        "label",
    }:
        return [InputSample(**value)]
    raise ValueError(
        "input factory must return InputSample, nonempty list[InputSample], an argument tuple, or an InputSample field dict"
    )


def _verify_provenance(current: dict[str, Any], original: dict[str, Any]) -> None:
    keys = (
        "weights_sha256",
        "constants_sha256",
        "input_sha256",
        "graph_sha256",
        "range_constraints",
        "samples",
    )
    mismatched = [key for key in keys if current.get(key) != original.get(key)]
    if mismatched:
        raise PolicyError(
            "reconstructed reference differs from compilation provenance: "
            + ", ".join(mismatched)
        )


def _source(policy: Policy, job_id: str) -> tuple[Path, dict[str, Any]]:
    root = policy.job_path(job_id)
    manifest = read_json(confined_path(root, "manifest.json"))
    if manifest.get("status") != "succeeded":
        raise PolicyError("source compilation job did not succeed")
    request_path = confined_path(root, "request.json")
    request_entry = next(
        (
            entry
            for entry in manifest.get("entries", [])
            if entry.get("path") == "request.json"
        ),
        None,
    )
    if request_entry is None or file_hash(request_path) != request_entry["sha256"]:
        raise PolicyError("source request failed manifest verification")
    request = read_json(request_path)
    if request.get("operation") != "compile":
        raise PolicyError("source job is not a compilation")
    artifact = next(
        (
            entry
            for entry in manifest.get("entries", [])
            if entry.get("artifact_id") == "compiled_model"
        ),
        None,
    )
    if artifact is None:
        raise PolicyError("source manifest has no compiled model")
    path = confined_path(root, artifact["path"])
    if (
        artifact["path"] != "artifacts/compiled.pt2"
        or file_hash(path) != artifact["sha256"]
        or path.stat().st_size != artifact["size_bytes"]
    ):
        raise PolicyError("compiled model failed manifest verification")
    provenance_path = confined_path(root, "input_provenance.json")
    provenance_entry = next(
        (
            entry
            for entry in manifest.get("entries", [])
            if entry.get("path") == "input_provenance.json"
        ),
        None,
    )
    if (
        provenance_entry is None
        or file_hash(provenance_path) != provenance_entry["sha256"]
    ):
        raise PolicyError("source provenance failed manifest verification")
    provenance = read_json(provenance_path)
    if (
        request["request"]["model"] not in policy.models
        or request["request"]["inputs"] not in policy.inputs
    ):
        raise PolicyError("source factories are no longer configured")
    return path, {**request["request"], "source_provenance": provenance}


def execute(
    policy: Policy,
    operation: str,
    request: dict[str, Any],
    path: Path,
    job_id: str,
    emit: Any,
) -> str:
    """Invoke only public core APIs and administrator-selected callables."""
    check_core_compatibility()
    import torch
    import torch_tensorrt
    from torch_tensorrt.artifacts import (
        collect_environment,
        hash_graph,
        hash_inputs,
        hash_weights,
    )

    environment = collect_environment(job_id=job_id)
    write_json(path / "environment.json", environment.to_dict())
    if operation == "collect_environment":
        return "environment"

    if not all(
        hasattr(torch_tensorrt.dynamo, name)
        for name in ("analyze", "compile_with_report")
    ):
        raise RuntimeError(
            "Installed Torch-TensorRT is missing the required public agent-workflow APIs"
        )

    source_path = None
    model_request = request
    if operation in ("validate", "benchmark"):
        source_path, model_request = _source(policy, request["source_job_id"])
    policy.validate_model_request(model_request)
    seed = model_request["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model_factory = policy.load_factory(policy.models[model_request["model"]])
    input_factory = policy.load_factory(policy.inputs[model_request["inputs"]])
    emit({"stage": "capture", "message": "Loading configured model and inputs"})
    captured = model_factory(**model_request["model_parameters"])
    if isinstance(captured, torch.export.ExportedProgram):
        model = captured.module()
    elif isinstance(captured, torch.nn.Module):
        model = captured.eval()
    else:
        raise ValueError("model factory must return torch.nn.Module or ExportedProgram")
    samples = _samples(input_factory(seed=seed, **model_request["input_parameters"]))
    exported = (
        captured
        if isinstance(captured, torch.export.ExportedProgram)
        else torch.export.export(model, samples[0].args, samples[0].kwargs)
    )
    capture_performed = not isinstance(captured, torch.export.ExportedProgram)
    provenance = {
        "seed": seed,
        "model": model_request["model"],
        "inputs": model_request["inputs"],
        "samples": [
            {
                "sample_id": sample.sample_id,
                "profile": sample.profile,
                "label": sample.label,
            }
            for sample in samples
        ],
        "weights_sha256": hash_weights(model.state_dict()),
        "constants_sha256": hash_inputs((), exported.constants),
        "graph_sha256": hash_graph(exported.graph_module),
        "range_constraints": {
            str(key): str(value)
            for key, value in sorted(
                exported.range_constraints.items(), key=lambda item: str(item[0])
            )
        },
        "input_sha256": [hash_inputs(sample.args, sample.kwargs) for sample in samples],
        "source_job_id": request.get("source_job_id"),
        "capture_performed": capture_performed,
    }
    if source_path is not None:
        _verify_provenance(provenance, model_request["source_provenance"])
    write_json(path / "input_provenance.json", provenance)

    if source_path is not None:
        from torch_tensorrt.verification import benchmark, validate

        compiled = torch_tensorrt.load(str(source_path)).module()
        emit(
            {
                "stage": operation,
                "message": "Executing verification against serialized compilation",
            }
        )
        if operation == "validate":
            report = validate(
                model,
                compiled,
                samples,
                atol=request["atol"],
                rtol=request["rtol"],
                relative_floor=request["relative_floor"],
                seed=seed,
                job_id=job_id,
            )
        else:
            if request["sample_index"] >= len(samples):
                raise ValueError(
                    "sample_index exceeds the configured input factory's sample count"
                )
            sample = samples[request["sample_index"]]
            report = benchmark(
                model,
                compiled,
                sample,
                warmup=request["warmup"],
                iterations=request["iterations"],
                percentiles=request["percentiles"],
                batch_size=request["batch_size"],
                seed=seed,
                job_id=job_id,
            )
            report = replace(
                report,
                preprocessing={
                    **report.preprocessing,
                    "sample_index": request["sample_index"],
                    "profile": sample.profile,
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                },
            )
        write_json(
            path / (operation.replace("validate", "validation") + ".json"),
            report.to_dict(),
        )
        return "validation" if operation == "validate" else "benchmark"

    from torch_tensorrt.dynamo import analyze, compile_with_report

    sample = samples[0]
    settings = dict(model_request["settings"])
    settings["timing_cache_path"] = str(path / "artifacts" / "timing_cache.bin")
    settings["cache_built_engines"] = False
    settings["reuse_cached_engines"] = False
    if operation == "analyze":
        report = analyze(
            exported,
            arg_inputs=sample.args,
            kwarg_inputs=sample.kwargs,
            job_id=job_id,
            event_sink=emit,
            diagnostics_dir=path / "diagnostics",
            **settings,
        )
        report = replace(report, capture_performed=capture_performed)
        write_json(path / "analysis.json", report.to_dict())
        return "analysis"

    result = compile_with_report(
        exported,
        arg_inputs=sample.args,
        kwarg_inputs=sample.kwargs,
        job_id=job_id,
        event_sink=emit,
        diagnostics_dir=path / "diagnostics",
        **settings,
    )
    if result.report.analysis is not None:
        result = replace(
            result,
            report=replace(
                result.report,
                analysis=replace(
                    result.report.analysis, capture_performed=capture_performed
                ),
            ),
        )
    emit({"stage": "serialization", "message": "Saving compiled ExportedProgram"})
    torch_tensorrt.save(
        result.module,
        str(path / "artifacts" / "compiled.pt2"),
        arg_inputs=sample.args,
        kwarg_inputs=sample.kwargs,
        output_format="exported_program",
        retrace=False,
    )
    write_json(
        path / "compilation.json",
        replace(
            result.report, artifact_ids=[*result.report.artifact_ids, "compiled_model"]
        ).to_dict(),
    )
    return "compilation"


def worker_main(
    policy: Policy,
    operation: str,
    request: dict[str, Any],
    staging: str,
    job_id: str,
    connection: Any,
    start_connection: Any,
    executor: Any = None,
) -> None:
    path = Path(staging)
    # Do not touch CUDA until the supervisor has durably recorded this PID.
    # If the supervisor dies first, its pipe closes and this worker exits.
    try:
        try:
            should_start = start_connection.recv() == "start"
        except EOFError:
            should_start = False
    finally:
        start_connection.close()
    if not should_start:
        connection.close()
        return
    if os.name == "posix":
        os.setsid()
    # Capture native writes too. The MCP SDK also reserves its wire descriptors,
    # but diagnostics belong to this job rather than the server's stderr.
    stdout = os.open(
        path / "diagnostics" / "stdout.log",
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    stderr = os.open(
        path / "diagnostics" / "stderr.log",
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    os.dup2(stdout, 1)
    os.dup2(stderr, 2)
    os.close(stdout)
    os.close(stderr)

    current_stage = "environment"

    def emit(event: Any) -> None:
        nonlocal current_stage
        value = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        stage = value.get("stage")
        current_stage = {
            "validate": "validation",
            "compile": "compilation",
            "analyze": "analysis",
            "collect_environment": "environment",
        }.get(stage, stage) or current_stage
        connection.send(value)

    try:
        report_name = (executor or execute)(
            policy, operation, request, path, job_id, emit
        )
        report = read_json(path / (report_name + ".json"))
        write_json(
            path / "worker-result.json",
            {
                "status": report.get("status", "succeeded"),
                "report": report_name,
                "failure": report.get("failure"),
            },
        )
    except BaseException as exc:
        traceback.print_exc()
        report = getattr(exc, "report", None)
        if report is not None and hasattr(report, "to_dict"):
            write_json(path / "compilation.json", report.to_dict())
        stage = current_stage
        failure = (
            report.failure.to_dict()
            if report is not None and getattr(report, "failure", None) is not None
            else None
        )
        if failure is None and executor is None:
            try:
                from torch_tensorrt.reporting import classify_exception

                failure = classify_exception(exc, stage).to_dict()
            except (ImportError, RuntimeError):
                pass
        failure = failure or {
            "stage": stage,
            "category": (
                "out_of_memory"
                if isinstance(exc, MemoryError)
                else (
                    "invalid_input"
                    if isinstance(exc, (ValueError, PolicyError))
                    else (
                        "unsupported_environment"
                        if isinstance(exc, ImportError)
                        else "internal_error"
                    )
                )
            ),
            "message": str(exc)[:2048],
            "retry_unchanged": False,
            "retry_after_change": True,
            "suggested_actions": [
                "Inspect the retained diagnostics and adjust the request or environment."
            ],
            "exception_type": type(exc).__name__,
        }
        write_json(
            path / "worker-result.json",
            {
                "status": "failed",
                "failure": failure,
            },
        )
    finally:
        connection.close()
