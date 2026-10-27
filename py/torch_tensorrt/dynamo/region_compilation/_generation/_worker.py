# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate generated kernels in a disposable process and a real TRT engine."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import operator
import re
import secrets
import statistics
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, TypeVar

import torch

_Result = TypeVar("_Result")


class _EvaluationError(RuntimeError):
    """Keep the failed executor stage separate from its diagnostic traceback."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.diagnostics: dict[str, Any] | None = None


def _run_stage(
    stage: str, function: Callable[..., _Result], *args: Any, **kwargs: Any
) -> _Result:
    try:
        return function(*args, **kwargs)
    except _EvaluationError:
        raise
    except Exception as error:
        raise _EvaluationError(stage, str(error)) from error


def _outputs(value: Any) -> tuple[torch.Tensor, ...]:
    values = tuple(value) if isinstance(value, (tuple, list)) else (value,)
    if not values or any(not isinstance(item, torch.Tensor) for item in values):
        raise ValueError("The reference must return a Tensor or flat tuple of Tensors")
    return values


def _dtype(name: str) -> torch.dtype:
    supported = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return supported[name.removeprefix("torch.")]
    except KeyError as error:
        raise ValueError(f"Unsupported generated-kernel dtype: {name}") from error


def _check_contract(
    values: Sequence[torch.Tensor], specs: Sequence[dict[str, Any]]
) -> None:
    if len(values) != len(specs):
        raise ValueError("Output count does not match the captured region")
    for index, (value, spec) in enumerate(zip(values, specs)):
        if (
            tuple(value.shape) != tuple(spec["shape"])
            or value.dtype != _dtype(spec["dtype"])
            or value.device != torch.device(spec["device"])
            or not value.is_contiguous()
        ):
            raise ValueError(f"Output {index} does not match the captured contract")


def _input_cases(
    specs: Sequence[dict[str, Any]],
    num_tests: int,
    check_nonfinite: bool,
    seed: int | None = None,
) -> Iterator[tuple[torch.Tensor, ...]]:
    if not specs or num_tests < 1:
        raise ValueError("At least one input and one random test are required")
    device = torch.device(specs[0]["device"])
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError("validation_seed must be an integer in [0, 2**63)")
    generator = torch.Generator(device=device).manual_seed(
        secrets.randbits(63) if seed is None else seed
    )

    def random_inputs() -> tuple[torch.Tensor, ...]:
        inputs = tuple(
            torch.randn(
                tuple(spec["shape"]),
                dtype=_dtype(spec["dtype"]),
                device=spec["device"],
                generator=generator,
            )
            for spec in specs
        )
        # Mix representable rounding boundaries into the random battery without
        # replacing small/scalar cases or changing the number of validation cases.
        for index, value in enumerate(inputs):
            if value.numel() >= 8:
                limits = torch.finfo(value.dtype)
                boundaries = value.new_tensor(
                    [1.0 + limits.eps, 1.0 - limits.eps / 2, limits.tiny]
                )
                value.reshape(-1)[-3:] = boundaries * (-1 if index % 2 else 1)
        return inputs

    for _ in range(num_tests):
        yield random_inputs()
    for value in (0.0, -1.0, 1.0):
        yield tuple(
            torch.full(
                tuple(spec["shape"]),
                value,
                dtype=_dtype(spec["dtype"]),
                device=spec["device"],
            )
            for spec in specs
        )
    if check_nonfinite:
        for index in range(len(specs)):
            # Exercise every input independently, including scalar tensors.
            for value in (float("nan"), float("inf"), -float("inf")):
                inputs = random_inputs()
                inputs[index].reshape(-1)[0] = value
                yield inputs


def _restore_metadata(
    reference: torch.fx.GraphModule,
    inputs: Sequence[torch.Tensor],
    outputs: Sequence[torch.Tensor],
) -> None:
    placeholders = [node for node in reference.graph.nodes if node.op == "placeholder"]
    output = next(node for node in reference.graph.nodes if node.op == "output")
    returned = output.args[0]
    nodes = tuple(returned) if isinstance(returned, (tuple, list)) else (returned,)
    if len(placeholders) != len(inputs) or len(nodes) != len(outputs):
        raise ValueError("Reference boundaries do not match the captured contract")
    for node, value in zip(placeholders, inputs):
        node.meta["val"] = value
    for node, value in zip(nodes, outputs):
        if not isinstance(node, torch.fx.Node):
            raise ValueError("Reference outputs must be graph nodes")
        node.meta["val"] = value


def _build_engine(
    reference: torch.fx.GraphModule,
    registration: Any,
    inputs: tuple[torch.Tensor, ...],
    num_outputs: int,
) -> torch.nn.Module:
    import torch_tensorrt
    from torch_tensorrt.dynamo.region_compilation._plugins import register_region_plugin
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

    op = register_region_plugin(
        reference, registration, region_id="generated_candidate"
    )
    graph = torch.fx.Graph()
    args = tuple(graph.placeholder(f"arg{index}") for index in range(len(inputs)))
    result = graph.call_function(op, args)
    if num_outputs > 1:
        graph.output(
            tuple(
                graph.call_function(operator.getitem, (result, i))
                for i in range(num_outputs)
            )
        )
    else:
        graph.output(result)
    module = torch.fx.GraphModule({}, graph).eval()
    compiled = torch_tensorrt.compile(
        module,
        inputs=list(inputs),
        ir="dynamo",
        strict=True,
        # The early compiler size check also applies to fully supported graphs.
        min_block_size=1,
        require_full_compilation=True,
        offload_module_to_cpu=False,
        disable_tf32=True,
    )
    if not isinstance(compiled, torch.nn.Module):
        raise _EvaluationError("engine", "Compilation did not return a TensorRT module")
    engines = [
        item for item in compiled.modules() if isinstance(item, TorchTensorRTModule)
    ]
    if len(engines) != 1 or str(op) not in engines[0].get_layer_info():
        raise _EvaluationError(
            "engine",
            f"Generated kernel {op} was not placed inside a TensorRT engine: "
            f"{[engine.get_layer_info() for engine in engines]}",
        )
    if any(
        node.op == "call_function" and node.target == op
        for item in compiled.modules()
        if isinstance(item, torch.fx.GraphModule)
        for node in item.graph.nodes
    ):
        raise _EvaluationError(
            "engine", "Generated kernel has an eager fallback outside TensorRT"
        )
    return compiled


def _build_reference_engine(
    reference: torch.fx.GraphModule, inputs: tuple[torch.Tensor, ...]
) -> torch.nn.Module:
    """Compile the unannotated region and reject any remaining eager computation."""
    import torch_tensorrt
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

    compiled = torch_tensorrt.compile(
        copy.deepcopy(reference),
        inputs=list(inputs),
        ir="dynamo",
        strict=True,
        min_block_size=1,
        require_full_compilation=True,
        offload_module_to_cpu=False,
        disable_tf32=True,
    )
    if not isinstance(compiled, torch.nn.Module):
        raise RuntimeError("TensorRT baseline compilation did not return a module")
    if not any(isinstance(item, TorchTensorRTModule) for item in compiled.modules()):
        raise RuntimeError("TensorRT baseline did not produce a TensorRT engine")
    for module in compiled.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if node.op == "call_method" or (
                node.op == "call_function" and node.target != operator.getitem
            ):
                raise RuntimeError("TensorRT baseline contains eager computation")
            if node.op == "call_module" and not isinstance(
                module.get_submodule(node.target),
                (torch.fx.GraphModule, TorchTensorRTModule),
            ):
                raise RuntimeError("TensorRT baseline contains an eager module")
    return compiled


def _capture_timer(
    function: Callable[..., Any], inputs: tuple[torch.Tensor, ...], iterations: int
) -> Callable[[], float]:
    """Prepare a graph once; each returned measurement excludes Python launch gaps."""
    if iterations < 1:
        raise ValueError("benchmark_iters must be positive")
    stream = torch.cuda.Stream(device=inputs[0].device)
    stream.wait_stream(torch.cuda.current_stream(inputs[0].device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            function(*inputs)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(iterations):
            function(*inputs)
    with torch.cuda.stream(stream):
        for _ in range(3):
            graph.replay()
        stream.synchronize()

    def measure() -> float:
        with torch.cuda.stream(stream):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            latency = start.elapsed_time(end) / iterations
        if not math.isfinite(latency) or latency <= 0:
            raise RuntimeError("Benchmark did not produce a positive finite latency")
        return float(latency)

    return measure


def _benchmark(
    functions: dict[str, Callable[..., Any]],
    inputs: tuple[torch.Tensor, ...],
    iterations: int,
    repeats: int = 5,
) -> dict[str, dict[str, Any]]:
    if type(repeats) is not int or repeats < 1:
        raise ValueError("benchmark_repeats must be a positive integer")
    timers = {
        name: _capture_timer(function, inputs, iterations)
        for name, function in functions.items()
    }
    samples: dict[str, list[float]] = {name: [] for name in timers}
    names = list(timers)

    def measure(name: str) -> None:
        sample = timers[name]()
        if not math.isfinite(sample) or sample <= 0:
            raise RuntimeError("Benchmark did not produce a positive finite latency")
        samples[name].append(sample)

    if "reference" in timers:
        measure("reference")
    # Alternate the order so the same arm does not always benefit from drift.
    for repeat in range(repeats):
        for name in names if repeat % 2 == 0 else reversed(names):
            measure(name)
    if "reference" in timers:
        measure("reference")
    results = {}
    for name, values in samples.items():
        median = statistics.median(values)
        spread = max(values) - min(values)
        results[name] = {
            "samples_ms": values,
            "median_ms": median,
            "spread_ms": spread,
            "relative_spread": spread / median,
            "mad_ms": statistics.median(abs(value - median) for value in values),
        }
    return results


def _latency(
    function: Callable[..., Any],
    inputs: tuple[torch.Tensor, ...],
    iterations: int,
    repeats: int = 5,
) -> float:
    return float(
        _benchmark({"candidate": function}, inputs, iterations, repeats)["candidate"][
            "median_ms"
        ]
    )


def _verify_case(
    reference: Callable[..., Any],
    candidate: Callable[..., Any],
    inputs: tuple[torch.Tensor, ...],
    output_specs: Sequence[dict[str, Any]],
    *,
    rtol: float | None,
    atol: float | None,
) -> None:
    originals = tuple(item.clone() for item in inputs)
    expected = _outputs(reference(*(item.clone() for item in inputs)))
    _check_contract(expected, output_specs)
    actual = _outputs(candidate(*inputs))
    _check_contract(actual, output_specs)
    for value, original in zip(inputs, originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0, equal_nan=True)
    for value, wanted in zip(actual, expected):
        torch.testing.assert_close(value, wanted, rtol=rtol, atol=atol, equal_nan=True)


def evaluate(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    enabled = request.get("diagnostics", False)
    if type(enabled) is not bool:
        raise _EvaluationError("source", "diagnostics must be a boolean")
    diagnostics: dict[str, Any] | None = {} if enabled else None
    try:
        result = _run_stage("compile", _evaluate, request, directory, diagnostics)
    except _EvaluationError as error:
        error.diagnostics = diagnostics
        raise
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    return result


def _candidate_diagnostics(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "launch": {
            key: copy.deepcopy(candidate[key])
            for key in (
                "kernel_name",
                "grid",
                "block",
                "constants",
                "num_warps",
                "num_stages",
                "shared_mem_bytes",
            )
            if key in candidate
        },
        "launch_basis": "validated_proposal",
        "compilation_scope": "backend_registration_and_tensorrt_engine_build",
        "compilation_ms": None,
        "compilation_status": "not_started",
        "compiler_resources": {
            "registers_per_thread": None,
            "shared_memory_bytes": None,
            "unavailable_reason": (
                "The registration API does not expose compiled resource metadata; "
                "launch shared_mem_bytes is requested dynamic shared memory."
            ),
        },
    }


@contextmanager
def _compilation_diagnostics(details: dict[str, Any] | None) -> Iterator[None]:
    """Time existing build work without compiling or profiling anything extra."""
    if details is None:
        yield
        return
    started = time.perf_counter()
    details["compilation_status"] = "failed"
    try:
        yield
        details["compilation_status"] = "completed"
    finally:
        details["compilation_ms"] = (time.perf_counter() - started) * 1000


def _check_backend_identity(backend: Any, request: dict[str, Any]) -> None:
    if "backend_version" in request and backend.version != request["backend_version"]:
        raise ValueError("Backend version changed between authoring and evaluation")
    if "proposal_schema_sha256" in request:
        digest = hashlib.sha256(
            json.dumps(
                backend.proposal_schema, sort_keys=True, allow_nan=False
            ).encode()
        ).hexdigest()
        if digest != request["proposal_schema_sha256"]:
            raise ValueError(
                "Backend proposal schema changed between authoring and evaluation"
            )


def _evaluate(
    request: dict[str, Any],
    directory: Path,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from ._backends import get_backend

    backend = _run_stage("source", get_backend, request["backend"])
    _run_stage("source", _check_backend_identity, backend, request)

    input_specs, output_specs = request["inputs"], request["outputs"]
    candidates = {"candidate": request["candidate"]}
    if request.get("incumbent") is not None:
        candidates["incumbent"] = request["incumbent"]
    for name, candidate in candidates.items():
        candidates[name] = _run_stage(
            "source",
            backend.validate_candidate,
            candidate,
            len(input_specs),
            len(output_specs),
        )
        if diagnostics is not None:
            diagnostics[name] = _candidate_diagnostics(candidates[name])
    objective = request.get("objective", "candidate")
    if objective not in {"candidate", "eager", "tensorrt"}:
        raise _EvaluationError("baseline", f"Unsupported objective: {objective}")

    if not torch.cuda.is_available():
        raise RuntimeError("Generated-kernel evaluation requires CUDA")
    device = torch.device(input_specs[0]["device"])
    if device.type != "cuda":
        raise ValueError("Generated-kernel evaluation requires CUDA inputs")
    torch.cuda.set_device(device)
    reference = torch.load(
        request["reference_path"], map_location=device, weights_only=False
    )
    if not isinstance(reference, torch.fx.GraphModule):
        raise TypeError("reference_path must contain a captured GraphModule")
    reference.eval()
    cases = _input_cases(
        input_specs,
        request["num_tests"],
        request["check_nonfinite"],
        seed=request.get("validation_seed"),
    )
    first = next(cases)
    expected = _outputs(reference(*(item.clone() for item in first)))
    _check_contract(expected, output_specs)
    _restore_metadata(reference, first, expected)
    functions: dict[str, Callable[..., Any]] = {"reference": reference}
    for name, candidate in candidates.items():
        with _compilation_diagnostics(
            diagnostics[name] if diagnostics is not None else None
        ):
            registration = _run_stage(
                "compile",
                backend.make_registration,
                candidate,
                input_specs,
                output_specs,
                directory / name,
            )
            functions[name] = _run_stage(
                "compile",
                _build_engine,
                reference,
                registration,
                first,
                len(output_specs),
            )
    if objective == "tensorrt":
        baseline_diagnostics = (
            {"compilation_scope": "tensorrt_reference_engine_build"}
            if diagnostics is not None
            else None
        )
        if diagnostics is not None:
            diagnostics["baseline"] = baseline_diagnostics
        with _compilation_diagnostics(baseline_diagnostics):
            functions["baseline"] = _run_stage(
                "baseline", _build_reference_engine, reference, first
            )
    tests = 0
    for inputs in itertools.chain((first,), cases):
        for name, function in functions.items():
            if name == "reference":
                continue
            try:
                _verify_case(
                    reference,
                    function,
                    tuple(item.clone() for item in inputs),
                    output_specs,
                    rtol=request.get("rtol"),
                    atol=request.get("atol"),
                )
            except Exception as error:
                raise _EvaluationError(
                    "baseline" if name == "baseline" else "correctness",
                    f"Correctness case {tests + 1} failed for {name}",
                ) from error
        tests += 1
    benchmark = _run_stage(
        "benchmark",
        _benchmark,
        functions,
        first,
        request["benchmark_iters"],
        request.get("benchmark_repeats", 5),
    )
    if objective == "eager":
        benchmark["baseline"] = dict(benchmark["reference"])
    result = {
        "status": "passed",
        "measurement": "cuda_graph",
        "latency_ms": benchmark["candidate"]["median_ms"],
        "reference_latency_ms": benchmark["reference"]["median_ms"],
        "reference_before_ms": benchmark["reference"]["samples_ms"][0],
        "reference_after_ms": benchmark["reference"]["samples_ms"][-1],
        "benchmark": benchmark,
        "objective": objective,
        "validation_phase": request.get("validation_phase", "search"),
        "tests": tests,
    }
    if "incumbent" in benchmark:
        result["incumbent_latency_ms"] = benchmark["incumbent"]["median_ms"]
    if "baseline" in benchmark:
        result["baseline_latency_ms"] = benchmark["baseline"]["median_ms"]
    return result


def _error_report(error: Exception, directory: Path, phase: str) -> dict[str, Any]:
    diagnostic = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    for path, replacement in (
        (directory.resolve(), "<evaluation>"),
        (Path.home(), "<home>"),
    ):
        diagnostic = diagnostic.replace(str(path), replacement)
    diagnostic = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", diagnostic)
    diagnostic = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", diagnostic)
    result: dict[str, Any] = {
        "status": "error",
        "stage": getattr(error, "stage", "compile"),
        "validation_phase": phase,
        "error": diagnostic[-16000:],
    }
    details = getattr(error, "diagnostics", None)
    if details is not None:
        result["diagnostics"] = details
    return result


def _write_result(path: Path, result: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    request: dict[str, Any] = {}
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        with torch.inference_mode():
            result = evaluate(request, args.request.parent)
    except Exception as error:
        result = _error_report(
            error, args.request.parent, request.get("validation_phase", "search")
        )
    _write_result(args.result, result)


if __name__ == "__main__":
    main()
