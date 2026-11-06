# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate generated kernels in a disposable process.

Search evaluations can screen a candidate: it is registered as the plugin an
engine would embed, but its kernel is launched directly, without building an
engine. Holdout evaluations always run it as a plugin inside a real TRT engine.

Failures carry the stage that owns them: ``setup`` (the request, device or
reference, including device memory running out at any stage; the author cannot
fix it), ``source``, ``compile``, ``engine``, ``correctness``, ``baseline``
(the TensorRT baseline) or ``benchmark``.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import logging
import math
import operator
import os
import re
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple, Sequence, TypeVar

import torch

from torch_tensorrt.logging import TRT_LOGGER

from .._plugins import _TensorBoundary
from ._reference import load_reference
from ._samples import check_samples, load_samples
from ._util import digest, read_json_object, scrub, write_json

_LOGGER = logging.getLogger(__name__)
_Result = TypeVar("_Result")

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
# torch.testing.assert_close's default (rtol, atol) for each supported dtype.
_TOLERANCES = {
    torch.float32: (1.3e-6, 1e-5),
    torch.float16: (1e-3, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-5),
}
# A different but equally careful evaluation order can land on the other side
# of the exact result, so allow a few times the eager reference's own error.
_ORACLE_ERROR_FACTOR = 4
# Eager errors within this many units in the last place are rounding only, as
# for elementwise operations; larger errors accumulate in reductions.
_ROUNDING_ULPS = 2
# (scale, offset) pairs whose magnitudes expose underflow, unguarded
# exponentials and cancellation.
_SCALED_CASES = ((1e-3, 0.0), (1e2, 0.0), (1.0, 1e2), (1.0, -1e2))
# Sentinel bytes on each side of a guarded tensor: wider than a typical tile
# overrun, such as a 1024-wide float32 block without a tail mask.
_GUARD_BYTES = 4096
# Padding byte patterns: 0xFF reads as NaN, and 0x7B as a large finite value,
# in float32, float16, and bfloat16 alike.
_GUARD_FILLS = (0xFF, 0x7B)
# Roughly 0.3-1 ms of device time: longer than the host takes to submit a
# graph replay, so the timing events bracket GPU work only.
_SPIN_CYCLES = 1_000_000
# Trial builds must place the whole region in one TensorRT engine on the GPU,
# whatever the user's own settings are.
_FORCED_COMPILE_OPTIONS = {
    "ir": "dynamo",
    "strict": True,
    # The early compiler size check also applies to fully supported graphs.
    "min_block_size": 1,
    "require_full_compilation": True,
    "offload_module_to_cpu": False,
}
# Exhausted device memory as PyTorch, the CUDA driver and runtime, Triton, and
# TensorRT report it: "out of memory", CUDA_ERROR_OUT_OF_MEMORY, "OutOfMemory".
_OUT_OF_MEMORY = re.compile(r"out[ _]?of[ _]?memory", re.IGNORECASE)


class _EvaluationError(RuntimeError):
    """Keep the failed executor stage separate from its diagnostic traceback."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.diagnostics: dict[str, Any] | None = None


def _run_stage(
    stage: str, function: Callable[..., _Result], *args: Any, **kwargs: Any
) -> _Result:
    with TRT_LOGGER.capture_errors() as logged:
        try:
            return function(*args, **kwargs)
        except _EvaluationError:
            raise
        except Exception as error:
            message = str(error)
            # TensorRT logs a failed allocation; the exception that follows,
            # such as "Failed to create execution context", may not say why.
            # Compiler errors quote the candidate's source after their first line.
            memory = next(
                (
                    text
                    for text in (message.partition("\n")[0], *logged)
                    if _OUT_OF_MEMORY.search(text)
                ),
                None,
            )
            if memory is None:
                raise _EvaluationError(stage, message) from error
            # Memory that other work holds says nothing about the candidate,
            # so, as a setup failure, it leaves the evaluation inconclusive.
            if memory not in message:
                message = f"{message} ({memory})"
            raise _EvaluationError("setup", message) from error


def _outputs(value: Any) -> tuple[torch.Tensor, ...]:
    values = tuple(value) if isinstance(value, (tuple, list)) else (value,)
    if not values or any(not isinstance(item, torch.Tensor) for item in values):
        raise ValueError("The reference must return a Tensor or flat tuple of Tensors")
    return values


def _dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(f"Unsupported generated-kernel dtype: {name}") from None


def _specs(items: Sequence[dict[str, Any]]) -> list[_TensorBoundary]:
    return [
        _TensorBoundary(
            tuple(item["shape"]), _dtype(item["dtype"]), torch.device(item["device"])
        )
        for item in items
    ]


def _check_contract(
    values: Sequence[torch.Tensor], specs: Sequence[_TensorBoundary]
) -> None:
    if len(values) != len(specs):
        raise ValueError(
            f"Produced {len(values)} outputs; the captured region has {len(specs)}"
        )
    for index, (value, spec) in enumerate(zip(values, specs)):
        actual = _TensorBoundary(tuple(value.shape), value.dtype, value.device)
        if actual != spec or not value.is_contiguous():
            raise ValueError(
                f"Output {index} is {value.dtype} {list(value.shape)} on "
                f"{value.device} (contiguous={value.is_contiguous()}); the captured "
                f"contract is contiguous {spec.dtype} {list(spec.shape)} on {spec.device}"
            )


def _input_cases(
    specs: Sequence[_TensorBoundary],
    num_tests: int,
    check_nonfinite: bool,
    seed: int,
    samples: Sequence[tuple[torch.Tensor, ...]] = (),
) -> Iterator[tuple[str, tuple[torch.Tensor, ...]]]:
    """Yield labelled input cases; the seed fixes every random value.

    ``samples`` are region inputs captured from real model batches; they
    follow the synthetic cases.
    """
    generator = torch.Generator(device=specs[0].device).manual_seed(seed)

    def random() -> tuple[torch.Tensor, ...]:
        inputs = tuple(
            torch.randn(
                spec.shape, dtype=spec.dtype, device=spec.device, generator=generator
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

    def rescaled(scale: float, offset: float) -> tuple[torch.Tensor, ...]:
        values = []
        for value in random():
            # Keep a product of two inputs finite: these cases test magnitude
            # handling, not trivial overflow.
            limit = math.sqrt(torch.finfo(value.dtype).max) / 2
            values.append((value * scale + offset).clamp_(-limit, limit))
        return tuple(values)

    for number in range(1, num_tests + 1):
        yield f"random {number}", random()
    for scale, offset in _SCALED_CASES:
        label = f"offset {offset:+g}" if offset else f"scaled x{scale:g}"
        yield label, rescaled(scale, offset)
    for constant in (0.0, -1.0, 1.0):
        yield f"constant {constant}", tuple(
            torch.full(spec.shape, constant, dtype=spec.dtype, device=spec.device)
            for spec in specs
        )
    if check_nonfinite:
        for index in range(len(specs)):
            # Exercise every input independently, including scalar tensors.
            for special in (math.nan, math.inf, -math.inf):
                inputs = random()
                inputs[index].reshape(-1)[0] = special
                yield f"input {index}[0] = {special}", inputs
    for number, sample in enumerate(samples, 1):
        yield f"sample {number}", tuple(
            value.to(spec.device) for value, spec in zip(sample, specs)
        )


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


def _compile(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    options: dict[str, Any],
) -> torch.nn.Module:
    import torch_tensorrt

    compiled = torch_tensorrt.compile(
        module, inputs=list(inputs), **{**options, **_FORCED_COMPILE_OPTIONS}
    )
    if not isinstance(compiled, torch.nn.Module):
        raise RuntimeError("TensorRT compilation did not return a module")
    return compiled


def _register_plugin(
    reference: torch.fx.GraphModule, registration: Any
) -> torch._ops.OpOverload:
    """Register a candidate as the plugin operator an engine would embed.

    This runs the plugin path's own compilation and checks, such as Triton's
    and cuTile's ahead-of-time compilation, without building an engine.
    """
    from torch_tensorrt.dynamo.region_compilation._plugins import register_region_plugin

    return register_region_plugin(
        reference, registration, region_id="generated_candidate"
    )


def _build_engine(
    reference: torch.fx.GraphModule,
    registration: Any,
    inputs: tuple[torch.Tensor, ...],
    num_outputs: int,
    options: dict[str, Any],
) -> torch.nn.Module:
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

    op = _register_plugin(reference, registration)
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
    compiled = _compile(torch.fx.GraphModule({}, graph).eval(), inputs, options)
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
    reference: torch.fx.GraphModule,
    inputs: tuple[torch.Tensor, ...],
    options: dict[str, Any],
) -> torch.nn.Module:
    """Compile the unannotated region and reject any remaining eager computation.

    Only the ``tensorrt`` objective, a speedup over TensorRT, needs this engine,
    and only to time it; candidates are always checked against eager PyTorch.
    """
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

    compiled = _compile(copy.deepcopy(reference), inputs, options)
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


@dataclass(frozen=True)
class _Timer:
    """Times one replay of a CUDA graph holding ``iterations`` calls."""

    graph: torch.cuda.CUDAGraph
    stream: torch.cuda.Stream
    iterations: int
    # Replays reuse kernels and buffers owned by the captured function and read
    # the inputs in place, so the timer keeps both alive.
    captured: tuple[Callable[..., Any], tuple[torch.Tensor, ...]]

    def __call__(self) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            # Queue the replay behind device work so its submission is not timed.
            torch.cuda._sleep(_SPIN_CYCLES)
            start.record()
            self.graph.replay()
            end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) / self.iterations


def _capture_timer(
    function: Callable[..., Any], inputs: tuple[torch.Tensor, ...], iterations: int
) -> Callable[[], float]:
    """Prepare a graph once; each returned measurement excludes Python launch gaps."""
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
    return _Timer(graph, stream, iterations, (function, inputs))


def _benchmark(
    timers: dict[str, Callable[[], float]], repeats: int
) -> dict[str, dict[str, Any]]:
    """Sample every arm in turn, bracketed by reference samples that expose drift."""
    samples: dict[str, list[float]] = {name: [] for name in timers}
    names = list(timers)

    def measure(name: str) -> None:
        sample = timers[name]()
        if not math.isfinite(sample) or sample <= 0:
            raise RuntimeError(
                f"Benchmark of {name} did not produce a positive finite latency"
            )
        samples[name].append(sample)

    measure("reference")
    # Alternate the order so the same arm does not always benefit from drift.
    for repeat in range(repeats):
        for name in names if repeat % 2 == 0 else reversed(names):
            measure(name)
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


class _Expected(NamedTuple):
    """The reference outputs one input case is checked against."""

    eager: tuple[torch.Tensor, ...]
    # The reference run on float64 inputs, or None when explicit tolerances
    # apply or an operation cannot run in float64.
    oracle: tuple[torch.Tensor, ...] | None


def _float64_reference(reference: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Copy the reference so that it also runs on float64 inputs.

    softmax and log_softmax with dtype=float32 capture as a half-to-float
    variant that rejects float64 inputs; its float64 form computes the same.
    """
    exact = copy.deepcopy(reference)
    for node in exact.graph.nodes:
        if (
            node.target
            in (torch.ops.aten._softmax.default, torch.ops.aten._log_softmax.default)
            and len(node.args) == 3
            and node.args[2] is True
        ):
            node.args = (*node.args[:2], False)
    exact.recompile()
    return exact


def _expected(
    reference: Callable[..., Any],
    inputs: tuple[torch.Tensor, ...],
    specs: Sequence[_TensorBoundary],
    *,
    exact_reference: Callable[..., Any] | None,
) -> _Expected:
    eager = _outputs(reference(*(value.clone() for value in inputs)))
    _check_contract(eager, specs)
    if exact_reference is None:
        return _Expected(eager, None)
    try:
        # Explicit dtype conversions in the graph still apply.
        exact = _outputs(exact_reference(*(value.double() for value in inputs)))
        if [value.shape for value in exact] != [value.shape for value in eager]:
            raise ValueError("float64 outputs have different shapes")
    except Exception as error:
        _LOGGER.warning(
            "Comparing with the eager reference at default tolerances; "
            "the float64 reference failed: %s",
            error,
        )
        return _Expected(eager, None)
    return _Expected(eager, tuple(value.double() for value in exact))


def _same(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left == right) | (left.isnan() & right.isnan())


def _same_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(
        torch.equal(
            left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8)
        )
    )


def _oracle_mismatch(
    actual: torch.Tensor, eager: torch.Tensor, oracle: torch.Tensor
) -> str | None:
    """Describe the worst element the float64 oracle rejects, if any.

    An element passes when it equals the eager value or the correctly rounded
    float64 value (NaN matching NaN), or lies within atol + rtol*|exact| plus
    a multiple of eager PyTorch's own error. Where eager only rounds, that is a
    few units in the last place of the element itself; where eager accumulates
    error, as reductions and contractions do, it is eager's largest error over
    the output, since a different summation order errs elsewhere. Eager may
    overflow where the exact result does not, so a candidate may either
    reproduce that or be accurate.
    """
    actual64, eager64 = actual.double(), eager.double()
    target = torch.where(oracle.isfinite(), oracle, eager64)
    finite = eager64.isfinite() & oracle.isfinite()
    errors = torch.where(finite, (eager64 - oracle).abs(), 0.0)
    limits = torch.finfo(eager.dtype)
    ulp = limits.eps * target.abs().clamp_min(limits.tiny)
    own_error = float(errors.max())
    if bool((errors <= _ROUNDING_ULPS * ulp).all()):
        slack, basis = _ORACLE_ERROR_FACTOR * ulp, "units in the last place"
    else:
        slack, basis = _ORACLE_ERROR_FACTOR * own_error, "eager's largest error"
    rtol, atol = _TOLERANCES[eager.dtype]
    allowance = atol + rtol * target.abs() + slack
    error = (actual64 - target).abs()
    accepted = (
        (target.isfinite() & (error <= allowance))
        | _same(actual64, eager64)
        | _same(actual64, oracle.to(eager.dtype).double())
    )
    if bool(accepted.all()):
        return None
    excess = torch.where(
        accepted, 0.0, (error / allowance).nan_to_num(nan=math.inf, posinf=math.inf)
    )
    worst = int(excess.argmax())
    position = tuple(
        int(i) for i in torch.unravel_index(torch.tensor(worst), actual.shape)
    )
    wanted = target.reshape(-1)[worst]
    detail = (
        f"got {float(actual64.reshape(-1)[worst]):.6g} at index {position}, "
        f"expected {float(wanted.to(eager.dtype)):.6g}"
    )
    if bool(wanted.isfinite()):
        detail += (
            f" (error {float(error.reshape(-1)[worst]):.3g} exceeds the allowance "
            f"{float(allowance.reshape(-1)[worst]):.3g}: atol {atol:g} + rtol {rtol:g}"
            f"*|expected| + {_ORACLE_ERROR_FACTOR}x {basis}; eager's own largest "
            f"float64 error is {own_error:.3g})"
        )
    return f"{detail}; {int((~accepted).sum())} of {actual.numel()} elements differ"


def _check_outputs(
    actual: Sequence[torch.Tensor],
    expected: _Expected,
    where: str,
    *,
    rtol: float | None,
    atol: float | None,
) -> None:
    for index, (value, eager) in enumerate(zip(actual, expected.eager)):
        if expected.oracle is not None:
            problem = _oracle_mismatch(value, eager, expected.oracle[index])
        else:
            try:
                torch.testing.assert_close(
                    value, eager, rtol=rtol, atol=atol, equal_nan=True
                )
                problem = None
            except AssertionError as error:
                problem = str(error)
        if problem is not None:
            raise _EvaluationError(
                "correctness", f"Output {index} is wrong for {where}: {problem}"
            )


def _verify_case(
    function: Callable[..., Any],
    inputs: tuple[torch.Tensor, ...],
    expected: _Expected,
    specs: Sequence[_TensorBoundary],
    label: str,
    *,
    rtol: float | None,
    atol: float | None,
) -> None:
    arguments = tuple(value.clone() for value in inputs)
    actual = _outputs(function(*arguments))
    _check_contract(actual, specs)
    for index, (argument, value) in enumerate(zip(arguments, inputs)):
        if not _same_bits(argument, value):
            raise _EvaluationError(
                "correctness", f"Input {index} was modified on case {label!r}"
            )
    _check_outputs(actual, expected, f"case {label!r}", rtol=rtol, atol=atol)


def _check_guard_bands(
    launch: Callable[[list[torch.Tensor], list[torch.Tensor]], None],
    inputs: tuple[torch.Tensor, ...],
    expected: _Expected,
    label: str,
    *,
    rtol: float | None,
    atol: float | None,
) -> None:
    """Launch on interior views of padded buffers and reject accesses outside them.

    The trial engine gives each output its own allocation with allocator slack,
    so its results cannot show a write past the end of a tensor. Two launches
    with different padding must agree bit for bit, so an out-of-bounds read
    that changes an output is caught even when an operation such as max drops
    the NaN padding of the first launch; a read whose value is discarded is not.
    """
    where = f"the guard-band launch on case {label!r}"
    results = []
    for fill in _GUARD_FILLS:
        guarded = {
            kind: [_padded(value, fill) for value in values]
            for kind, values in (("input", inputs), ("output", expected.eager))
        }
        for (_, view), value in zip(guarded["input"], inputs):
            view.copy_(value)
        for _, view in guarded["output"]:
            view.fill_(math.nan)
        launch(
            [view for _, view in guarded["input"]],
            [view for _, view in guarded["output"]],
        )
        torch.cuda.synchronize()
        for kind, buffers in guarded.items():
            for index, (buffer, view) in enumerate(buffers):
                start = view.storage_offset()
                for side, band in (
                    ("before", buffer[:start]),
                    ("past", buffer[start + view.numel() :]),
                ):
                    if not bool((band.view(torch.uint8) == fill).all()):
                        raise _EvaluationError(
                            "correctness",
                            f"Out-of-bounds write {side} {kind} {index} in {where}",
                        )
        for index, ((_, view), value) in enumerate(zip(guarded["input"], inputs)):
            if not _same_bits(view, value):
                raise _EvaluationError(
                    "correctness", f"Input {index} was modified in {where}"
                )
        results.append([view for _, view in guarded["output"]])
    for index, (first, second) in enumerate(zip(*results)):
        if not _same_bits(first, second):
            raise _EvaluationError(
                "correctness",
                f"Output {index} reads memory outside its inputs in {where}: it "
                "changed when only the padding around the tensors changed",
            )
    _check_outputs(
        results[0],
        expected,
        f"{where} (outputs start as NaN, so NaN marks an unwritten element)",
        rtol=rtol,
        atol=atol,
    )


def _launcher(
    launch: Callable[..., None],
    candidate: dict[str, Any],
    output_specs: Sequence[_TensorBoundary],
    directory: Path,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Run a candidate through the backend's direct launch into fresh outputs.

    The launch runs the candidate's source with its launch settings, as the
    plugin does, but a backend may compile that source separately, as Triton
    and cuTile do, so only an engine runs the plugin's own binary.
    """

    def run(*inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = [
            torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            for spec in output_specs
        ]
        launch(candidate, list(inputs), outputs, directory)
        return tuple(outputs)

    return run


def _padded(like: torch.Tensor, fill: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a buffer filled with ``fill`` bytes and its interior view."""
    pad = -(-_GUARD_BYTES // like.element_size())
    buffer = torch.empty(like.numel() + 2 * pad, dtype=like.dtype, device=like.device)
    buffer.view(torch.uint8).fill_(fill)
    return buffer, buffer[pad : pad + like.numel()].view(like.shape)


def evaluate(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    diagnostics: dict[str, Any] | None = {} if request["diagnostics"] else None
    try:
        result = _run_stage("setup", _evaluate, request, directory, diagnostics)
    except _EvaluationError as error:
        error.diagnostics = diagnostics
        raise
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    return result


def _candidate_diagnostics(candidate: dict[str, Any], screened: bool) -> dict[str, Any]:
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
        # A screened kernel's direct launch may compile it again on first use,
        # outside this timing; a failure there still marks compilation failed.
        "compilation_scope": (
            "backend_and_plugin_registration"
            if screened
            else "backend_registration_and_tensorrt_engine_build"
        ),
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
    if backend.version != request["backend_version"]:
        raise ValueError("Backend version changed between authoring and evaluation")
    if digest(backend.proposal_schema) != request["proposal_schema_sha256"]:
        raise ValueError(
            "Backend proposal schema changed between authoring and evaluation"
        )


def _evaluate(
    request: dict[str, Any],
    directory: Path,
    diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Check one candidate; anything not attributed to a stage is ``setup``."""
    from ._backends import get_backend

    backend = get_backend(request["backend"])
    _check_backend_identity(backend, request)
    input_specs, output_specs = _specs(request["inputs"]), _specs(request["outputs"])
    arity = len(input_specs), len(output_specs)
    candidate = _run_stage(
        "source", backend.validate_candidate, request["candidate"], *arity
    )
    launch = getattr(backend, "launch", None)
    # Only an engine proves the kernel where it will run, so holdout checks
    # always build one.
    screened = (
        launch is not None
        and request["validation_phase"] == "search"
        and request.get("screening") == "launch"
    )
    if diagnostics is not None:
        diagnostics["candidate"] = _candidate_diagnostics(candidate, screened)

    device = input_specs[0].device
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("Generated-kernel evaluation requires a CUDA device")
    torch.cuda.set_device(device)
    # JSON data, not a pickle, so a shared or copied run cannot run code here.
    reference = load_reference(request["reference_path"])
    reference.eval()
    rtol, atol = request["rtol"], request["atol"]
    samples: list[tuple[torch.Tensor, ...]] = []
    if request.get("samples_path") is not None:
        # weights_only loading, so a shared or copied run cannot run code here.
        samples = load_samples(request["samples_path"])
        check_samples(samples, input_specs)
    cases = _input_cases(
        input_specs,
        request["num_tests"],
        request["check_nonfinite"],
        request["validation_seed"],
        samples,
    )
    first_label, first = next(cases)
    exact_reference = _float64_reference(reference) if rtol is None else None
    first_expected = _expected(
        reference, first, output_specs, exact_reference=exact_reference
    )
    if first_expected.oracle is None:
        exact_reference = None
    _restore_metadata(reference, first, first_expected.eager)
    iterations = request["benchmark_iters"]
    timers = {"reference": _capture_timer(reference, first, iterations)}

    options = request["compile_options"]
    # A screen decides no speedup objective, so it builds the baseline engine
    # only when the controller asks: a baseline that TensorRT cannot build
    # must still end the search early.
    if request["objective"] == "tensorrt" and (
        not screened or request.get("check_baseline")
    ):
        # A timing arm only: TensorRT's own numerics are not held to the
        # candidate's tolerance.
        details = None
        if diagnostics is not None:
            details = diagnostics["baseline"] = {
                "compilation_scope": "tensorrt_reference_engine_build"
            }
        with _compilation_diagnostics(details):
            baseline = _run_stage(
                "baseline", _build_reference_engine, reference, first, options
            )
        timers["baseline"] = _run_stage(
            "baseline", _capture_timer, baseline, first, iterations
        )

    def build(name: str, proposal: dict[str, Any]) -> Callable[..., Any]:
        with _compilation_diagnostics(
            diagnostics[name] if diagnostics is not None else None
        ):
            registration = _run_stage(
                "compile",
                backend.make_registration,
                proposal,
                input_specs,
                output_specs,
                directory / name,
            )
            if screened and launch is not None:
                # Registering the plugin compiles what an engine would embed,
                # so the screen rejects what only the plugin path refuses.
                _run_stage("compile", _register_plugin, reference, registration)
                return _launcher(launch, proposal, output_specs, directory / name)
            engine: Callable[..., Any] = _run_stage(
                "compile",
                _build_engine,
                reference,
                registration,
                first,
                len(output_specs),
                options,
            )
            return engine

    execute = build("candidate", candidate)
    if launch is not None:
        try:
            _run_stage(
                "correctness",
                _check_guard_bands,
                # A launch that raises, such as the first one of a kernel that
                # fails to JIT-compile, did not compute anything to check.
                lambda inputs, outputs: _run_stage(
                    "compile",
                    launch,
                    candidate,
                    inputs,
                    outputs,
                    directory / "candidate",
                ),
                first,
                first_expected,
                first_label,
                rtol=rtol,
                atol=atol,
            )
        except _EvaluationError as error:
            if error.stage == "compile" and diagnostics is not None:
                diagnostics["candidate"]["compilation_status"] = "failed"
            raise
    tests = 0
    for label, inputs in itertools.chain([(first_label, first)], cases):
        expected = (
            first_expected
            if inputs is first
            else _expected(
                reference, inputs, output_specs, exact_reference=exact_reference
            )
        )
        _run_stage(
            "correctness",
            _verify_case,
            execute,
            inputs,
            expected,
            output_specs,
            label,
            rtol=rtol,
            atol=atol,
        )
        tests += 1
    timers["candidate"] = _run_stage(
        "benchmark", _capture_timer, execute, first, iterations
    )

    incumbent_error = None
    if request.get("incumbent") is not None:
        # A timing arm only: it passed its own checks when it was selected.
        try:
            incumbent = _run_stage(
                "source", backend.validate_candidate, request["incumbent"], *arity
            )
            if diagnostics is not None:
                diagnostics["incumbent"] = _candidate_diagnostics(incumbent, screened)
            timers["incumbent"] = _run_stage(
                "benchmark",
                _capture_timer,
                build("incumbent", incumbent),
                first,
                iterations,
            )
        except _EvaluationError as error:
            incumbent_error = f"{error.stage}: {scrub(str(error), directory)}"[:2000]
    benchmark = _run_stage(
        "benchmark", _benchmark, timers, request["benchmark_repeats"]
    )
    if request["objective"] == "eager":
        benchmark["baseline"] = dict(benchmark["reference"])
    result = {
        "status": "passed",
        "execution": "launch" if screened else "engine",
        "measurement": "cuda_graph",
        "latency_ms": benchmark["candidate"]["median_ms"],
        "reference_latency_ms": benchmark["reference"]["median_ms"],
        "reference_before_ms": benchmark["reference"]["samples_ms"][0],
        "reference_after_ms": benchmark["reference"]["samples_ms"][-1],
        "benchmark": benchmark,
        "objective": request["objective"],
        "validation_phase": request["validation_phase"],
        "tests": tests,
        "comparison": (
            "explicit_tolerances"
            if rtol is not None
            else "eager_defaults" if exact_reference is None else "float64_reference"
        ),
        "guard_check": "unavailable" if launch is None else "passed",
    }
    if "incumbent" in benchmark:
        result["incumbent_latency_ms"] = benchmark["incumbent"]["median_ms"]
    if incumbent_error is not None:
        result["incumbent_error"] = incumbent_error
    if "baseline" in benchmark:
        result["baseline_latency_ms"] = benchmark["baseline"]["median_ms"]
    return result


def _error_report(error: Exception, directory: Path, phase: str) -> dict[str, Any]:
    trace = "".join(traceback.format_exception(error))
    result: dict[str, Any] = {
        "status": "error",
        "stage": getattr(error, "stage", "setup"),
        "validation_phase": phase,
        "error": scrub(trace, directory)[-16000:],
    }
    details = getattr(error, "diagnostics", None)
    if details is not None:
        result["diagnostics"] = details
    return result


def _disable_core_dumps() -> None:
    """A crashing candidate must not leave a multi-gigabyte core in the cache."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass


def _publish(path: Path, result: dict[str, Any], exit: Callable[[int], Any]) -> None:
    write_json(path, result)
    sys.stdout.flush()
    sys.stderr.flush()
    exit(0)


def main(exit: Callable[[int], Any] = os._exit) -> None:
    """Write the result, then leave without tearing down CUDA or TensorRT.

    After a sticky CUDA error, such as a candidate's illegal memory access,
    destroying the evaluation's engines can crash the process, so the result
    is published while they are still alive.
    """
    _disable_core_dumps()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    request: dict[str, Any] = {}
    try:
        request = read_json_object(args.request)
        with torch.inference_mode():
            result = evaluate(request, args.request.parent)
    except Exception as error:
        report = _error_report(
            error, args.request.parent, request.get("validation_phase", "search")
        )
        _publish(args.result, report, exit)
        return
    _publish(args.result, result, exit)


if __name__ == "__main__":
    main()
