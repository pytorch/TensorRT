# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate generated kernels in a disposable process.

Search evaluations can screen a candidate: it is registered as the plugin an
engine would embed, but its kernel is launched directly, without building an
engine. Holdout evaluations always run it as a plugin inside a real TRT engine.

Failures carry the stage that owns them: ``setup`` (the request, device or
reference, including device memory running out outside the candidate's own
compilation, launches and runs; the author cannot fix it), ``source``,
``compile``, ``resources`` (device memory ran out while compiling, registering,
launching or running the candidate), ``engine``, ``correctness``, ``baseline``
(the TensorRT baseline) or ``benchmark``. A setup failure that evaluating
again cannot change is reported with ``retryable: false``.
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
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple, Sequence, TypeVar

import torch

from torch_tensorrt.logging import TRT_LOGGER

from .._plugins import _TensorBoundary
from ._reference import load_reference
from ._samples import (
    MAGNITUDE_SCALES,
    check_samples,
    integer_valued,
    load_samples,
    magnitude_limit,
)
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
# The error over a whole output may be at most this many times eager
# PyTorch's own: a small systematic error, such as a 1% scale, stays inside
# the elementwise allowance but not inside the total (measured on attention).
_AGGREGATE_FACTOR = 2
# ... or this many times a floor, whichever is larger: one unit roundoff of the
# exact values in the output dtype, but at least this relative error, and at
# least atol per element. Float32 kernels that approximate exp or order a long
# recurrence differently can err several times more than eager PyTorch,
# though far less than any error worth rejecting (measured on a Mamba scan:
# 2.5x eager at 2.3e-7, and 3.2x at 5.6e-6 where atol is 100 times larger).
_AGGREGATE_FLOOR = 2**-20
# Dtypes the oracle evaluates in float64 instead; integer and bool dtypes keep
# their exact semantics.
_PROMOTED_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})
# Operations that reinterpret a tensor's bits, which no wider dtype preserves.
_BIT_VIEWS = (torch.ops.aten.view.dtype, torch.ops.prims.view_of_dtype.default)
# (scale, offset) pairs whose magnitudes expose underflow, unguarded
# exponentials and cancellation.
_SCALED_CASES = (
    *((scale, 0.0) for scale in MAGNITUDE_SCALES),
    (1.0, 1e2),
    (1.0, -1e2),
)
# Every input takes each of these values in turn, where its sign allows.
_CONSTANTS = (0.0, -1.0, 1.0)
# Values injected at element 0 of each input in turn.
_NONFINITE = (math.nan, math.inf, -math.inf)
# A case checks too little when eager PyTorch's output is NaN or infinite at
# more than this fraction of any output's elements.
_INCONCLUSIVE_FRACTION = 0.5
# Robustness cases, the magnitude cases that take inputs beyond the sampled
# range, allow an error of this fraction of the exact output's L2 norm ...
_ROBUSTNESS_FLOOR = 1e-2
# ... and, per element, of this fraction of its largest exact magnitude:
# enough for the cancellation correct kernels show there (up to 1.2% of it at
# single elements, measured on cached Nemotron and Qwen3.5 kernels), yet an
# error comparable to the output's scale, which the norm of a large output
# would average away, still fails.
_ROBUSTNESS_ELEMENT_FLOOR = 0.1
# Host bytes of expected outputs that an evaluation computes before building
# anything; above it, each case's expected outputs are computed in turn.
_EXPECTATION_BYTES = 8 << 30
# Sentinel bytes on each side of a guarded tensor: wider than a typical tile
# overrun, such as a 1024-wide float32 block without a tail mask.
_GUARD_BYTES = 4096
# The alignment of 128-bit loads and stores, which KernelGenerator.aligned_io
# promises every plugin pointer; without it a pointer is only aligned to its
# element size, and direct launches use such pointers (see _evaluate).
_VECTOR_BYTES = 16
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
# Float64 score matrices that PyTorch's math attention kernel holds at once,
# beyond the memory eager PyTorch's fused kernels need: 1.9 to 2.5 measured
# for causal, masked and grouped-query attention on an RTX 5090.
_ATTENTION_SCORE_COPIES = 3


class _EvaluationError(RuntimeError):
    """Keep the failed executor stage separate from its diagnostic traceback.

    ``retryable=False`` marks a failure that evaluating again cannot change,
    such as a reference that no oracle can evaluate.
    """

    def __init__(self, stage: str, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.stage = stage
        self.retryable = retryable
        self.diagnostics: dict[str, Any] | None = None


def _run_stage(
    stage: str,
    function: Callable[..., _Result],
    *args: Any,
    out_of_memory: str | None = None,
    **kwargs: Any,
) -> _Result:
    """Run ``function``, attributing its failure to ``stage``.

    Device memory running out is ``out_of_memory``'s: by default "resources"
    in "compile", where the candidate's own compilation and launches run, and
    "setup" elsewhere.
    """
    if out_of_memory is None:
        out_of_memory = "resources" if stage == "compile" else "setup"
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
            # The candidate's own compilation and launches run as "compile",
            # and its runs during correctness checks as "resources": running
            # out there is the candidate's to answer for.
            if memory not in message:
                message = f"{message} ({memory})"
            raise _EvaluationError(out_of_memory, message) from error


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


class _Domain(NamedTuple):
    """The values that captured samples show for one region input."""

    # The smallest and largest finite sample values; None when there are none.
    low: float | None
    high: float | None
    # How synthetic cases vary the input (see _input_cases):
    # "copied": no finite sample value, so every case takes a sample's own;
    # "constant": one finite value, such as a weight of ones or a scalar
    # setting; "mask": the values 0 and 1; "integer": other integer values,
    # such as positions and bounds; "small": fewer than 8 elements of other
    # values; "continuous": the rest.
    kind: str
    # Every finite sample value is an integer, so magnitude cases leave it.
    integer: bool = False
    # For a mask, the fraction of ones among its finite sample values.
    density: float | None = None
    # For a mask or an integer input: every sample's elements, flattened,
    # never decrease, as a window's [start, end] bounds do.
    monotone: bool = False


def _domains(
    samples: Sequence[Sequence[torch.Tensor]], specs: Sequence[_TensorBoundary]
) -> list[_Domain]:
    """Describe each input's finite range and kind over all samples."""
    domains = []
    for index, spec in enumerate(specs):
        values = [sample[index] for sample in samples]
        present = [
            finite
            for finite in (value[value.isfinite()] for value in values)
            if finite.numel()
        ]
        if not present:
            domains.append(_Domain(None, None, "copied", True))
            continue
        low = min(float(value.min()) for value in present)
        high = max(float(value.max()) for value in present)
        integer = all(integer_valued(value) for value in values)
        density = None
        if low == high:
            kind = "constant"
        elif integer and low == 0 and high == 1:
            kind = "mask"
            density = sum(float(value.float().sum()) for value in present) / sum(
                value.numel() for value in present
            )
        elif integer:
            kind = "integer"
        elif math.prod(spec.shape) < 8:
            kind = "small"
        else:
            kind = "continuous"
        monotone = kind in ("mask", "integer") and all(
            value.numel() > 1
            and bool((value.reshape(-1)[1:] >= value.reshape(-1)[:-1]).all())
            for value in values
        )
        domains.append(_Domain(low, high, kind, integer, density, monotone))
    return domains


def _signs(domain: _Domain) -> tuple[bool, bool]:
    """Whether an input may take negative and positive values.

    Its samples decide; an input whose samples are all zero may take either.
    """
    if domain.low is None or domain.high is None or domain.low == domain.high == 0:
        return True, True
    return domain.low < 0, domain.high > 0


def _as_input(values: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Round varied values to ``source``'s dtype, keeping its non-finite values.

    Values beyond the dtype's range become its largest finite magnitude:
    non-finite inputs are the business of the injection cases.
    """
    limit = torch.finfo(source.dtype).max
    varied = values.clamp(-limit, limit).to(source.dtype)
    return torch.where(source.isfinite(), varied, source)


def _input_cases(
    specs: Sequence[_TensorBoundary],
    num_tests: int,
    check_nonfinite: bool,
    seed: int,
    samples: Sequence[tuple[torch.Tensor, ...]] = (),
    injected: dict[str, int] | None = None,
    originals: dict[str, float] | None = None,
    robust: set[str] | None = None,
) -> Iterator[tuple[str, tuple[torch.Tensor, ...]]]:
    """Yield labelled input cases; the seed fixes every random value.

    ``samples`` are region inputs captured from real model batches; they
    follow the synthetic cases. Without them, random inputs are standard
    normal. With them, every synthetic case starts from a sample, in turn:

    - Random cases scale each continuous value by exp(z / 2), z standard
      normal, which keeps its sign and zeros, and clamp it into the input's
      sampled range, so that constraints the model maintains, such as a
      negative decay, hold. The first random case drives the guard bands,
      engine builds and timing, so it keeps every other input as the sample
      has it. The other cases vary those too: integer-valued inputs, such as
      positions and bounds, become random integers within the sampled range;
      0/1 masks become random masks with the sample's density of ones; both
      are sorted when every sample's elements never decrease, so a window's
      bounds keep start <= end; and
      constant inputs, such as a weight of ones or a scalar setting, and
      inputs of fewer than 8 elements are scaled by exp(z / 2), without the
      clamp; their zeros become standard normal values of the input's sign.
    - Magnitude cases scale or offset every input that is not
      integer-valued, keeping its sign: an offset of +100 applies only to
      inputs with positive values and -100 only to inputs with negative
      ones. Integer-valued inputs keep the random base case's values.
      Those that scale by 1e2 or offset an input leave the sampled range: they
      are robustness cases, which ``robust`` collects (see _check_outputs).
    - Constant cases set every input to the constant, if its sign allows.

    Non-finite sample values stay in place. ``injected`` maps the label of
    each case with a non-finite value set at element 0 of one input to that
    input's index, and ``originals`` maps it to the finite value it replaced,
    as the case is yielded.
    """
    generator = torch.Generator(device=specs[0].device).manual_seed(seed)
    domains = _domains(samples, specs) if samples else None
    draws = itertools.count()

    def anchor() -> tuple[torch.Tensor, ...]:
        """Copy the next sample in turn to the inputs' device."""
        sample = samples[next(draws) % len(samples)]
        return tuple(
            value.to(spec.device, copy=True) for value, spec in zip(sample, specs)
        )

    def draw(spec: _TensorBoundary, uniform: bool = False) -> torch.Tensor:
        sample = torch.rand if uniform else torch.randn
        return sample(
            spec.shape, device=spec.device, generator=generator, dtype=torch.float64
        )

    def vary(
        value: torch.Tensor, spec: _TensorBoundary, domain: _Domain, every: bool
    ) -> torch.Tensor:
        """Vary one sample input; ``every`` varies more than continuous ones."""
        if domain.kind == "copied" or (domain.kind != "continuous" and not every):
            return value
        assert domain.low is not None and domain.high is not None
        if domain.kind in ("mask", "integer"):
            if domain.kind == "mask":
                assert domain.density is not None
                varied = (draw(spec, uniform=True) < domain.density).double()
            else:
                span = domain.high - domain.low + 1
                varied = torch.floor(draw(spec, uniform=True) * span) + domain.low
                varied = varied.clamp_(max=domain.high)
            if domain.monotone:
                # Keep the samples' order, so that a window's bounds keep
                # start <= end rather than being empty half the time.
                varied = varied.reshape(-1).sort().values.reshape(varied.shape)
        else:
            varied = value.double() * torch.exp(draw(spec) / 2)
            if domain.kind == "continuous":
                varied = varied.clamp_(domain.low, domain.high)
            else:
                negative, positive = _signs(domain)
                zeros = draw(spec)
                if not negative:
                    zeros = zeros.abs()
                elif not positive:
                    zeros = -zeros.abs()
                varied = torch.where(value == 0, zeros, varied)
        return _as_input(varied, value)

    def random(every: bool = True) -> tuple[torch.Tensor, ...]:
        if domains is not None:
            return tuple(
                vary(value, spec, domain, every)
                for value, spec, domain in zip(anchor(), specs, domains)
            )
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

    def allowed(domain: _Domain, value: float) -> bool:
        """Whether a constant or offset keeps an input's observed sign."""
        negative, positive = _signs(domain)
        return (value <= 0 or positive) and (value >= 0 or negative)

    def rescaled(scale: float, offset: float) -> tuple[tuple[torch.Tensor, ...], bool]:
        """The case, and whether it moved some input beyond its sampled range."""
        values, beyond = [], False
        for index, value in enumerate(random()):
            # Keep a product of two inputs finite: these cases test magnitude
            # handling, not trivial overflow.
            limit = magnitude_limit(value.dtype)
            if domains is not None:
                domain = domains[index]
                if domain.kind != "copied" and not domain.integer:
                    if allowed(domain, offset):
                        varied = (value.double() * scale + offset).clamp_(-limit, limit)
                        value = _as_input(varied, value)
                        beyond |= scale > 1 or offset != 0
                values.append(value)
                continue
            values.append((value * scale + offset).clamp_(-limit, limit))
        return tuple(values), beyond

    def constants(constant: float) -> tuple[torch.Tensor, ...]:
        if domains is None:
            return tuple(
                torch.full(spec.shape, constant, dtype=spec.dtype, device=spec.device)
                for spec in specs
            )
        return tuple(
            (
                _as_input(
                    torch.full(
                        spec.shape, constant, dtype=torch.float64, device=spec.device
                    ),
                    value,
                )
                if domain.kind != "copied" and allowed(domain, constant)
                else value
            )
            for value, spec, domain in zip(random(), specs, domains)
        )

    for number in range(1, num_tests + 1):
        yield f"random {number}", random(every=number > 1)
    for scale, offset in _SCALED_CASES:
        label = f"offset {offset:+g}" if offset else f"scaled x{scale:g}"
        inputs, beyond = rescaled(scale, offset)
        if beyond and robust is not None:
            robust.add(label)
        yield label, inputs
    for constant in _CONSTANTS:
        yield f"constant {constant}", constants(constant)
    if check_nonfinite:
        for index in range(len(specs)):
            # Exercise every input independently, including scalar tensors.
            for special in _NONFINITE:
                inputs = random()
                element = inputs[index].reshape(-1)
                label = f"input {index}[0] = {special}"
                if originals is not None:
                    originals[label] = float(element[0])
                element[0] = special
                if injected is not None:
                    injected[label] = index
                yield label, inputs
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

    # Eager PyTorch, with TF32 off when it is compared with the oracle.
    eager: tuple[torch.Tensor, ...]
    # The reference evaluated without rounding, as float64 values, or None
    # when explicit tolerances apply.
    oracle: tuple[torch.Tensor, ...] | None
    # Eager PyTorch again with TF32 matmuls and convolutions, when
    # fp32_matmul_precision="high" allows them.
    relaxed: tuple[torch.Tensor, ...] | None = None
    # How the oracle was evaluated, such as "float64 on cuda:0".
    basis: str | None = None
    # For a case with an injected non-finite input, the output elements that
    # it reaches, where a candidate may propagate a non-finite value.
    propagated: tuple[torch.Tensor, ...] | None = None
    # For a case with an injected NaN, the output elements that depend on the
    # injected element and turn non-finite with it; the candidate must be
    # non-finite at one of them at least, or show that it reads the element.
    # None when there are none.
    required: tuple[torch.Tensor, ...] | None = None
    # With ``required``: the input's index, the element's original value and
    # another value that changes those outputs.
    injection: tuple[int, float, float] | None = None
    # A robustness case, judged by _robustness_mismatch.
    robustness: bool = False

    def to(self, device: torch.device | str) -> _Expected:
        def move(
            values: tuple[torch.Tensor, ...] | None,
        ) -> tuple[torch.Tensor, ...] | None:
            return None if values is None else tuple(v.to(device) for v in values)

        return self._replace(
            eager=tuple(value.to(device) for value in self.eager),
            oracle=move(self.oracle),
            relaxed=move(self.relaxed),
            propagated=move(self.propagated),
            required=move(self.required),
        )


class _Unpromotable(ValueError):
    """The reference depends on a dtype's bits, so no oracle can widen it."""


def _promoted(value: Any, dtype: torch.dtype, device: torch.device | None) -> Any:
    if isinstance(value, torch.dtype):
        return dtype if value in _PROMOTED_DTYPES else value
    if isinstance(value, torch.device) and device is not None:
        return device
    if isinstance(value, list):
        return [_promoted(item, dtype, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_promoted(item, dtype, device) for item in value)
    return value


def _float64_reference(
    reference: torch.fx.GraphModule,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.fx.GraphModule:
    """Copy the reference so that it computes in ``dtype`` without rounding.

    Every floating dtype argument (casts, factories, and reductions' ``dtype``)
    becomes ``dtype``, so the body's own float16, bfloat16 and float32
    conversions round nothing; integer and bool dtypes keep their meaning. A
    ``device`` replaces the captured devices, to run the copy there. softmax
    and log_softmax with dtype=float32 capture as a half-to-float variant that
    rejects wider inputs; its other form computes the same.
    """
    exact = copy.deepcopy(reference)
    for node in exact.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target in _BIT_VIEWS:
            raise _Unpromotable(
                f"{node.target} reinterprets the bits of a tensor, which no "
                "wider dtype preserves"
            )
        if (
            node.target
            in (torch.ops.aten._softmax.default, torch.ops.aten._log_softmax.default)
            and len(node.args) == 3
            and node.args[2] is True
        ):
            node.args = (*node.args[:2], False)
        node.args = _promoted(node.args, dtype, device)
        node.kwargs = {
            key: _promoted(value, dtype, device) for key, value in node.kwargs.items()
        }
    exact.recompile()
    return exact


class _Oracle:
    """Evaluate the reference without rounding, where it can run.

    The order is float64 on the region's device; float64 on the CPU, when the
    device lacks the memory or a float64 kernel; and, only for regions whose
    boundaries are all float16 or bfloat16, the reference promoted to float32,
    whose rounding is far below theirs. Once an evaluation falls back, later
    cases start there. If none runs, the evaluation is inconclusive: it never
    falls back to comparing with eager PyTorch.
    """

    def __init__(self, reference: Callable[..., Any]) -> None:
        self.reference = reference
        self.level = 0
        self.failures: list[str] = []
        # Whether a level failed for want of memory, which can free up.
        self.memory = False
        self.basis: str | None = None
        self.variants: dict[tuple[torch.dtype, torch.device], Callable[..., Any]] = {}
        # _attention_bytes for each set of input shapes.
        self.attention: dict[tuple[torch.Size, ...], int] = {}

    def _attention_bytes(self, inputs: Sequence[torch.Tensor]) -> int:
        shapes = tuple(value.shape for value in inputs)
        if shapes not in self.attention:
            self.attention[shapes] = _attention_bytes(self.reference, inputs)
        return self.attention[shapes]

    def _variant(
        self, dtype: torch.dtype, device: torch.device, home: torch.device
    ) -> Callable[..., Any]:
        key = (dtype, device)
        if key not in self.variants:
            if isinstance(self.reference, torch.fx.GraphModule):
                self.variants[key] = _float64_reference(
                    self.reference, dtype, None if device == home else device
                )
            elif dtype == torch.float64:
                # A plain function cannot be rewritten; it runs as given.
                self.variants[key] = self.reference
            else:
                raise _Unpromotable("only a captured graph can be promoted")
        return self.variants[key]

    def __call__(
        self,
        inputs: Sequence[torch.Tensor],
        eager: Sequence[torch.Tensor],
        eager_peak_bytes: int | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], str]:
        home = inputs[0].device
        levels = [(torch.float64, home)]
        if home.type != "cpu":
            levels.append((torch.float64, torch.device("cpu")))
        if all(v.dtype in (torch.float16, torch.bfloat16) for v in (*inputs, *eager)):
            levels.append((torch.float32, home))
        while self.level < len(levels):
            dtype, device = levels[self.level]
            name = f"{str(dtype).removeprefix('torch.')} on {device}"
            try:
                if device.type == "cuda" and dtype == torch.float64:
                    _check_oracle_memory(
                        device,
                        inputs,
                        eager,
                        eager_peak_bytes,
                        attention_bytes=(
                            0
                            if eager_peak_bytes is None
                            else self._attention_bytes(inputs)
                        ),
                    )
                function = self._variant(dtype, device, home)
                # The exact values must not round to TF32, whatever the
                # process's settings are.
                with _tf32(False):
                    exact = _outputs(function(*(v.to(device, dtype) for v in inputs)))
                if [v.shape for v in exact] != [v.shape for v in eager]:
                    raise ValueError("its outputs have different shapes")
                self.basis = name
                return tuple(v.to(home, torch.float64) for v in exact), name
            except _Unpromotable as error:
                self.failures.append(f"{name}: {error}")
                break
            except Exception as error:
                message = str(error).partition("\n")[0][:300]
                self.failures.append(f"{name}: {message}")
                self.memory |= isinstance(error, torch.OutOfMemoryError) or bool(
                    _OUT_OF_MEMORY.search(message)
                )
                _LOGGER.warning("The reference failed in %s: %s", name, message)
                self.level += 1
                if device.type == "cuda":
                    # Return a failed attempt's memory before the next one.
                    torch.cuda.empty_cache()
        raise _EvaluationError(
            "setup",
            "The region's PyTorch reference could not be evaluated without "
            f"rounding ({'; '.join(self.failures)}), so candidates cannot be "
            "checked against exact values; set KernelGenerator rtol and atol to "
            "compare with eager PyTorch at those tolerances instead",
            # Only memory that other work holds can change the outcome.
            retryable=self.memory,
        )


def _attention_bytes(
    reference: Callable[..., Any], inputs: Sequence[torch.Tensor]
) -> int:
    """Float64 memory for the reference's attention that eager's peak misses.

    Eager PyTorch's fused attention kernels never hold the score matrix, an
    element for each query and key position of each head, but the float64
    reference runs PyTorch's math kernel, which holds up to
    _ATTENTION_SCORE_COPIES such matrices at once. A call frees them before
    the next, so the largest call counts. The shapes come from running the
    reference on meta tensors; a call after an operation that cannot run
    there is not counted, and a failed GPU evaluation still falls back.
    """
    attention = torch.ops.aten.scaled_dot_product_attention.default
    if not isinstance(reference, torch.fx.GraphModule) or not any(
        node.target is attention for node in reference.graph.nodes
    ):
        return 0
    scores: list[int] = []

    class _Shapes(torch.fx.Interpreter):  # type: ignore[misc,unused-ignore]
        def call_function(self, target: Any, args: Any, kwargs: Any) -> Any:
            if target is attention:
                named = {**dict(zip(("query", "key"), args)), **kwargs}
                query, key = named["query"], named["key"]
                # Grouped-query attention repeats the key's heads.
                heads = max(math.prod(query.shape[:-2]), math.prod(key.shape[:-2]))
                scores.append(heads * query.shape[-2] * key.shape[-2])
            return super().call_function(target, args, kwargs)

    meta = torch.device("meta")
    try:
        _Shapes(_float64_reference(reference, device=meta)).run(
            *(torch.empty(v.shape, dtype=torch.float64, device=meta) for v in inputs)
        )
    except Exception:
        pass
    return _ATTENTION_SCORE_COPIES * max(scores, default=0) * 8


def _check_oracle_memory(
    device: torch.device,
    inputs: Sequence[torch.Tensor],
    eager: Sequence[torch.Tensor],
    eager_peak_bytes: int | None,
    attention_bytes: int = 0,
) -> None:
    """Refuse a float64 evaluation that would not fit the device's free memory.

    float64 needs up to 8 / itemsize times eager PyTorch's own peak, for the
    smallest floating itemsize at the boundary; measured ratios are 2-2.7
    where float64 runs the same algorithms as eager PyTorch. Attention may
    not: ``attention_bytes`` holds the score matrices that its float64 math
    kernel keeps (see _attention_bytes), without which attention over 1024
    to 4096 positions needed 5.5 to 18 times the estimate when eager PyTorch
    ran a fused kernel. Where eager PyTorch ran the math kernel itself, its
    peak holds score matrices already, so the estimate is the larger of the
    two rather than their sum.
    """
    if eager_peak_bytes is None:
        return
    itemsize = min(v.element_size() for v in (*inputs, *eager) if v.is_floating_point())
    needed = eager_peak_bytes * 8 // itemsize
    if attention_bytes:
        boundary = sum(v.numel() * v.element_size() for v in (*inputs, *eager))
        needed = max(needed, attention_bytes + boundary * 8 // itemsize)
    free = torch.cuda.mem_get_info(device)[0]
    # Memory this process caches but does not use is free to it as well.
    free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    if needed > free:
        raise torch.OutOfMemoryError(
            f"float64 needs about {needed / 2**30:.3g} GiB, but "
            f"{free / 2**30:.3g} GiB is free"
        )


@contextmanager
def _tf32(allowed: bool) -> Iterator[None]:
    """Allow or forbid TF32 for float32 matmuls and convolutions, then restore."""
    matmul, convolution = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    torch.backends.cuda.matmul.allow_tf32 = allowed
    torch.backends.cudnn.allow_tf32 = allowed
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = convolution


def _tf32_matmuls() -> Any:
    """Let eager PyTorch use TF32 for float32 matmuls and convolutions."""
    return _tf32(True)


def _expected(
    reference: Callable[..., Any],
    inputs: tuple[torch.Tensor, ...],
    specs: Sequence[_TensorBoundary],
    *,
    exact_reference: Callable[..., Any] | None,
    fp32_matmul_precision: str = "highest",
) -> _Expected:
    """Evaluate eager PyTorch and, unless tolerances are explicit, the oracle.

    ``exact_reference`` is an :class:`_Oracle`, or a reference to evaluate
    without rounding through one. Eager PyTorch's error against the oracle
    sets every candidate's allowance, so eager PyTorch then runs with TF32 off,
    for cuDNN convolutions too, which PyTorch's defaults let use it:
    ``"highest"`` holds float32 candidates to IEEE float32's error, and
    ``"high"`` adds the TF32 run as ``relaxed``. Explicit tolerances compare
    with eager PyTorch at PyTorch's defaults, as timing runs it.
    """
    device = inputs[0].device
    measured, start = device.type == "cuda", 0
    if measured:
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.memory_allocated(device)
    with nullcontext() if exact_reference is None else _tf32(False):
        eager = _outputs(reference(*(value.clone() for value in inputs)))
    peak = torch.cuda.max_memory_allocated(device) - start if measured else None
    _check_contract(eager, specs)
    relaxed = None
    if fp32_matmul_precision == "high":
        with _tf32_matmuls():
            relaxed = _outputs(reference(*(value.clone() for value in inputs)))
    if exact_reference is None:
        return _Expected(eager, None, relaxed)
    oracle = (
        exact_reference
        if isinstance(exact_reference, _Oracle)
        else _Oracle(exact_reference)
    )
    exact, basis = oracle(inputs, eager, peak)
    return _Expected(eager, exact, relaxed, basis)


def _replaced(
    inputs: Sequence[torch.Tensor], index: int, value: float
) -> tuple[torch.Tensor, ...]:
    """Copy ``inputs``, setting element 0 of input ``index`` to ``value``."""
    copies = tuple(item.clone() for item in inputs)
    copies[index].reshape(-1)[0] = value
    return copies


def _propagation(
    reference: Callable[..., Any],
    inputs: tuple[torch.Tensor, ...],
    index: int,
    expected: _Expected,
    original: float | None = None,
) -> _Expected:
    """Mark the outputs that the non-finite element 0 of input ``index`` reaches.

    Eager PyTorch shows them as NaN when that element is NaN. A candidate may
    hold a non-finite value there, even where eager PyTorch makes a finite one
    of an infinity, as exp(-inf) or a masked fill with the dtype's minimum do;
    a finite value there must still be accurate.

    When the injected value is a NaN and ``original``, the finite value it
    replaced, is known, also mark the outputs that the NaN should reach:
    those it makes non-finite that are finite with the original value and
    change when the element changes. Outputs where eager PyTorch multiplies
    the element by zero, which a masked kernel need not read, do not change
    with it, so they are not marked. A candidate that reads the element
    usually makes one of them non-finite too, whatever its order of
    operations; one whose maximum, minimum or clamp drops NaN, as CUDA's
    fmaxf and Triton's tl.maximum do, must instead show that those outputs
    change with the element (see _check_outputs). One that ignores the input
    or hard-codes its value can do neither.
    """
    reached = _outputs(reference(*_replaced(inputs, index, math.nan)))
    expected = expected._replace(propagated=tuple(value.isnan() for value in reached))
    if (
        original is None
        or not math.isfinite(original)
        or not bool(inputs[index].reshape(-1)[0].isnan())
    ):
        return expected
    limit = torch.finfo(inputs[index].dtype).max
    changed = original + math.copysign(max(1.0, abs(original)), original)
    changed = max(-limit, min(limit, changed))
    clean = _outputs(reference(*_replaced(inputs, index, original)))
    moved = _outputs(reference(*_replaced(inputs, index, changed)))
    required = tuple(
        ~poisoned.isfinite() & before.isfinite() & after.isfinite() & (before != after)
        for poisoned, before, after in zip(reached, clean, moved)
    )
    if not any(bool(value.any()) for value in required):
        return expected
    return expected._replace(required=required, injection=(index, original, changed))


def _case_expectation(
    reference: Callable[..., Any],
    label: str,
    inputs: tuple[torch.Tensor, ...],
    specs: Sequence[_TensorBoundary],
    *,
    oracle: _Oracle | None,
    precision: str,
    injected: dict[str, int],
    originals: dict[str, float],
    robustness: bool = False,
) -> _Expected:
    """The expected outputs of one labelled case, as an evaluation checks it.

    ``injected`` and ``originals`` are the maps that _input_cases filled;
    ``robustness`` marks a case in its ``robust`` set.
    """
    expected = _expected(
        reference,
        inputs,
        specs,
        exact_reference=oracle,
        fp32_matmul_precision=precision,
    )
    if label in injected:
        expected = _propagation(
            reference, inputs, injected[label], expected, originals.get(label)
        )
    return expected._replace(robustness=robustness)


def _case_count(
    num_tests: int, check_nonfinite: bool, num_inputs: int, num_samples: int
) -> tuple[int, int]:
    """The number of cases _input_cases yields, and of those it injects."""
    injected = len(_NONFINITE) * num_inputs if check_nonfinite else 0
    total = num_tests + len(_SCALED_CASES) + len(_CONSTANTS) + injected + num_samples
    return total, injected


def _expectation_bytes(
    specs: Sequence[_TensorBoundary],
    cases: int,
    injected: int,
    *,
    oracle: bool,
    relaxed: bool,
) -> int:
    """Bytes of every case's expected outputs: eager PyTorch's, the float64
    oracle's, the TF32 ones and, for injected cases, two element masks."""
    elements = sum(math.prod(spec.shape) for spec in specs)
    native = sum(math.prod(spec.shape) * spec.dtype.itemsize for spec in specs)
    per_case = native * (2 if relaxed else 1) + (8 * elements if oracle else 0)
    return cases * per_case + injected * 2 * elements


def _unchecked(expected: _Expected) -> str | None:
    """Describe an output that a case mostly leaves unchecked, if there is one.

    Wherever eager PyTorch's output is NaN or infinite, any candidate value
    passes.
    """
    for index, value in enumerate(expected.eager):
        count = int((~value.isfinite()).sum())
        if count > _INCONCLUSIVE_FRACTION * value.numel():
            return (
                f"output {index} is NaN or infinite at {count} of "
                f"{value.numel()} elements"
            )
    return None


def _require_conclusive(
    unchecked: dict[str, str], labels: Sequence[str], sampled: bool
) -> None:
    """Refuse synthetic inputs that leave every random case mostly unchecked.

    ``unchecked`` describes the inconclusive cases among those not injected
    with a non-finite value. Without samples, random inputs are standard
    normal: when eager PyTorch's outputs on all of them are mostly NaN or
    infinite, the inputs leave the region's domain, and a candidate checked
    on them is not shown to be correct. Captured samples are the model's own
    values, so cases derived from them stand.
    """
    randoms = [label for label in labels if label.startswith("random ")]
    if sampled or not randoms or any(label not in unchecked for label in randoms):
        return
    detail = "; ".join(f"{label}: {unchecked[label]}" for label in randoms)
    raise _EvaluationError(
        "setup",
        "Eager PyTorch's outputs are mostly NaN or infinite on every random "
        f"validation case ({detail}), so candidates would be checked at few "
        "output elements: the unconstrained N(0,1) validation inputs leave the "
        "region's domain, as a decay that the model keeps negative does when it "
        "turns positive. Compile with real example tensors rather than Input "
        "specifications, or pass KernelGenerator(validation_inputs=...), so that "
        "validation derives its inputs from the model's own values",
        retryable=False,
    )


def _warn_nonfinite(
    label: str, inputs: Sequence[torch.Tensor], expected: _Expected
) -> None:
    """Report a reference that overflows on finite inputs.

    Those elements are not checked, so the case checks less than it seems:
    the inputs probably leave the domain that the model gives the region.
    """
    if not all(bool(value.isfinite().all()) for value in inputs):
        return
    counts = [int((~value.isfinite()).sum()) for value in expected.eager]
    if any(counts):
        _LOGGER.warning(
            "Eager PyTorch produced %s non-finite output elements on case %r from "
            "finite inputs; candidates are not checked there",
            "+".join(map(str, counts)),
            label,
        )


def _same(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left == right) | (left.isnan() & right.isnan())


def _same_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(
        torch.equal(
            left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8)
        )
    )


def _norm(values: torch.Tensor, where: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(torch.where(where, values, 0.0)))


def _oracle_mismatch(
    actual: torch.Tensor,
    eager: torch.Tensor,
    oracle: torch.Tensor,
    relaxed: torch.Tensor | None = None,
    *,
    basis: str = "float64",
    propagated: torch.Tensor | None = None,
) -> str | None:
    """Describe how the candidate misses the exact values, if it does.

    Where eager PyTorch's output is not finite, the exact result is out of
    range or undefined, so the candidate may hold any value there; elsewhere
    it must be finite and accurate. An element passes when it equals the eager
    value or the correctly rounded exact value, or lies within atol +
    rtol*|exact| plus a multiple of eager PyTorch's own error. Where eager only
    rounds, that is a few units in the last place of the element itself; where
    eager accumulates error, as reductions and contractions do, it is eager's
    largest error over the output, since a different summation order errs
    elsewhere. The error over the whole output must also stay within
    _AGGREGATE_FACTOR times eager's own, or times a floor when eager's is
    smaller: one unit roundoff of the exact values, at least _AGGREGATE_FLOOR
    relative, and at least atol per element. ``relaxed``, eager PyTorch with TF32
    matmuls, widens eager's own error to the larger of the two. Where
    ``propagated``, an injected non-finite input reaches the output, and a
    non-finite candidate value passes too.
    """
    actual64, eager64 = actual.double(), eager.double()
    checked = eager64.isfinite()
    known = checked & oracle.isfinite()
    target = torch.where(oracle.isfinite(), oracle, eager64)
    errors = torch.where(known, (eager64 - oracle).abs(), 0.0)
    own_norm = _norm(eager64 - oracle, known)
    precision = ""
    if relaxed is not None:
        relaxed64 = relaxed.double()
        within = known & relaxed64.isfinite()
        errors = torch.maximum(
            errors, torch.where(within, (relaxed64 - oracle).abs(), 0.0)
        )
        own_norm = max(own_norm, _norm(relaxed64 - oracle, within))
        precision = " with TF32 matmuls allowed"
    limits = torch.finfo(eager.dtype)
    ulp = limits.eps * target.abs().clamp_min(limits.tiny)
    own_error = float(errors.max())
    exact_norm = _norm(oracle, known)
    own = (
        f"against the reference evaluated in {basis}, eager PyTorch{precision} "
        f"errs by {own_error:.3g} at most and by "
        f"{own_norm / max(exact_norm, 1e-300):.3g} in relative L2 norm"
    )
    if bool((errors <= _ROUNDING_ULPS * ulp).all()):
        slack, kind = _ORACLE_ERROR_FACTOR * ulp, "units in the last place"
    else:
        slack, kind = _ORACLE_ERROR_FACTOR * own_error, "eager's largest error"
    rtol, atol = _TOLERANCES[eager.dtype]
    allowance = atol + rtol * target.abs() + slack
    error = (actual64 - target).abs()
    accepted = (
        ~checked
        | (error <= allowance)
        | _same(actual64, eager64)
        | _same(actual64, oracle.to(eager.dtype).double())
    )
    if relaxed is not None:
        accepted |= _same(actual64, relaxed64)
    if propagated is not None:
        accepted |= propagated & ~actual64.isfinite()
    if not bool(accepted.all()):
        excess = torch.where(
            accepted,
            0.0,
            (error / allowance).nan_to_num(nan=math.inf, posinf=math.inf),
        )
        worst = int(excess.argmax())
        position = tuple(
            int(i) for i in torch.unravel_index(torch.tensor(worst), actual.shape)
        )
        detail = (
            f"got {float(actual64.reshape(-1)[worst]):.6g} at index {position}, "
            f"expected {float(target.reshape(-1)[worst].to(eager.dtype)):.6g} "
            f"(error {float(error.reshape(-1)[worst]):.3g} exceeds the allowance "
            f"{float(allowance.reshape(-1)[worst]):.3g}: atol {atol:g} + rtol "
            f"{rtol:g}*|expected| + {_ORACLE_ERROR_FACTOR}x {kind})"
        )
        return (
            f"{detail}; {int((~accepted).sum())} of {actual.numel()} elements "
            f"differ; {own}"
        )
    # A correctly rounded output errs by up to one unit roundoff per element,
    # and the elementwise atol already allows any error below it.
    finite = known & actual64.isfinite()
    floor = max(
        max(limits.eps / 2, _AGGREGATE_FLOOR) * exact_norm,
        atol * math.sqrt(int(finite.sum())),
    )
    bound = _AGGREGATE_FACTOR * max(own_norm, floor)
    deviation = _norm(actual64 - oracle, finite)
    if deviation > bound:
        return (
            f"every element is within its allowance, but the error over all "
            f"{int(finite.sum())} finite elements is "
            f"{deviation / max(exact_norm, 1e-300):.3g} in relative L2 norm, "
            f"above {_AGGREGATE_FACTOR}x eager PyTorch's own error or "
            f"{_AGGREGATE_FACTOR}x the rounding and atol floor, whichever is larger "
            f"({bound / max(exact_norm, 1e-300):.3g}): a small error shared by "
            f"many elements, such as a wrong scale or a shifted window; {own}"
        )
    return None


def _robustness_mismatch(
    actual: torch.Tensor,
    eager: torch.Tensor,
    oracle: torch.Tensor,
    relaxed: torch.Tensor | None = None,
    *,
    basis: str = "float64",
) -> str | None:
    """Describe how the candidate fails a robustness case, if it does.

    Robustness cases take inputs beyond the sampled range, where a correct
    kernel's different but standard arithmetic, such as exponentials of
    differences of long prefix sums, can err far more than eager PyTorch's
    own. They check what a kernel must get right there. Wherever eager
    PyTorch's output is finite, the candidate's must be finite too, which an
    unguarded exponential or an overflowing intermediate breaks. The error
    over the output may reach _AGGREGATE_FACTOR times eager's own or
    _ROBUSTNESS_FLOOR of the exact values' norm, whichever is larger. Each
    element may err by up to _ORACLE_ERROR_FACTOR times eager's largest error
    or _ROBUSTNESS_ELEMENT_FLOOR of the output's largest exact magnitude,
    plus atol, so that a few grossly wrong elements cannot hide in the norm
    of a large output. ``relaxed`` widens eager's error as in
    _oracle_mismatch.
    """
    actual64, eager64 = actual.double(), eager.double()
    finite = eager64.isfinite()
    broken = finite & ~actual64.isfinite()
    if bool(broken.any()):
        first = int(broken.reshape(-1).nonzero()[0])
        position = tuple(
            int(i) for i in torch.unravel_index(torch.tensor(first), actual.shape)
        )
        return (
            f"got {float(actual64.reshape(-1)[first]):.6g} at index {position}, "
            f"where eager PyTorch is finite "
            f"({float(eager64.reshape(-1)[first]):.6g}); {int(broken.sum())} of "
            f"{actual.numel()} elements are NaN or infinite where eager PyTorch "
            "is finite, as after an unguarded exponential or an overflowing "
            "intermediate"
        )
    known = finite & oracle.isfinite()
    errors = torch.where(known, (eager64 - oracle).abs(), 0.0)
    own_norm = _norm(eager64 - oracle, known)
    precision = ""
    if relaxed is not None:
        relaxed64 = relaxed.double()
        within = known & relaxed64.isfinite()
        errors = torch.maximum(
            errors, torch.where(within, (relaxed64 - oracle).abs(), 0.0)
        )
        own_norm = max(own_norm, _norm(relaxed64 - oracle, within))
        precision = " with TF32 matmuls allowed"
    exact_norm = _norm(oracle, known)
    scale = max(exact_norm, 1e-300)
    own_error = float(errors.max()) if errors.numel() else 0.0
    largest = float(torch.where(known, oracle.abs(), 0.0).max()) if known.any() else 0.0
    own = (
        f"against the reference evaluated in {basis}, eager PyTorch{precision} "
        f"errs by {own_error:.3g} at most and by {own_norm / scale:.3g} in "
        "relative L2 norm"
    )
    _, atol = _TOLERANCES[eager.dtype]
    limit = atol + max(
        _ORACLE_ERROR_FACTOR * own_error, _ROBUSTNESS_ELEMENT_FLOOR * largest
    )
    error = torch.where(known, (actual64 - oracle).abs(), 0.0)
    if error.numel() and float(error.max()) > limit:
        worst = int(error.argmax())
        position = tuple(
            int(i) for i in torch.unravel_index(torch.tensor(worst), actual.shape)
        )
        return (
            f"got {float(actual64.reshape(-1)[worst]):.6g} at index {position}, "
            f"expected {float(oracle.reshape(-1)[worst]):.6g} (error "
            f"{float(error.reshape(-1)[worst]):.3g} exceeds {limit:.3g}: atol "
            f"{atol:g} + the larger of {_ORACLE_ERROR_FACTOR}x eager's largest "
            f"error and {_ROBUSTNESS_ELEMENT_FLOOR:.0%} of the output's largest "
            f"exact magnitude); {own}"
        )
    deviation = _norm(actual64 - oracle, known)
    bound = max(_AGGREGATE_FACTOR * own_norm, _ROBUSTNESS_FLOOR * exact_norm)
    if deviation > bound:
        return (
            f"the error over all {int(known.sum())} checked elements is "
            f"{deviation / scale:.3g} in relative L2 norm, above the larger of "
            f"{_AGGREGATE_FACTOR}x eager PyTorch's own error and "
            f"{_ROBUSTNESS_FLOOR:.0%} ({bound / scale:.3g}); {own}"
        )
    return None


def _check_outputs(
    actual: Sequence[torch.Tensor],
    expected: _Expected,
    where: str,
    *,
    rtol: float | None,
    atol: float | None,
    reads: Callable[[], bool] | None = None,
) -> None:
    """Raise unless ``actual`` meets ``expected``.

    For an injected NaN that must reach the output, ``reads`` may show
    instead that the candidate's outputs change with the injected element.
    """
    expected = expected.to(actual[0].device)
    for index, (value, eager) in enumerate(zip(actual, expected.eager)):
        reached = None if expected.propagated is None else expected.propagated[index]
        if expected.oracle is not None and expected.robustness:
            problem = _robustness_mismatch(
                value,
                eager,
                expected.oracle[index],
                None if expected.relaxed is None else expected.relaxed[index],
                basis=expected.basis or "float64",
            )
        elif expected.oracle is not None:
            problem = _oracle_mismatch(
                value,
                eager,
                expected.oracle[index],
                None if expected.relaxed is None else expected.relaxed[index],
                basis=expected.basis or "float64",
                propagated=reached,
            )
        else:
            free = ~eager.isfinite()
            if reached is not None:
                free |= reached & ~value.isfinite()
            try:
                # Where eager PyTorch is not finite, any value passes.
                torch.testing.assert_close(
                    torch.where(free, eager, value),
                    eager,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                )
                problem = None
            except AssertionError as error:
                problem = str(error)
        if problem is not None:
            raise _EvaluationError(
                "correctness", f"Output {index} is wrong for {where}: {problem}"
            )
    if expected.required is not None and not any(
        bool((required & ~value.isfinite()).any())
        for value, required in zip(actual, expected.required)
    ):
        if reads is not None and reads():
            # The NaN was dropped, as a NaN-ignoring maximum or clamp does,
            # but the outputs depend on the element: the kernel reads it.
            return
        counts = "+".join(str(int(required.sum())) for required in expected.required)
        raise _EvaluationError(
            "correctness",
            f"Every output is finite for {where}, but eager PyTorch's outputs "
            f"are NaN or infinite at {counts} elements that depend on the "
            "injected element: they are finite with its original value and "
            "change when it changes. "
            + (
                "The candidate's outputs there stay the same when the element "
                "takes another finite value: the kernel ignores that input or "
                "hard-codes its value"
                if reads is not None
                else "A kernel that ignores that input or hard-codes its value "
                "neither propagates the NaN nor changes them"
            ),
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
    aligned: bool = True,
) -> None:
    """Check ``function`` on one case, given copies of ``inputs``.

    Unless ``aligned``, the copies are aligned only to their element size.
    """
    place = _copy if aligned else _unaligned
    arguments = tuple(map(place, inputs))
    # The candidate's own launch or engine run: device memory running out
    # there is the candidate's to answer for.
    actual = _outputs(
        _run_stage("correctness", function, *arguments, out_of_memory="resources")
    )
    _check_contract(actual, specs)
    for index, (argument, value) in enumerate(zip(arguments, inputs)):
        if not _same_bits(argument, value):
            raise _EvaluationError(
                "correctness", f"Input {index} was modified on case {label!r}"
            )

    def reads() -> bool:
        """Whether the outputs that the injected element reaches change with it."""
        assert expected.injection is not None and expected.required is not None
        index, original, changed = expected.injection
        before, after = [
            _outputs(
                _run_stage(
                    "correctness",
                    function,
                    *map(place, _replaced(inputs, index, value)),
                    out_of_memory="resources",
                )
            )
            for value in (original, changed)
        ]
        return any(
            bool(
                (
                    required.to(old.device)
                    & old.isfinite()
                    & new.isfinite()
                    & (old != new)
                ).any()
            )
            for required, old, new in zip(expected.required, before, after)
        )

    _check_outputs(
        actual,
        expected,
        (
            f"robustness case {label!r} (its inputs leave the sampled range, so "
            "it requires finite outputs wherever eager PyTorch's are finite and "
            f"an error within {_ROBUSTNESS_FLOOR:.0%} of the exact output, not "
            "the in-range bounds)"
            if expected.robustness and expected.oracle is not None
            else f"case {label!r}"
        ),
        rtol=rtol,
        atol=atol,
        reads=None if expected.injection is None else reads,
    )


def _check_guard_bands(
    launch: Callable[[list[torch.Tensor], list[torch.Tensor]], None],
    inputs: tuple[torch.Tensor, ...],
    expected: _Expected,
    label: str,
    *,
    rtol: float | None,
    atol: float | None,
    aligned: bool = True,
) -> None:
    """Launch on interior views of padded buffers and reject accesses outside them.

    The trial engine gives each output its own allocation with allocator slack,
    so its results cannot show a write past the end of a tensor. Two launches
    with different padding must agree bit for bit, so an out-of-bounds read
    that changes an output is caught even when an operation such as max drops
    the NaN padding of the first launch; a read whose value is discarded is not.
    Unless ``aligned``, the views are aligned only to their element size.
    """
    where = f"the guard-band launch on case {label!r}"
    results = []
    for fill in _GUARD_FILLS:
        guarded = {
            kind: [_padded(value, fill, aligned) for value in values]
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
        try:
            torch.cuda.synchronize()
        except RuntimeError as error:
            # Such as "misaligned address" from a kernel that assumes vector
            # alignment that the task does not promise.
            alignment = (
                "16 bytes (codegen.aligned_io)"
                if aligned
                else "their element size only, as codegen.aligned_io is false"
            )
            raise _EvaluationError(
                "correctness",
                f"The kernel failed in {where}, whose tensors are aligned to "
                f"{alignment}: {error}",
            ) from error
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
    aligned: bool = True,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Run a candidate through the backend's direct launch into fresh outputs.

    The launch runs the candidate's source with its launch settings, as the
    plugin does, but a backend may compile that source separately, as Triton
    and cuTile do, so only an engine runs the plugin's own binary. Unless
    ``aligned``, the outputs are aligned only to their element size.
    """

    def run(*inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = [
            _empty(spec.shape, spec.dtype, spec.device, aligned)
            for spec in output_specs
        ]
        launch(candidate, list(inputs), outputs, directory)
        return tuple(outputs)

    return run


def _misalignment(address: int, element_size: int) -> int:
    """Elements to skip from ``address`` to one aligned to the element size only.

    Such an address, a multiple of the element size but not of twice it, suits
    no vector load or store of two or more elements. Zero when every address
    of that element size is a multiple of _VECTOR_BYTES anyway.
    """
    if element_size % _VECTOR_BYTES == 0:
        return 0
    return 1 if address % (2 * element_size) == 0 else 0


def _empty(
    shape: Sequence[int], dtype: torch.dtype, device: torch.device, aligned: bool
) -> torch.Tensor:
    """A contiguous tensor; unless ``aligned``, aligned to its element size only."""
    if aligned:
        return torch.empty(shape, dtype=dtype, device=device)
    numel = math.prod(shape)
    buffer = torch.empty(numel + 1, dtype=dtype, device=device)
    start = _misalignment(buffer.data_ptr(), buffer.element_size())
    return buffer[start : start + numel].view(shape)


def _copy(value: torch.Tensor) -> torch.Tensor:
    return value.clone()


def _unaligned(value: torch.Tensor) -> torch.Tensor:
    """A copy of ``value`` at an address aligned to its element size only."""
    return _empty(value.shape, value.dtype, value.device, False).copy_(value)


def _padded(
    like: torch.Tensor, fill: int, aligned: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a buffer filled with ``fill`` bytes and its interior view.

    Unless ``aligned``, the view is aligned to its element size only.
    """
    size = like.element_size()
    pad = -(-_GUARD_BYTES // size)
    buffer = torch.empty(
        like.numel() + 2 * pad + 1, dtype=like.dtype, device=like.device
    )
    buffer.view(torch.uint8).fill_(fill)
    if not aligned:
        pad += _misalignment(buffer.data_ptr() + pad * size, size)
    return buffer, buffer[pad : pad + like.numel()].view(like.shape)


def evaluate(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    diagnostics: dict[str, Any] | None = {} if request["diagnostics"] else None
    # A shared-memory fit records the variant evaluated in place of the
    # candidate, which every report must name, passed or not.
    fit: dict[str, Any] = {}
    try:
        result = _run_stage("setup", _evaluate, request, directory, diagnostics, fit)
    except _EvaluationError as error:
        error.diagnostics = diagnostics
        error.fit = fit
        raise
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    result.update(fit)
    return result


def _fit_shared_memory(
    backend: Any,
    request: dict[str, Any],
    candidate: dict[str, Any],
    specs: tuple[list[_TensorBoundary], list[_TensorBoundary]],
    directory: Path,
    error: _EvaluationError,
    fit: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a variant of a search candidate whose plugin fits in shared memory.

    Only search evaluations fit, so holdout and cache checks judge the exact
    kernel they were given. The variant is validated like any proposal.
    """
    from ._backends import SharedMemoryLimitError

    method = getattr(backend, "fit_shared_memory", None)
    if (
        not request.get("auto_fit")
        or request["validation_phase"] != "search"
        or method is None
        or not isinstance(error.__cause__, SharedMemoryLimitError)
    ):
        return None
    fitted, log = _run_stage(
        "compile",
        method,
        candidate,
        *specs,
        directory,
        request.get("codegen"),
        error.__cause__,
    )
    fit["fit_log"] = list(log)
    fit["shared_memory"] = {
        "required_bytes": error.__cause__.required,
        "limit_bytes": error.__cause__.limit,
        "largest_allocations": list(error.__cause__.allocations),
    }
    if fitted is None:
        return None
    validated: dict[str, Any] = _run_stage(
        "source", backend.validate_candidate, fitted, len(specs[0]), len(specs[1])
    )
    fit["fitted_candidate"] = validated
    return validated


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
    fit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check one candidate; anything not attributed to a stage is ``setup``."""
    from ._backends import get_backend, registration_for

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
    # Without aligned_io, a plugin can get pointers aligned only to their
    # element size: TensorRT can bind it to a view at any element offset of a
    # larger buffer, such as one input of an eliminated concatenation. Direct
    # launches then use such pointers: the guard-band launch of every
    # evaluation and, when screening, every correctness case and timed launch.
    # They reject kernels that assume more, such as CUDA C++ float4 loads,
    # and they make Triton's and cuTile's own launches, which specialize on
    # the alignment of the actual pointers, compile the code the plugin runs
    # instead of 16-byte aligned code that times faster. Engine runs keep aligned buffers: TensorRT requires
    # 256-byte aligned engine inputs and outputs, and the engine runs the
    # plugin's own binary, compiled for the aligned_io setting whatever its
    # pointers are. The eager reference and the TensorRT baseline are timed
    # on aligned inputs as well, as they run in PyTorch and in an engine.
    aligned = bool((request.get("codegen") or {}).get("aligned_io", False))
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

    def cases(
        injected: dict[str, int] | None = None,
        originals: dict[str, float] | None = None,
        robust: set[str] | None = None,
    ) -> Iterator[tuple[str, tuple[torch.Tensor, ...]]]:
        # The seed fixes every value, so each pass yields the same cases.
        return _input_cases(
            input_specs,
            request["num_tests"],
            request["check_nonfinite"],
            request["validation_seed"],
            samples,
            injected,
            originals,
            robust,
        )

    # Older requests predate the setting, and so mean IEEE float32.
    precision = request.get("fp32_matmul_precision", "highest")
    oracle = _Oracle(reference) if rtol is None else None
    injected: dict[str, int] = {}
    originals: dict[str, float] = {}
    # Magnitude cases that took inputs beyond the sampled range.
    robust: set[str] = set()
    # Inconclusive cases among those without injected non-finite values.
    unchecked: dict[str, str] = {}
    noted: set[str] = set()

    def expect(label: str, inputs: tuple[torch.Tensor, ...]) -> _Expected:
        expected = _case_expectation(
            reference,
            label,
            inputs,
            output_specs,
            oracle=oracle,
            precision=precision,
            injected=injected,
            originals=originals,
            robustness=label in robust,
        )
        note(label, inputs, expected)
        return expected

    def note(label: str, inputs: tuple[torch.Tensor, ...], expected: _Expected) -> None:
        if label in noted:
            return
        noted.add(label)
        if label not in injected:
            problem = _unchecked(expected)
            if problem is not None:
                unchecked[label] = problem
                _LOGGER.warning(
                    "Case %r checks little: %s, where any candidate value passes",
                    label,
                    problem,
                )
        _warn_nonfinite(label, inputs, expected)

    count, injections = _case_count(
        request["num_tests"],
        request["check_nonfinite"],
        len(input_specs),
        len(samples),
    )
    # Every case's expected outputs come first, while nothing else holds
    # device memory: the float64 oracle is the evaluation's largest
    # allocation. They wait on the CPU, unless together they would take more
    # host memory than _EXPECTATION_BYTES: then each case's are computed when
    # it is checked. The first case drives the guard bands, engine builds and
    # timers, so its expected outputs stay on the device.
    kept = (
        _expectation_bytes(
            output_specs,
            count,
            injections,
            oracle=oracle is not None,
            relaxed=precision == "high",
        )
        <= _EXPECTATION_BYTES
    )
    expectations: list[_Expected] = []
    labels: list[str] = []
    first_label, first_expected = "", _Expected((), None)
    first: tuple[torch.Tensor, ...] = ()
    for label, inputs in cases(injected, originals, robust):
        labels.append(label)
        if not first or kept:
            expected = expect(label, inputs)
            if not first:
                first_label, first, first_expected = label, inputs, expected
            if kept:
                expectations.append(expected.to("cpu"))
            del expected
        elif not samples and label.startswith("random "):
            # Whether synthetic inputs check enough needs eager PyTorch only.
            eager = _outputs(reference(*(value.clone() for value in inputs)))
            note(label, inputs, _Expected(eager, None))
            del eager
        del inputs
    _require_conclusive(unchecked, labels, bool(samples))
    torch.cuda.empty_cache()
    _restore_metadata(reference, first, first_expected.eager)
    iterations = request["benchmark_iters"]

    options = request["compile_options"]
    # A screen decides no speedup objective, so it builds the baseline engine
    # only when the controller asks: a baseline that TensorRT cannot build
    # must still end the search early.
    baseline = None
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

    def build(name: str, proposal: dict[str, Any]) -> Callable[..., Any]:
        with _compilation_diagnostics(
            diagnostics[name] if diagnostics is not None else None
        ):
            registration = _run_stage(
                "compile",
                registration_for,
                backend,
                proposal,
                input_specs,
                output_specs,
                directory / name,
                request.get("codegen"),
            )
            if screened and launch is not None:
                # Registering the plugin compiles what an engine would embed,
                # so the screen rejects what only the plugin path refuses.
                _run_stage("compile", _register_plugin, reference, registration)
                return _launcher(
                    launch, proposal, output_specs, directory / name, aligned
                )
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

    try:
        execute = build("candidate", candidate)
    except _EvaluationError as error:
        fitted = _fit_shared_memory(
            backend,
            request,
            candidate,
            (input_specs, output_specs),
            directory / "candidate",
            error,
            {} if fit is None else fit,
        )
        if fitted is None:
            raise
        # The fitted variant replaces the candidate for every later check.
        candidate = fitted
        if diagnostics is not None:
            diagnostics["candidate"] = _candidate_diagnostics(candidate, screened)
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
                aligned=aligned,
            )
        except _EvaluationError as error:
            if error.stage == "compile" and diagnostics is not None:
                diagnostics["candidate"]["compilation_status"] = "failed"
            raise
    tests = 0
    for number, (label, inputs) in enumerate(cases()):
        if kept:
            expected = expectations[number]
        elif number == 0:
            expected = first_expected
        else:
            expected = expect(label, inputs)
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
            aligned=aligned or not screened,
        )
        tests += 1
        del inputs, expected
    del expectations
    # Timers capture CUDA graphs whose memory pools stay reserved, so they
    # follow the correctness checks. Eager PyTorch is timed at PyTorch's
    # default TF32 settings, as a model runs it, whatever its checks used.
    timers = {"reference": _capture_timer(reference, first, iterations)}
    if baseline is not None:
        timers["baseline"] = _run_stage(
            "baseline", _capture_timer, baseline, first, iterations
        )
    # The candidate and the incumbent run as the correctness checks ran them.
    timed = first if aligned or not screened else tuple(map(_unaligned, first))
    timers["candidate"] = _run_stage(
        "benchmark", _capture_timer, execute, timed, iterations
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
                timed,
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
            if oracle is None
            else f"{(oracle.basis or 'float64').partition(' ')[0]}_reference"
        ),
        "inputs": "sample-derived" if samples else "unconstrained N(0,1)",
        "guard_check": "unavailable" if launch is None else "passed",
    }
    if oracle is not None:
        result["oracle"] = oracle.basis
        result["fp32_matmul_precision"] = precision
    if unchecked:
        # Cases that checked less than half of an output: eager PyTorch's
        # output is NaN or infinite there, so any candidate value passed.
        result["inconclusive_cases"] = dict(unchecked)
    if robust and oracle is not None:
        # Cases beyond the sampled range, checked for finiteness and a
        # _ROBUSTNESS_FLOOR relative error rather than the in-range bounds.
        result["robustness_cases"] = sorted(robust)
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
    if getattr(error, "retryable", True) is False:
        # Evaluating again cannot change the outcome; the controller does not.
        result["retryable"] = False
    details = getattr(error, "diagnostics", None)
    if details is not None:
        result["diagnostics"] = details
    # A failure after a shared-memory fit belongs to the fitted variant.
    result.update(getattr(error, "fit", None) or {})
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


# The controller kills a worker as soon as its time limit is up. The worker's
# own deadline, this much later, only ends a worker whose controller is gone.
_DEADLINE_MARGIN_S = 30.0
# Seconds between checks that the controller that started the worker is alive.
_PARENT_POLL_S = 1.0


def _end_orphan() -> None:
    """End the worker as its controller would have: with its process group."""
    import signal

    # The controller starts every worker in a session of its own. A worker
    # started otherwise, as by a test, must not signal its caller's group.
    if os.getpgrp() == os.getpid():
        os.killpg(os.getpid(), signal.SIGKILL)
    os._exit(1)


def _watch_parent(
    parent: int,
    end: Callable[[], Any] = _end_orphan,
    interval: float = _PARENT_POLL_S,
) -> None:
    """Call ``end`` once ``parent`` is no longer this process's parent.

    An orphan is adopted by init or a subreaper, so its parent id changes.
    """
    while os.getppid() == parent:
        time.sleep(interval)
    end()


def _bound_lifetime(timeout_s: float | None, parent: int | None) -> None:
    """End the worker even when the controller that started it cannot.

    The controller kills the worker's process group when an evaluation ends
    or times out. If the controller is itself killed by a signal that skips
    Python's cleanup (SIGKILL, or SIGTERM or SIGHUP without a handler), that
    never happens, and the worker, which runs in a session of its own, gets
    neither the signal nor a terminal hangup. Such an orphan would hold the
    GPU through the evaluations that other compilations start next, skewing
    their timings. So the worker ends itself once its parent is no longer
    ``parent``, checked every ``_PARENT_POLL_S`` seconds, and in any case
    ``_DEADLINE_MARGIN_S`` after ``timeout_s``, writing every thread's
    traceback to stderr first. Either is skipped when its argument is None.
    """
    import faulthandler
    import threading

    if timeout_s is not None:
        try:
            faulthandler.dump_traceback_later(timeout_s + _DEADLINE_MARGIN_S, exit=True)
        except OverflowError:
            pass  # Centuries away, which is no deadline at all.
    if parent is not None:
        threading.Thread(
            target=_watch_parent, args=(parent,), name="parent-watchdog", daemon=True
        ).start()


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
    parser.add_argument(
        "--timeout",
        type=float,
        help="Seconds the controller waits for the result before killing the worker",
    )
    parser.add_argument(
        "--parent",
        type=int,
        help="Process id of the controller; the worker ends if that process dies",
    )
    args = parser.parse_args()
    _bound_lifetime(args.timeout, args.parent)
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
