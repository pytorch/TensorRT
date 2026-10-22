# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate generated kernels in a disposable process and a real TRT engine."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import operator
import secrets
import statistics
import traceback
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch


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
    specs: Sequence[dict[str, Any]], num_tests: int, check_nonfinite: bool
) -> Iterator[tuple[torch.Tensor, ...]]:
    if not specs or num_tests < 1:
        raise ValueError("At least one input and one random test are required")
    device = torch.device(specs[0]["device"])
    generator = torch.Generator(device=device).manual_seed(secrets.randbits(63))

    def random_inputs() -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.randn(
                tuple(spec["shape"]),
                dtype=_dtype(spec["dtype"]),
                device=spec["device"],
                generator=generator,
            )
            for spec in specs
        )

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
    engines = [
        item for item in compiled.modules() if isinstance(item, TorchTensorRTModule)
    ]
    if len(engines) != 1 or str(op) not in engines[0].get_layer_info():
        raise RuntimeError(
            f"Generated kernel {op} was not placed inside a TensorRT engine: "
            f"{[engine.get_layer_info() for engine in engines]}"
        )
    if any(
        node.op == "call_function" and node.target == op
        for item in compiled.modules()
        if isinstance(item, torch.fx.GraphModule)
        for node in item.graph.nodes
    ):
        raise RuntimeError("Generated kernel has an eager fallback outside TensorRT")
    return compiled


def _latency(
    function: Callable[..., Any], inputs: tuple[torch.Tensor, ...], iterations: int
) -> float:
    """Measure captured GPU work without per-invocation Python launch gaps."""
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
    samples = []
    with torch.cuda.stream(stream):
        for _ in range(3):
            graph.replay()
        stream.synchronize()
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / iterations)
    latency = statistics.median(samples)
    if not math.isfinite(latency) or latency <= 0:
        raise RuntimeError("Benchmark did not produce a positive finite latency")
    return float(latency)


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
    from ._candidate import make_registration

    if not torch.cuda.is_available():
        raise RuntimeError("Generated-kernel evaluation requires CUDA")
    input_specs, output_specs = request["inputs"], request["outputs"]
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
    cases = _input_cases(input_specs, request["num_tests"], request["check_nonfinite"])
    first = next(cases)
    expected = _outputs(reference(*(item.clone() for item in first)))
    _check_contract(expected, output_specs)
    _restore_metadata(reference, first, expected)
    registration = make_registration(
        request["candidate"], request["backend"], input_specs, output_specs, directory
    )
    compiled = _build_engine(reference, registration, first, len(output_specs))
    tests = 0
    for inputs in itertools.chain((first,), cases):
        try:
            _verify_case(
                reference,
                compiled,
                inputs,
                output_specs,
                rtol=request.get("rtol"),
                atol=request.get("atol"),
            )
        except Exception as error:
            raise RuntimeError(f"Correctness case {tests + 1} failed") from error
        tests += 1
    iterations = request["benchmark_iters"]
    before = _latency(reference, first, iterations)
    latency = _latency(compiled, first, iterations)
    after = _latency(reference, first, iterations)
    return {
        "status": "passed",
        "measurement": "cuda_graph",
        "latency_ms": latency,
        "reference_latency_ms": statistics.median([before, after]),
        "reference_before_ms": before,
        "reference_after_ms": after,
        "tests": tests,
    }


def _write_result(path: Path, result: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        with torch.inference_mode():
            result = evaluate(request, args.request.parent)
    except Exception:
        result = {"status": "error", "error": traceback.format_exc()[-16000:]}
    _write_result(args.result, result)


if __name__ == "__main__":
    main()
