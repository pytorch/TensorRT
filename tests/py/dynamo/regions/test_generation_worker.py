# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""The generation executor owns correctness checks and failure reporting."""

import json
import subprocess
import sys

import pytest
import torch

import torch_tensorrt
from torch_tensorrt.dynamo.region_compilation._generation import _worker


def _spec(shape=(7,), dtype="float32"):
    return {"shape": list(shape), "dtype": dtype, "device": "cpu"}


def test_validation_rejects_a_wrong_candidate_and_input_mutation():
    inputs = (torch.arange(7, dtype=torch.float32),)
    reference = lambda x: x + 1
    with pytest.raises(AssertionError):
        _worker._verify_case(
            reference,
            lambda x: x + 2,
            inputs,
            [_spec()],
            rtol=None,
            atol=None,
        )
    with pytest.raises(AssertionError):
        _worker._verify_case(
            reference,
            lambda x: x.add_(1),
            inputs,
            [_spec()],
            rtol=None,
            atol=None,
        )


def test_validation_handles_multiple_outputs_and_preserves_nonfinite_semantics():
    inputs = (torch.tensor([float("nan"), float("inf"), -float("inf"), -1.0]),)

    def reference(x):
        return torch.relu(x), x + 1

    _worker._verify_case(
        reference,
        reference,
        inputs,
        [_spec((4,)), _spec((4,))],
        rtol=None,
        atol=None,
    )
    with pytest.raises(AssertionError):
        _worker._verify_case(
            reference,
            lambda x: (torch.nan_to_num(torch.relu(x)), x + 1),
            inputs,
            [_spec((4,)), _spec((4,))],
            rtol=None,
            atol=None,
        )


@pytest.mark.parametrize(
    "candidate",
    [lambda x: x.double(), lambda x: x[:2], lambda x: (x, x)],
)
def test_validation_rejects_boundary_contract_mismatches(candidate):
    with pytest.raises(ValueError, match="captured"):
        _worker._verify_case(
            lambda x: x.clone(),
            candidate,
            (torch.zeros(7),),
            [_spec()],
            rtol=None,
            atol=None,
        )


def test_input_cases_cover_scalar_nonfinite_and_independent_random_inputs():
    cases = list(_worker._input_cases([_spec(()), _spec((7,))], 2, True))
    assert len(cases) == 11
    assert not torch.equal(cases[0][1], cases[1][1])
    assert torch.count_nonzero(cases[2][1]) == 0
    assert torch.all(cases[3][1] == -1)
    assert torch.all(cases[4][1] == 1)
    assert torch.isnan(cases[5][0]) and torch.isfinite(cases[5][1]).all()
    assert torch.isposinf(cases[6][0])
    assert torch.isneginf(cases[7][0])
    assert torch.isnan(cases[8][1][0]) and torch.isfinite(cases[8][0])


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_seeded_cases_are_repeatable_and_holdout_uses_new_random_values(dtype):
    specs = [_spec((16,), dtype), _spec((), dtype)]
    first = list(_worker._input_cases(specs, 2, True, seed=123))
    again = list(_worker._input_cases(specs, 2, True, seed=123))
    holdout = list(_worker._input_cases(specs, 2, True, seed=456))
    for case, repeated in zip(first, again):
        for value, wanted in zip(case, repeated):
            torch.testing.assert_close(value, wanted, rtol=0, atol=0, equal_nan=True)
    assert not torch.equal(first[0][0], holdout[0][0])
    assert not torch.equal(first[0][1], holdout[0][1])
    limits = torch.finfo(getattr(torch, dtype))
    assert first[0][0][-3].item() == 1 + limits.eps
    assert first[0][0][-2].item() == 1 - limits.eps / 2
    assert first[0][0][-1].item() == limits.tiny


def test_benchmark_interleaves_arms_and_reports_spread(monkeypatch):
    calls = []
    data = {"candidate": iter([3.0, 1.0, 2.0]), "incumbent": iter([5.0, 4.0, 6.0])}

    def capture(name, inputs, iterations):
        calls.append("capture:" + name)

        def measure():
            calls.append(name)
            return next(data[name])

        return measure

    monkeypatch.setattr(_worker, "_capture_timer", capture)
    results = _worker._benchmark(
        {"candidate": "candidate", "incumbent": "incumbent"}, (), 3, repeats=3
    )
    assert calls == [
        "capture:candidate",
        "capture:incumbent",
        "candidate",
        "incumbent",
        "incumbent",
        "candidate",
        "candidate",
        "incumbent",
    ]
    assert results["candidate"] == {
        "samples_ms": [3.0, 1.0, 2.0],
        "median_ms": 2.0,
        "spread_ms": 2.0,
        "relative_spread": 1.0,
        "mad_ms": 1.0,
    }
    assert results["incumbent"]["median_ms"] == 5.0


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_benchmark_rejects_invalid_measurements(monkeypatch, value):
    monkeypatch.setattr(_worker, "_capture_timer", lambda *args: lambda: value)
    with pytest.raises(RuntimeError, match="positive finite"):
        _worker._benchmark({"candidate": lambda: None}, (), 1)


def test_benchmark_reference_brackets_all_candidate_samples(monkeypatch):
    calls = []

    def capture(name, inputs, iterations):
        def measure():
            calls.append(name)
            return 1.0

        return measure

    monkeypatch.setattr(_worker, "_capture_timer", capture)
    results = _worker._benchmark(
        {"reference": "reference", "candidate": "candidate"}, (), 1, repeats=3
    )
    assert calls[0] == calls[-1] == "reference"
    assert len(results["candidate"]["samples_ms"]) == 3
    assert len(results["reference"]["samples_ms"]) == 5


def test_evaluation_reports_source_errors_before_building_an_engine(tmp_path):
    with pytest.raises(_worker._EvaluationError) as failure:
        _worker.evaluate(
            {
                "candidate": {},
                "backend": "triton",
                "inputs": [_spec()],
                "outputs": [_spec()],
            },
            tmp_path,
        )
    assert failure.value.stage == "source"


@pytest.mark.parametrize(
    "identity, message",
    [
        ({"backend_version": "obsolete"}, "Backend version changed"),
        ({"proposal_schema_sha256": "0" * 64}, "Backend proposal schema changed"),
    ],
)
def test_evaluation_rejects_backend_drift_before_source_or_device_work(
    tmp_path, identity, message
):
    with pytest.raises(_worker._EvaluationError, match=message) as failure:
        _worker.evaluate({"backend": "triton", **identity}, tmp_path)
    assert failure.value.stage == "source"


def test_tensorrt_baseline_rejects_eager_fallback(monkeypatch):
    reference = torch.fx.symbolic_trace(torch.nn.ReLU())
    options = {}

    def compile(module, **kwargs):
        options.update(kwargs)
        return module

    monkeypatch.setattr(torch_tensorrt, "compile", compile)
    with pytest.raises(RuntimeError, match="did not produce a TensorRT engine"):
        _worker._build_reference_engine(reference, (torch.randn(7),))
    assert options["require_full_compilation"] is True
    assert options["min_block_size"] == 1


def test_error_report_retains_cause_and_stage_without_local_paths(tmp_path):
    try:

        def fail():
            raise ValueError(f"wrong value at {tmp_path}/candidate.py\x1b[31m")

        _worker._run_stage("correctness", fail)
    except _worker._EvaluationError as error:
        report = _worker._error_report(error, tmp_path, "holdout")
    assert report["stage"] == "correctness"
    assert report["validation_phase"] == "holdout"
    assert "wrong value at <evaluation>/candidate.py" in report["error"]
    assert str(tmp_path) not in report["error"]
    assert "\x1b" not in report["error"]


def test_restore_metadata_rebuilds_plugin_boundaries_without_intermediate_values():
    class Reference(torch.nn.Module):
        def forward(self, x, y):
            return x + y, x - y

    reference = torch.fx.symbolic_trace(Reference())
    inputs = (torch.randn(7), torch.randn(7))
    outputs = reference(*inputs)
    _worker._restore_metadata(reference, inputs, outputs)
    placeholders = [node for node in reference.graph.nodes if node.op == "placeholder"]
    assert placeholders[0].meta["val"] is inputs[0]
    output = next(node for node in reference.graph.nodes if node.op == "output")
    assert output.args[0][1].meta["val"] is outputs[1]


def test_main_reports_executor_failure_and_replaces_result_atomically(
    tmp_path, monkeypatch
):
    request = tmp_path / "request.json"
    result = tmp_path / "result.json"
    request.write_text(json.dumps({"candidate": {}}))
    result.write_text('{"status": "passed"}')

    def fail(*args):
        raise RuntimeError("candidate compilation failed")

    monkeypatch.setattr(_worker, "evaluate", fail)
    monkeypatch.setattr(
        sys, "argv", ["worker", "--request", str(request), "--result", str(result)]
    )
    _worker.main()
    report = json.loads(result.read_text())
    assert report["status"] == "error"
    assert report["stage"] == "compile"
    assert "candidate compilation failed" in report["error"]
    assert "latency_ms" not in report
    assert not result.with_suffix(".json.tmp").exists()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    reason="CUDA and TensorRT QDP support are required",
)
@pytest.mark.parametrize(
    "correct, objective", [(True, "eager"), (True, "tensorrt"), (False, "candidate")]
)
def test_worker_checks_a_multi_output_kernel_in_a_real_engine(
    tmp_path, correct, objective
):
    pytest.importorskip("triton")

    class Reference(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x + 0.25), x * x

    reference_path = tmp_path / "reference.pt"
    torch.save(torch.fx.symbolic_trace(Reference()), reference_path)
    source = """
import triton
import triton.language as tl

@triton.jit
def generated(arg0, out0, out1, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(arg0 + offsets, offsets < N, other=0.0)
    shifted = values + 0.25
    activated = tl.where(shifted < 0.0, 0.0, shifted)
    tl.store(out0 + offsets, activated, offsets < N)
    tl.store(out1 + offsets, values * values, offsets < N)
"""
    if not correct:
        source = source.replace("values + 0.25", "values + 1.0")
    spec = {"shape": [3, 129], "dtype": "float32", "device": "cuda:0"}
    request = {
        "reference_path": str(reference_path),
        "backend": "triton",
        "candidate": {
            "source": source,
            "kernel_name": "generated",
            "constants": [
                {"name": "N", "value": 387},
                {"name": "BLOCK", "value": 128},
            ],
            "grid": [4],
            "num_warps": 4,
            "num_stages": None,
        },
        "inputs": [spec],
        "outputs": [spec, spec],
        "num_tests": 2,
        "check_nonfinite": True,
        "benchmark_iters": 3,
        "rtol": None,
        "atol": None,
        "validation_seed": 37,
        "validation_phase": "holdout",
        "objective": objective,
        "benchmark_repeats": 3,
    }
    if correct:
        request["incumbent"] = dict(request["candidate"])
    request_path, result_path = tmp_path / "request.json", tmp_path / "result.json"
    request_path.write_text(json.dumps(request))
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch_tensorrt.dynamo.region_compilation._generation._worker",
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(result_path.read_text())
    if correct:
        assert report["status"] == "passed", report
        assert report["measurement"] == "cuda_graph"
        assert report["tests"] == 8
        assert report["latency_ms"] > 0
        assert report["reference_latency_ms"] > 0
        assert report["incumbent_latency_ms"] > 0
        assert report["baseline_latency_ms"] > 0
        assert len(report["benchmark"]["candidate"]["samples_ms"]) == 3
        assert report["validation_phase"] == "holdout"
    else:
        assert report["status"] == "error", report
        assert report["stage"] == "correctness"
        assert "Tensor-likes are not close" in report["error"]
