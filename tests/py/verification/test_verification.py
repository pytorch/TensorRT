# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU correctness coverage without importing the native compiler package."""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import pytest
import torch


def _load_package(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(
        name, path / "__init__.py", submodule_search_locations=[str(path)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_source = Path(__file__).resolve().parents[3] / "py" / "torch_tensorrt"
_reporting = _load_package("_verification_test_reporting", _source / "reporting")
_previous_reporting = sys.modules.get("torch_tensorrt.reporting")
sys.modules["torch_tensorrt.reporting"] = _reporting
try:
    _verification = _load_package("_verification_test_core", _source / "verification")
finally:
    if _previous_reporting is None:
        sys.modules.pop("torch_tensorrt.reporting", None)
    else:
        sys.modules["torch_tensorrt.reporting"] = _previous_reporting

InputSample = _verification.InputSample
VerificationError = _verification.VerificationError
validate = _verification.validate
benchmark = _verification.benchmark


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


class Constant(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return self.value


def _compare(reference, actual, **kwargs):
    return validate(
        Constant(reference),
        Constant(actual),
        [InputSample(args=(torch.ones(1),))],
        **kwargs,
    )


@pytest.mark.parametrize("value", [2**24, 2**53, 2**63 - 2])
def test_integer_equality_is_exact_despite_float_rounding(value):
    report = _compare(
        torch.tensor([value], dtype=torch.int64),
        torch.tensor([value + 1], dtype=torch.int64),
        atol=1e10,
        rtol=1e10,
    )
    assert not report.passed
    assert report.outputs[0].max_abs_error == 1
    assert report.outputs[0].mismatch_count == 1
    assert report.outputs[0].reason == "exact_value_mismatch"
    assert report.failure.category.value == "validation_error"
    assert json.loads(report.to_json())["status"] == "failed"


def test_int64_subtraction_cannot_wrap():
    report = _compare(
        torch.tensor([-(2**63)], dtype=torch.int64),
        torch.tensor([2**63 - 1], dtype=torch.int64),
    )
    assert not report.passed
    assert report.outputs[0].max_abs_error == float(2**64 - 1)


def test_bool_equality_is_exact():
    assert not _compare(
        torch.tensor([True, False]), torch.tensor([True, True]), atol=100
    ).passed


def test_float64_small_error_survives_comparison():
    report = _compare(
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([1.0 + 1e-10], dtype=torch.float64),
        atol=0,
        rtol=0,
    )
    assert not report.passed
    assert report.outputs[0].max_abs_error > 0


def test_complex_comparison_uses_magnitude_and_preserves_precision():
    reference = torch.tensor([1 + 2j, 0j], dtype=torch.complex128)
    actual = reference + torch.tensor([3e-8 + 4e-8j, 0j], dtype=torch.complex128)
    assert _compare(reference, actual, atol=6e-8, rtol=0).passed
    report = _compare(reference, actual, atol=4e-8, rtol=0)
    assert not report.passed
    assert report.outputs[0].max_abs_error == pytest.approx(5e-8)


def test_finite_extreme_values_do_not_pass_due_to_overflow():
    report = _compare(
        torch.tensor([-1.7e308], dtype=torch.float64),
        torch.tensor([1.7e308], dtype=torch.float64),
        atol=0,
        rtol=1.5,
    )
    assert not report.passed
    assert report.outputs[0].max_abs_error is None
    assert report.outputs[0].max_relative_error == 2
    assert report.to_json()
    complex_value = torch.tensor([complex(1.7e308, 1.7e308)], dtype=torch.complex128)
    assert _compare(complex_value, complex_value, atol=0, rtol=0).passed


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_outputs_fail_even_when_equal_and_serialize(value):
    report = _compare(torch.tensor([value]), torch.tensor([value]))
    assert not report.passed
    assert report.outputs[0].reason == "nonfinite_output"
    assert report.outputs[0].mismatch_count == 1
    assert report.outputs[0].max_abs_error is None
    json.loads(report.to_json(), parse_constant=lambda value: pytest.fail(value))


def test_empty_outputs_have_explicit_policy():
    report = _compare(torch.empty(0, 3), torch.empty(0, 3))
    assert report.passed
    assert report.outputs[0].reason == "empty_tensor"
    assert report.outputs[0].mean_abs_error is None
    assert not _compare(torch.empty(0, 3), torch.empty(0, 4)).passed


@pytest.mark.parametrize(
    "actual,reason",
    [
        ([torch.ones(1)], "pytree_structure_mismatch"),
        ((torch.ones(2),), "shape_mismatch"),
        ((torch.ones(1, dtype=torch.float64),), "dtype_mismatch"),
    ],
)
def test_output_structure_shape_and_dtype_are_checked(actual, reason):
    report = _compare((torch.ones(1),), actual)
    assert not report.passed
    assert report.outputs[0].reason == reason


def test_unsupported_output_device_is_reported():
    report = _compare(torch.ones(1), torch.ones(1, device="meta"))
    assert not report.passed
    assert report.outputs[0].reason == "unsupported_tensor_device"


def test_nested_scalar_outputs_and_per_sample_profile_evidence():
    output = {
        "tensor": torch.tensor(2.0),
        "meta": (None, True, 123, "ok", 0.125, 2 + 1j),
    }
    samples = [
        InputSample(
            args=(torch.ones(1),), sample_id="min", profile="min", label="edge"
        ),
        InputSample(args=(torch.ones(1),), sample_id="max", profile="max"),
    ]
    report = validate(Constant(output), Constant(output), samples, job_id="job")
    assert report.passed and report.sample_count == report.passed_samples == 2
    assert report.profile_points == ["min", "max"]
    assert report.samples[0].label == "edge"
    assert len(report.samples[0].input_fingerprint) == 64
    assert "output['meta'][2]" in {comparison.path for comparison in report.outputs}
    assert _reporting.ValidationReport.from_json(report.to_json()) == report


def test_input_sample_rejects_non_sequence_args():
    with pytest.raises(VerificationError, match="args must be a tuple or list"):
        InputSample(args=torch.ones(1))


def test_input_sample_rejects_non_dict_kwargs():
    with pytest.raises(VerificationError, match="kwargs must be a dict"):
        InputSample(args=(), kwargs=[("value", torch.ones(1))])


def test_input_fingerprint_rejects_non_finite_python_float():
    sample = InputSample(args=(float("nan"),))
    with pytest.raises(VerificationError, match="finite"):
        validate(torch.nn.Identity(), torch.nn.Identity(), [sample])


def test_noniterable_samples_raise_verification_error():
    with pytest.raises(VerificationError, match="nonempty iterable"):
        validate(torch.nn.Identity(), torch.nn.Identity(), None)


def test_relative_statistics_and_near_zero_pass_rule_are_separate():
    report = _compare(
        torch.tensor([0.0, 2.0], dtype=torch.float64),
        torch.tensor([0.001, 2.002], dtype=torch.float64),
        atol=0.0011,
        rtol=0.001,
        relative_floor=0.01,
    )
    assert report.passed
    assert report.outputs[0].max_relative_error == pytest.approx(0.1)
    assert report.relative_floor == 0.01


class Aliasing(torch.nn.Module):
    def forward(self, left, right, *, duplicate):
        assert left is duplicate
        assert left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        assert right.storage_offset() == 1
        assert left.requires_grad
        return left[1:3] + right[:2]


def test_input_snapshot_preserves_aliasing_views_and_requires_grad():
    source = torch.arange(4.0, requires_grad=True)
    sample = InputSample(args=(source, source[1:]), kwargs={"duplicate": source})
    assert validate(Aliasing(), Aliasing(), [sample]).passed
    assert torch.equal(source, torch.arange(4.0))


class RetainsCallerModule(torch.nn.Module):
    def __deepcopy__(self, memo):
        return self

    def forward(self, x):
        return x


def test_isolated_module_preserves_specific_verification_errors():
    with pytest.raises(VerificationError, match="still shares submodule objects"):
        validate(
            RetainsCallerModule(),
            torch.nn.Identity(),
            [InputSample(args=(torch.ones(1),))],
        )


class MutatesInput(torch.nn.Module):
    def forward(self, x):
        x.add_(1)
        return x


class MutatesBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("counter", torch.zeros(1))

    def forward(self, x):
        self.counter.add_(1)
        return x + self.counter


@pytest.mark.parametrize("api", ["validate", "benchmark"])
@pytest.mark.parametrize("kind", ["input", "buffer"])
def test_stateless_protocol_rejects_mutation_and_preserves_caller(kind, api):
    model = MutatesInput() if kind == "input" else MutatesBuffer()
    model.train()
    x = torch.ones(3)
    sample = InputSample(args=(x,))
    with pytest.raises(VerificationError, match="stateless"):
        if api == "validate":
            validate(model, model, [sample])
        else:
            benchmark(model, model, sample, warmup=0, iterations=1)
    assert model.training
    assert torch.equal(x, torch.ones(3))
    if kind == "buffer":
        assert model.counter.item() == 0


class RandomOutput(torch.nn.Module):
    def forward(self, x):
        assert not self.training
        assert torch.is_inference_mode_enabled()
        return x + torch.rand_like(x) + random.random()


def test_reproducible_calls_restore_rng_training_modes_and_inputs():
    model = RandomOutput().train()
    sample = InputSample(args=(torch.ones(3),))
    torch_state = torch.random.get_rng_state().clone()
    python_state = random.getstate()
    report = validate(model, model, [sample, sample], seed=99)
    assert report.passed
    benchmark(model, model, sample, warmup=1, iterations=2, seed=99)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    assert model.training


def test_cpu_benchmark_protocol_statistics_speedup_and_throughput():
    report = benchmark(
        Identity(),
        Identity(),
        InputSample(args=(torch.ones(3),)),
        warmup=1,
        iterations=5,
        percentiles=(0, 50, 95, 100),
        batch_size=3,
    )
    assert report.device == "cpu"
    assert (
        len(report.reference.wall_samples_ms)
        == len(report.compiled.wall_samples_ms)
        == 5
    )
    assert not report.reference.device_samples_ms
    assert all(
        math.isfinite(value) and value > 0 for value in report.compiled.wall_samples_ms
    )
    assert (
        report.speedup["wall"]
        == report.reference.wall_statistics["median"]
        / report.compiled.wall_statistics["median"]
    )
    assert report.compiled.wall_statistics["p0"] == min(report.compiled.wall_samples_ms)
    assert report.compiled.wall_statistics["p100"] == max(
        report.compiled.wall_samples_ms
    )
    assert (
        report.compiled.wall_statistics["p50"]
        == report.compiled.wall_statistics["median"]
    )
    assert (
        report.throughput["compiled_items_per_second"]
        == 3 * report.throughput["compiled_invocations_per_second"]
    )
    assert not report.transfers["included"] and not report.preprocessing["included"]
    assert _reporting.BenchmarkReport.from_json(report.to_json()) == report


def test_exported_graph_round_trip_keeps_shape_constraints(tmp_path):
    class Add(torch.nn.Module):
        def forward(self, x):
            return x + x

    eager = Add().eval()
    batch = torch.export.Dim("batch", min=1, max=4)
    ep = torch.export.export(
        eager, (torch.ones(2, 3),), dynamic_shapes={"x": {0: batch}}
    )
    path = tmp_path / "model.pt2"
    torch.export.save(ep, path)
    module = torch.export.load(path).module()
    report = validate(
        eager,
        module,
        [
            InputSample(args=(torch.ones(1, 3),), profile="min"),
            InputSample(args=(torch.ones(4, 3),), profile="max"),
        ],
    )
    assert report.passed
    benchmark(
        eager, module, InputSample(args=(torch.ones(2, 3),)), warmup=0, iterations=2
    )
    with pytest.raises((RuntimeError, AssertionError), match="4"):
        validate(eager, module, [InputSample(args=(torch.ones(5, 3),))])


@pytest.mark.parametrize(
    "kwargs",
    [{"atol": -1}, {"rtol": float("nan")}, {"relative_floor": 0}, {"seed": -1}],
)
def test_invalid_validation_protocol_is_rejected(kwargs):
    with pytest.raises(VerificationError):
        validate(Identity(), Identity(), [InputSample(args=(torch.ones(1),))], **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"warmup": -1},
        {"percentiles": None},
        {"percentiles": [float("inf")]},
        {"percentiles": [101]},
        {"batch_size": 0},
    ],
)
def test_invalid_benchmark_protocol_is_rejected(kwargs):
    with pytest.raises(VerificationError):
        benchmark(Identity(), Identity(), InputSample(args=(torch.ones(1),)), **kwargs)
