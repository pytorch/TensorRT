# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""The generation executor owns correctness checks and failure reporting."""

import gc
import itertools
import json
import math
import os
import re
import subprocess
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch

import torch_tensorrt
from torch_tensorrt.dynamo.region_compilation import _plugins
from torch_tensorrt.dynamo.region_compilation._generation import (
    _backends,
    _util,
    _worker,
)
from torch_tensorrt.dynamo.region_compilation._generation._util import digest

from ._generation_utils import save_reference

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Kernel generation requires a POSIX host"
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)
DEFAULT_COMPILE_OPTIONS = {
    "disable_tf32": False,
    "optimization_level": None,
    "use_fp32_acc": False,
    "tiling_optimization_level": "none",
    "version_compatible": False,
}


def _spec(shape=(7,), dtype="float32", device="cpu"):
    return {"shape": list(shape), "dtype": dtype, "device": device}


def _boundary(shape=(7,), dtype=torch.float32, device="cpu"):
    return _plugins._TensorBoundary(tuple(shape), dtype, torch.device(device))


def _check(reference, candidate, inputs, specs, *, rtol=None, atol=None):
    expected = _worker._expected(
        reference, inputs, specs, exact_reference=reference if rtol is None else None
    )
    _worker._verify_case(
        candidate, inputs, expected, specs, "random 1", rtol=rtol, atol=atol
    )


def test_request_specs_become_tensor_boundaries():
    specs = _worker._specs([_spec((3, 4), "bfloat16", "cuda:0"), _spec((), "float16")])
    assert specs == [
        _boundary((3, 4), torch.bfloat16, "cuda:0"),
        _boundary((), torch.float16),
    ]
    with pytest.raises(ValueError, match="Unsupported generated-kernel dtype"):
        _worker._specs([_spec(dtype="float64")])


def test_validation_names_a_wrong_output_and_a_modified_input():
    inputs = (torch.arange(7, dtype=torch.float32),)
    reference = lambda x: (x + 1, x * 2)
    specs = [_boundary(), _boundary()]
    with pytest.raises(
        _worker._EvaluationError, match=r"Output 1 is wrong for case 'random 1': got 2"
    ) as failure:
        _check(reference, lambda x: (x + 1, x * 2 + 2), inputs, specs)
    assert failure.value.stage == "correctness"
    with pytest.raises(
        _worker._EvaluationError, match="Input 0 was modified on case 'random 1'"
    ):
        _check(reference, lambda x: (x.add_(1), x * 2), inputs, specs)


def test_validation_preserves_nonfinite_semantics():
    inputs = (torch.tensor([float("nan"), float("inf"), -float("inf"), -1.0]),)
    reference = lambda x: (torch.relu(x), x + 1)
    specs = [_boundary((4,)), _boundary((4,))]
    _check(reference, reference, inputs, specs)
    with pytest.raises(_worker._EvaluationError, match="Output 0 .* got 0 at index"):
        _check(
            reference, lambda x: (torch.nan_to_num(torch.relu(x)), x + 1), inputs, specs
        )
    with pytest.raises(_worker._EvaluationError, match="Output 1 .* got inf"):
        _check(reference, lambda x: (torch.relu(x), (x + 1).abs()), inputs, specs)


@pytest.mark.parametrize(
    "candidate",
    [lambda x: x.double(), lambda x: x[:2], lambda x: (x, x), lambda x: x.t()],
)
def test_validation_rejects_boundary_contract_mismatches(candidate):
    with pytest.raises(ValueError, match="captured"):
        _check(
            lambda x: x.clone(), candidate, (torch.zeros(7, 7),), [_boundary((7, 7))]
        )


def test_explicit_tolerances_compare_with_the_eager_reference():
    inputs = (torch.linspace(1, 2, 7),)
    expected = _worker._expected(
        lambda x: x * 3, inputs, [_boundary()], exact_reference=None
    )
    assert expected.oracle is None
    nearby = lambda x: x * 3 + 1e-3
    _worker._verify_case(
        nearby, inputs, expected, [_boundary()], "random 1", rtol=1e-2, atol=1e-2
    )
    with pytest.raises(_worker._EvaluationError, match="Tensor-likes are not close"):
        _worker._verify_case(
            nearby, inputs, expected, [_boundary()], "random 1", rtol=1e-5, atol=1e-5
        )


def _correctly_rounded_mm(a, b):
    return torch.mm(a.double(), b.double()).to(a.dtype)


def _tiled_row_sum(x):  # the block-tiled order a typical generated kernel uses
    return x.view(x.shape[0], -1, 128).sum(-1).sum(-1)


@requires_cuda
@pytest.mark.parametrize(
    "dtype, reference, candidate, shapes, output",
    [
        ("float32", lambda x: x.sum(-1), _tiled_row_sum, [(64, 4096)], (64,)),
        *(
            (
                dtype,
                torch.mm,
                _correctly_rounded_mm,
                [(128, 4096), (4096, 128)],
                (128, 128),
            )
            for dtype in ("float32", "float16", "bfloat16")
        ),
    ],
)
def test_default_tolerance_accepts_reassociated_reductions(
    dtype, reference, candidate, shapes, output
):
    specs = _worker._specs([_spec(shape, dtype, "cuda:0") for shape in shapes])
    outputs = [_boundary(output, specs[0].dtype, "cuda:0")]
    for seed in range(3):
        for label, inputs in _worker._input_cases(specs, 3, True, seed):
            expected = _worker._expected(
                reference, inputs, outputs, exact_reference=reference
            )
            assert expected.oracle is not None
            _worker._verify_case(
                candidate, inputs, expected, outputs, label, rtol=None, atol=None
            )


@requires_cuda
def test_default_tolerance_rejects_cancellation_on_offset_inputs():
    def layer_norm(x):
        return torch.nn.functional.layer_norm(x, x.shape[-1:])

    def one_pass(x):  # E[x^2] - E[x]^2 cancels catastrophically for offset inputs.
        mean = x.mean(-1, keepdim=True)
        variance = (x * x).mean(-1, keepdim=True) - mean * mean
        return (x - mean) * torch.rsqrt(variance + 1e-5)

    specs = _worker._specs([_spec((64, 1024), device="cuda:0")])
    failures = {}
    for label, inputs in _worker._input_cases(specs, 3, False, 5):
        expected = _worker._expected(
            layer_norm, inputs, specs, exact_reference=layer_norm
        )
        try:
            _worker._verify_case(
                one_pass, inputs, expected, specs, label, rtol=None, atol=None
            )
        except _worker._EvaluationError as error:
            failures[label] = str(error)
    assert list(failures) == ["offset +100", "offset -100"]
    assert failures["offset +100"].startswith(
        "Output 0 is wrong for case 'offset +100'"
    )
    assert "exceeds the allowance" in failures["offset +100"]


def test_oracle_rule_accepts_eager_values_and_correct_rounding_on_overflow():
    eager = torch.tensor([math.nan, math.inf, math.inf, 1.0], dtype=torch.float16)
    # Eager overflowed: entry 0 to NaN through inf - inf, entry 2 in an
    # intermediate result that float64 keeps in range.
    oracle = torch.tensor([1e6, 1e6, 3.0, 1.0], dtype=torch.float64)
    accepted = [
        [math.nan, math.inf, math.inf, 1.0],  # the eager values
        [math.inf, math.inf, 3.0, 1.0],  # correctly rounded
    ]
    for values in accepted:
        actual = torch.tensor(values, dtype=torch.float16)
        assert _worker._oracle_mismatch(actual, eager, oracle) is None
    for values in ([-math.inf, math.inf, 3.0, 1.0], [math.inf, math.inf, 3.0, 1.01]):
        actual = torch.tensor(values, dtype=torch.float16)
        assert _worker._oracle_mismatch(actual, eager, oracle) is not None


def test_oracle_falls_back_to_eager_when_float64_is_unavailable():
    weight = torch.ones(7, 7)
    reference = lambda x: torch.mm(x, weight)  # float32 weight cannot meet float64
    inputs = (torch.randn(2, 7),)
    expected = _worker._expected(
        reference, inputs, [_boundary((2, 7))], exact_reference=reference
    )
    assert expected.oracle is None
    with pytest.raises(_worker._EvaluationError, match="Tensor-likes are not close"):
        _worker._check_outputs(
            (reference(*inputs) + 1,), expected, "case 'x'", rtol=None, atol=None
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_oracle_keeps_elementwise_outputs_tight(dtype):
    # Eager GELU only rounds, so each element keeps a few-ulp allowance: a
    # tensor-wide allowance would let the tanh approximation pass in half
    # precision.
    torch.manual_seed(0)
    x = torch.randn(64, 1024, dtype=dtype)
    eager, exact = torch.nn.functional.gelu(x), torch.nn.functional.gelu(x.double())
    approximate = torch.nn.functional.gelu(x.float(), approximate="tanh").to(dtype)
    problem = _worker._oracle_mismatch(approximate, eager, exact)
    assert problem is not None
    if dtype != torch.float32:
        assert "units in the last place" in problem
    rounded = torch.nn.functional.gelu(x.float()).to(dtype)
    assert _worker._oracle_mismatch(rounded, eager, exact) is None


@requires_cuda
def test_softmax_to_float32_regions_keep_a_float64_reference():
    reference = torch.fx.symbolic_trace(
        lambda x: torch.ops.aten._softmax.default(x, -1, True)
    )
    inputs = (torch.randn(4, 8, dtype=torch.float16, device="cuda"),)
    expected = _worker._expected(
        reference,
        inputs,
        [_boundary((4, 8), torch.float32, "cuda:0")],
        exact_reference=_worker._float64_reference(reference),
    )
    assert expected.oracle is not None and expected.oracle[0].dtype == torch.float64


@requires_cuda
def test_guard_bands_catch_a_read_that_changes_an_output():
    x = torch.randn(1000, device="cuda")
    shifted = torch.cat([x[1:], x.new_full((1,), -math.inf)])
    expected = _worker._Expected((torch.maximum(x, shifted),), None)

    def launch(inputs, outputs):
        # Compares each element with its successor, including one past the end.
        source = inputs[0]
        beyond = source.as_strided(source.shape, (1,), source.storage_offset() + 1)
        outputs[0].copy_(torch.maximum(source, beyond))

    with pytest.raises(_worker._EvaluationError, match="reads memory outside"):
        _worker._check_guard_bands(launch, (x,), expected, "random 1", rtol=0, atol=0)


def test_input_cases_are_labelled_and_cover_magnitudes_and_nonfinite_values():
    cases = dict(_worker._input_cases([_boundary(()), _boundary((64,))], 2, True, 3))
    assert list(cases) == [
        "random 1",
        "random 2",
        "scaled x0.001",
        "scaled x100",
        "offset +100",
        "offset -100",
        "constant 0.0",
        "constant -1.0",
        "constant 1.0",
        *(f"input {i}[0] = {v}" for i in range(2) for v in ("nan", "inf", "-inf")),
    ]
    assert not torch.equal(cases["random 1"][1], cases["random 2"][1])
    assert cases["scaled x0.001"][1].abs().max() < 0.01
    assert cases["scaled x100"][1].abs().max() > 10
    assert (cases["offset +100"][1] - 100).abs().max() < 10
    assert (cases["offset -100"][1] + 100).abs().max() < 10
    assert torch.count_nonzero(cases["constant 0.0"][1]) == 0
    assert torch.all(cases["constant -1.0"][1] == -1)
    assert torch.isnan(cases["input 0[0] = nan"][0])
    assert torch.isfinite(cases["input 0[0] = nan"][1]).all()
    assert torch.isposinf(cases["input 0[0] = inf"][0])
    assert torch.isneginf(cases["input 1[0] = -inf"][1][0])
    assert torch.isfinite(cases["input 1[0] = -inf"][0])


def test_input_cases_end_with_the_captured_samples():
    specs = [_boundary((4,)), _boundary(())]
    samples = [
        (torch.arange(4.0), torch.tensor(5.0)),
        (torch.zeros(4), torch.tensor(-1.0)),
    ]
    cases = list(_worker._input_cases(specs, 1, False, 3, samples))
    assert len(cases) == 1 + 4 + 3 + 2
    assert [label for label, _ in cases[-2:]] == ["sample 1", "sample 2"]
    for (_, case), sample in zip(cases[-2:], samples):
        assert all(torch.equal(value, wanted) for value, wanted in zip(case, sample))


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_seeded_cases_are_repeatable_and_holdout_uses_new_random_values(dtype):
    specs = _worker._specs([_spec((16,), dtype), _spec((), dtype)])
    first = list(_worker._input_cases(specs, 2, True, 123))
    again = list(_worker._input_cases(specs, 2, True, 123))
    holdout = list(_worker._input_cases(specs, 2, True, 456))
    for (label, case), (repeated_label, repeated) in zip(first, again):
        assert label == repeated_label
        for value, wanted in zip(case, repeated):
            torch.testing.assert_close(value, wanted, rtol=0, atol=0, equal_nan=True)
    assert not torch.equal(first[0][1][0], holdout[0][1][0])
    assert not torch.equal(first[0][1][1], holdout[0][1][1])
    limits = torch.finfo(getattr(torch, dtype))
    assert first[0][1][0][-3].item() == 1 + limits.eps
    assert first[0][1][0][-2].item() == 1 - limits.eps / 2
    assert first[0][1][0][-1].item() == limits.tiny
    # Scaled cases probe magnitudes, not overflow, even in float16.
    for label, case in first[2:6]:
        assert all(torch.isfinite(value * value).all() for value in case), label


def _timers(**samples):
    calls = []

    def timer(name, values):
        values = iter(values)

        def measure():
            calls.append(name)
            return next(values)

        return measure

    return {name: timer(name, values) for name, values in samples.items()}, calls


def test_benchmark_interleaves_arms_inside_reference_brackets():
    timers, calls = _timers(
        reference=[1.0] * 5, candidate=[3.0, 1.0, 2.0], incumbent=[5.0, 4.0, 6.0]
    )
    results = _worker._benchmark(timers, 3)
    assert calls == [
        "reference",
        *("reference", "candidate", "incumbent"),
        *("incumbent", "candidate", "reference"),
        *("reference", "candidate", "incumbent"),
        "reference",
    ]
    assert results["candidate"] == {
        "samples_ms": [3.0, 1.0, 2.0],
        "median_ms": 2.0,
        "spread_ms": 2.0,
        "relative_spread": 1.0,
        "mad_ms": 1.0,
    }
    assert results["incumbent"]["median_ms"] == 5.0
    assert len(results["reference"]["samples_ms"]) == 5


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_benchmark_rejects_invalid_measurements(value):
    timers, _ = _timers(reference=[1.0] * 3, candidate=[value])
    with pytest.raises(
        RuntimeError, match="candidate did not produce a positive finite"
    ):
        _worker._benchmark(timers, 1)


@requires_cuda
def test_timer_owns_its_graph_work_and_queues_replays_behind_device_work(monkeypatch):
    spins = []
    sleep = torch.cuda._sleep
    monkeypatch.setattr(
        torch.cuda, "_sleep", lambda cycles: spins.append(cycles) or sleep(cycles)
    )

    class Function:
        def __call__(self, x):
            return torch.relu(x + 0.25)

    function = Function()
    alive = weakref.ref(function)
    timer = _worker._capture_timer(function, (torch.randn(3, 129, device="cuda"),), 2)
    del function
    gc.collect()
    assert alive() is not None
    assert timer() > 0
    assert spins == [_worker._SPIN_CYCLES]


def _write_past(tensor):
    torch.as_strided(tensor, (tensor.numel() + 1,), (1,))[-1].fill_(0)


def _write_before(tensor):
    torch.as_strided(tensor, (1,), (1,), tensor.storage_offset() - 1).fill_(0)


@requires_cuda
class TestGuardBands:
    reference = staticmethod(lambda x: torch.relu(x + 0.25))

    def run(self, launch):
        specs = _worker._specs([_spec((3, 129), device="cuda:0")])
        label, inputs = next(_worker._input_cases(specs, 1, False, 3))
        expected = _worker._expected(
            self.reference, inputs, specs, exact_reference=self.reference
        )
        _worker._check_guard_bands(
            launch, inputs, expected, label, rtol=None, atol=None
        )

    def test_a_correct_launch_passes(self):
        self.run(lambda inputs, outputs: outputs[0].copy_(self.reference(inputs[0])))

    @pytest.mark.parametrize(
        "write, message",
        [
            (_write_past, "Out-of-bounds write past output 0"),
            (_write_before, "Out-of-bounds write before output 0"),
        ],
    )
    def test_writes_outside_an_output_fail(self, write, message):
        def launch(inputs, outputs):
            outputs[0].copy_(self.reference(inputs[0]))
            write(outputs[0])

        with pytest.raises(_worker._EvaluationError, match=message):
            self.run(launch)

    def test_input_writes_fail(self):
        def past_input(inputs, outputs):
            outputs[0].copy_(self.reference(inputs[0]))
            _write_past(inputs[0])

        with pytest.raises(_worker._EvaluationError, match="past input 0"):
            self.run(past_input)
        with pytest.raises(_worker._EvaluationError, match="Input 0 was modified"):
            self.run(
                lambda inputs, outputs: outputs[0].copy_(
                    self.reference(inputs[0].relu_())
                )
            )

    def test_an_unwritten_output_element_fails(self):
        def launch(inputs, outputs):
            outputs[0].view(-1)[:-1] = self.reference(inputs[0]).view(-1)[:-1]

        with pytest.raises(
            _worker._EvaluationError,
            match=r"guard-band launch .* got nan at index \(2, 128\)",
        ):
            self.run(launch)

    @pytest.mark.parametrize("masked", [True, False])
    def test_a_triton_kernel_without_tail_masks_fails(self, masked, tmp_path):
        pytest.importorskip("triton")
        mask = ", offsets < N" if masked else ""
        (tmp_path / "shift_kernel.py").write_text(
            "import triton\nimport triton.language as tl\n\n"
            "@triton.jit\n"
            "def shift(arg0, out0, N: tl.constexpr, BLOCK: tl.constexpr):\n"
            "    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
            f"    value = tl.load(arg0 + offsets{mask}) + 0.25\n"
            f"    tl.store(out0 + offsets, tl.where(value < 0.0, 0.0, value){mask})\n"
        )
        sys.path.insert(0, str(tmp_path))
        try:
            from shift_kernel import shift
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("shift_kernel", None)

        def launch(inputs, outputs):
            shift[(4,)](inputs[0], outputs[0], N=387, BLOCK=128)

        if masked:
            self.run(launch)
        else:
            with pytest.raises(_worker._EvaluationError, match="past output 0"):
                self.run(launch)


@pytest.fixture
def evaluation(monkeypatch, tmp_path):
    """A full request whose backend and TensorRT builds are stubs; the rest is real."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    save_reference(torch.nn.ReLU(), torch.randn(7), path=tmp_path / "reference.json")
    calls, registered = [], []
    engines = {"candidate": torch.relu, "wrong": lambda x: torch.relu(x) + 1}

    def validate(proposal, *arity):
        if proposal["kernel_name"] == "invalid":
            raise ValueError("invalid proposal")
        return dict(proposal, grid=[2])

    def build_engine(reference, registration, inputs, num_outputs, options):
        calls.append((registration["kernel_name"], options))
        if registration["kernel_name"] == "broken":
            raise RuntimeError("engine build failed")
        return engines[registration["kernel_name"]]

    def register_plugin(reference, registration):
        registered.append(registration["kernel_name"])
        if registration["kernel_name"] == "unembeddable":
            raise RuntimeError("requires non-zero scratch memory")
        return torch.ops.aten.relu.default

    def build_baseline(reference, inputs, options):
        calls.append(("baseline", options))
        return engines["wrong"]  # a timing arm only: never checked

    backend = SimpleNamespace(
        version="1",
        proposal_schema={"type": "object"},
        validate_candidate=validate,
        make_registration=lambda proposal, *args: proposal,
    )
    monkeypatch.setattr(_backends, "get_backend", lambda name: backend)
    monkeypatch.setattr(_worker, "_build_engine", build_engine)
    monkeypatch.setattr(_worker, "_register_plugin", register_plugin)
    monkeypatch.setattr(_worker, "_build_reference_engine", build_baseline)
    candidate = {
        "source": "source must not be copied into diagnostics",
        "kernel_name": "candidate",
        "grid": [1],
        "block": [32],
        "shared_mem_bytes": 512,
    }
    request = {
        "reference_path": str(tmp_path / "reference.json"),
        "backend": "stub:Backend",
        "backend_version": "1",
        "proposal_schema_sha256": digest(backend.proposal_schema),
        "inputs": [_spec((3, 129), device="cuda:0")],
        "outputs": [_spec((3, 129), device="cuda:0")],
        "objective": "candidate",
        "diagnostics": False,
        "num_tests": 1,
        "benchmark_iters": 2,
        "benchmark_repeats": 2,
        "check_nonfinite": False,
        "rtol": None,
        "atol": None,
        "validation_seed": 11,
        "validation_phase": "search",
        "compile_options": {"disable_tf32": False, "optimization_level": 3},
        "candidate": candidate,
    }
    return SimpleNamespace(
        request=request,
        backend=backend,
        calls=calls,
        registered=registered,
        directory=tmp_path,
    )


def _failure(evaluation):
    with pytest.raises(_worker._EvaluationError) as failure:
        _worker.evaluate(evaluation.request, evaluation.directory)
    return failure.value


def test_evaluation_checks_only_the_candidate_and_times_every_arm(evaluation):
    evaluation.request.update(
        objective="tensorrt",
        incumbent=dict(evaluation.request["candidate"], kernel_name="wrong"),
    )
    result = _worker.evaluate(evaluation.request, evaluation.directory)
    assert result["status"] == "passed"
    assert result["tests"] == 1 + 4 + 3
    assert result["guard_check"] == "unavailable"
    assert result["validation_phase"] == "search"
    assert "incumbent_error" not in result
    for key in (
        "latency_ms",
        "reference_latency_ms",
        "incumbent_latency_ms",
        "baseline_latency_ms",
    ):
        assert result[key] > 0
    assert len(result["benchmark"]["candidate"]["samples_ms"]) == 2
    options = evaluation.request["compile_options"]
    # The baseline is built once, before any candidate work.
    assert evaluation.calls == [
        ("baseline", options),
        ("candidate", options),
        ("wrong", options),
    ]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "kernel_name, stage", [("invalid", "source"), ("broken", "compile")]
)
def test_incumbent_failures_leave_the_candidate_result_standing(
    evaluation, kernel_name, stage
):
    evaluation.request["incumbent"] = dict(
        evaluation.request["candidate"], kernel_name=kernel_name
    )
    result = _worker.evaluate(evaluation.request, evaluation.directory)
    assert result["status"] == "passed"
    assert result["incumbent_error"].startswith(f"{stage}: ")
    assert "incumbent" not in result["benchmark"]
    assert "incumbent_latency_ms" not in result


def test_a_failing_candidate_never_builds_the_incumbent(evaluation):
    evaluation.request["candidate"]["kernel_name"] = "wrong"
    evaluation.request["incumbent"] = dict(
        evaluation.request["candidate"], kernel_name="candidate"
    )
    error = _failure(evaluation)
    assert error.stage == "correctness"
    assert "Output 0 is wrong for case 'random 1'" in str(error)
    assert [name for name, _ in evaluation.calls] == ["wrong"]


def test_evaluation_reports_source_errors_before_building_an_engine(evaluation):
    evaluation.request["candidate"]["kernel_name"] = "invalid"
    assert _failure(evaluation).stage == "source"
    assert evaluation.calls == []


@pytest.mark.parametrize(
    "identity, message",
    [
        ({"backend_version": "obsolete"}, "Backend version changed"),
        ({"proposal_schema_sha256": "0" * 64}, "Backend proposal schema changed"),
    ],
)
def test_backend_drift_is_a_setup_failure(evaluation, identity, message):
    evaluation.request.update(identity)
    error = _failure(evaluation)
    assert error.stage == "setup"
    assert message in str(error)


class _Half(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x).half()


@pytest.mark.parametrize(
    "breakage",
    [
        lambda request, directory: request.update(
            reference_path=str(directory / "missing.json")
        ),
        lambda request, directory: save_reference(
            _Half(), torch.randn(7), path=request["reference_path"]
        ),
        lambda request, directory: request.pop("num_tests"),
    ],
    ids=["missing reference", "reference breaks the contract", "malformed request"],
)
def test_controller_failures_are_setup_failures(evaluation, breakage):
    breakage(evaluation.request, evaluation.directory)
    assert _failure(evaluation).stage == "setup"
    assert evaluation.calls == []


def test_reference_timing_fails_setup_before_candidate_work(evaluation, monkeypatch):
    def capture(function, inputs, iterations):
        raise RuntimeError("operation not permitted when stream is capturing")

    monkeypatch.setattr(_worker, "_capture_timer", capture)
    assert _failure(evaluation).stage == "setup"
    assert evaluation.calls == []


def test_tensorrt_baseline_build_failure_is_a_baseline_failure(evaluation, monkeypatch):
    def build(*args):
        raise RuntimeError("TensorRT baseline contains eager computation")

    evaluation.request["objective"] = "tensorrt"
    monkeypatch.setattr(_worker, "_build_reference_engine", build)
    assert _failure(evaluation).stage == "baseline"
    assert evaluation.calls == []


def test_tensorrt_baseline_timing_failure_is_a_baseline_failure(
    evaluation, monkeypatch
):
    capture, captured = _worker._capture_timer, []

    def capture_reference_only(*args):
        captured.append(args)
        if len(captured) > 1:
            raise RuntimeError("operation not permitted when stream is capturing")
        return capture(*args)

    evaluation.request["objective"] = "tensorrt"
    monkeypatch.setattr(_worker, "_capture_timer", capture_reference_only)
    assert _failure(evaluation).stage == "baseline"
    assert [name for name, _ in evaluation.calls] == ["baseline"]


def test_backends_with_a_launch_hook_get_a_guard_band_check(evaluation):
    def launch(candidate, inputs, outputs, directory):
        outputs[0].copy_(inputs[0].relu())

    evaluation.backend.launch = launch
    result = _worker.evaluate(evaluation.request, evaluation.directory)
    assert result["guard_check"] == "passed"

    def unmasked(candidate, inputs, outputs, directory):
        launch(candidate, inputs, outputs, directory)
        _write_past(outputs[0])

    evaluation.backend.launch = unmasked
    error = _failure(evaluation)
    assert error.stage == "correctness"
    assert "Out-of-bounds write past output 0" in str(error)


def _screened(evaluation, launches=None):
    """Give the stub backend a direct launch and screen the search with it."""

    def launch(candidate, inputs, outputs, directory):
        if launches is not None:
            launches.append(candidate["kernel_name"])
        outputs[0].copy_(inputs[0].relu())

    evaluation.backend.launch = launch
    evaluation.request["screening"] = "launch"


def test_search_screening_launches_kernels_without_building_engines(evaluation):
    launches = []
    _screened(evaluation, launches)
    evaluation.request.update(
        objective="tensorrt",
        diagnostics=True,
        incumbent=dict(evaluation.request["candidate"], kernel_name="incumbent"),
    )
    result = _worker.evaluate(evaluation.request, evaluation.directory)
    assert result["status"] == "passed" and result["execution"] == "launch"
    assert result["tests"] == 1 + 4 + 3 and result["guard_check"] == "passed"
    # Both were registered as plugins, but neither they nor the TensorRT
    # baseline were built as engines.
    assert evaluation.registered == ["candidate", "incumbent"]
    assert evaluation.calls == []
    assert "baseline_latency_ms" not in result and "baseline" not in result["benchmark"]
    assert result["latency_ms"] > 0 and result["incumbent_latency_ms"] > 0
    assert {"candidate", "incumbent"} <= set(launches)
    for name in ("candidate", "incumbent"):
        details = result["diagnostics"][name]
        assert details["compilation_scope"] == "backend_and_plugin_registration"
        assert details["compilation_status"] == "completed"
    json.dumps(result, allow_nan=False)


def test_screening_rejects_what_only_the_plugin_registration_refuses(evaluation):
    launches = []
    _screened(evaluation, launches)
    evaluation.request["candidate"]["kernel_name"] = "unembeddable"
    evaluation.request["diagnostics"] = True
    error = _failure(evaluation)
    assert error.stage == "compile"
    assert "requires non-zero scratch memory" in str(error)
    assert launches == [] and evaluation.calls == []
    assert error.diagnostics["candidate"]["compilation_status"] == "failed"


@pytest.mark.parametrize("breaks", [False, True])
def test_a_screen_builds_the_tensorrt_baseline_when_asked(
    evaluation, monkeypatch, breaks
):
    def build(*args):
        raise RuntimeError("TensorRT baseline contains eager computation")

    _screened(evaluation)
    evaluation.request.update(objective="tensorrt", check_baseline=True)
    if breaks:
        monkeypatch.setattr(_worker, "_build_reference_engine", build)
        assert _failure(evaluation).stage == "baseline"
        assert evaluation.registered == []
        return
    result = _worker.evaluate(evaluation.request, evaluation.directory)
    assert result["execution"] == "launch" and result["baseline_latency_ms"] > 0
    assert [name for name, _ in evaluation.calls] == ["baseline"]


@pytest.mark.parametrize(
    "phase, screening", [("holdout", "launch"), ("search", "engine"), ("search", None)]
)
def test_holdout_and_engine_screening_build_trial_engines(evaluation, phase, screening):
    _screened(evaluation)
    evaluation.request.update(
        validation_phase=phase, screening=screening, objective="tensorrt"
    )
    result = _worker.evaluate(evaluation.request, evaluation.directory)
    assert result["execution"] == "engine"
    assert [name for name, _ in evaluation.calls] == ["baseline", "candidate"]
    assert result["baseline_latency_ms"] > 0


def test_screening_needs_a_backend_launch(evaluation):
    evaluation.request["screening"] = "launch"
    result = _worker.evaluate(evaluation.request, evaluation.directory)
    assert result["execution"] == "engine"
    assert [name for name, _ in evaluation.calls] == ["candidate"]


def test_screened_kernels_are_checked_on_every_case(evaluation):
    def launch(candidate, inputs, outputs, directory):
        # Wrong only where an input is exactly -1, as the constant case is.
        outputs[0].copy_(torch.where(inputs[0] == -1, 1.0, inputs[0].relu()))

    _screened(evaluation)
    evaluation.backend.launch = launch
    error = _failure(evaluation)
    assert error.stage == "correctness"
    assert "Output 0 is wrong for case 'constant -1.0'" in str(error)
    assert evaluation.calls == []


@pytest.mark.parametrize("screening", ["launch", "engine"])
def test_a_launch_that_raises_is_a_compile_failure(evaluation, screening):
    def launch(candidate, inputs, outputs, directory):
        raise RuntimeError("JIT compilation failed: unknown intrinsic")

    _screened(evaluation)
    evaluation.backend.launch = launch
    evaluation.request.update(screening=screening, diagnostics=True)
    error = _failure(evaluation)
    assert error.stage == "compile"
    assert "unknown intrinsic" in str(error)
    # The registration completed, but compiling what the launch runs did not.
    assert error.diagnostics["candidate"]["compilation_status"] == "failed"


def test_a_kernel_wrong_only_on_real_magnitudes_fails_on_their_sample(
    evaluation, monkeypatch
):
    # Correct for every synthetic magnitude, wrong past 1e3.
    monkeypatch.setattr(
        _worker,
        "_build_engine",
        lambda *args: lambda x: torch.relu(x).clamp(max=1e3),
    )
    path = evaluation.directory / "samples.pt"
    evaluation.request["samples_path"] = str(path)
    torch.save([[torch.randn(3, 129)]], path)
    result = _worker.evaluate(evaluation.request, evaluation.directory)
    assert result["status"] == "passed" and result["tests"] == 1 + 4 + 3 + 1
    large = torch.randn(3, 129)
    large[1, 7] = 1e4
    torch.save([[torch.randn(3, 129)], [large]], path)
    error = _failure(evaluation)
    assert error.stage == "correctness"
    assert "Output 0 is wrong for case 'sample 2'" in str(error)
    assert "at index (1, 7)" in str(error)


class _Payload:
    """Unpickling this runs a command; weights-only loading refuses it."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


@pytest.mark.parametrize(
    "content, message",
    [
        (lambda marker: [[_Payload(marker)]], "Weights only load failed"),
        (lambda marker: [[torch.zeros(7)]], r"sample 1 input 0 is float32 \[7\];"),
        (lambda marker: {"x": [torch.zeros(3, 129)]}, "does not hold a list"),
    ],
    ids=["pickled code", "wrong shape", "wrong structure"],
)
def test_saved_samples_load_only_as_checked_tensor_data(evaluation, content, message):
    path = evaluation.directory / "samples.pt"
    marker = evaluation.directory / "executed"
    torch.save(content(marker), path)
    evaluation.request["samples_path"] = str(path)
    error = _failure(evaluation)
    assert error.stage == "setup"
    assert re.search(message, str(error))
    assert not marker.exists()
    assert evaluation.calls == []


def test_optional_diagnostics_measure_existing_builds_without_inventing_resources(
    evaluation, monkeypatch
):
    evaluation.request.update(
        diagnostics=True,
        objective="tensorrt",
        incumbent=evaluation.request["candidate"],
    )
    # Every timed block calls perf_counter twice, back to back.
    ticks = itertools.count(0.0, 0.25)
    monkeypatch.setattr(_worker.time, "perf_counter", lambda: next(ticks))
    result = _worker.evaluate(evaluation.request, evaluation.directory)
    details = result["diagnostics"]
    for name in ("candidate", "incumbent"):
        assert details[name]["compilation_ms"] == 250.0
        assert details[name]["compilation_status"] == "completed"
        assert details[name]["launch"]["grid"] == [2]
        assert details[name]["launch"]["shared_mem_bytes"] == 512
        assert details[name]["launch_basis"] == "validated_proposal"
        resources = details[name]["compiler_resources"]
        assert resources["registers_per_thread"] is None
        assert resources["shared_memory_bytes"] is None
        assert "does not expose" in resources["unavailable_reason"]
        assert "source" not in details[name]["launch"]
    assert details["baseline"]["compilation_ms"] == 250.0
    assert details["baseline"]["compilation_status"] == "completed"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("stage", ["compile", "engine", "correctness", "baseline"])
def test_failed_evaluation_keeps_measured_compilation_diagnostics(
    evaluation, monkeypatch, stage
):
    evaluation.request.update(diagnostics=True, objective="tensorrt")

    def fail(*args):
        raise _worker._EvaluationError(stage, "build failed")

    if stage == "compile":
        evaluation.backend.make_registration = fail
    elif stage == "engine":
        monkeypatch.setattr(_worker, "_build_engine", fail)
    elif stage == "baseline":
        monkeypatch.setattr(_worker, "_build_reference_engine", fail)
    else:
        evaluation.request["candidate"]["kernel_name"] = "wrong"
    report = _worker._error_report(_failure(evaluation), evaluation.directory, "search")
    assert report["stage"] == stage
    details = report["diagnostics"]
    assert details["baseline"]["compilation_status"] == (
        "failed" if stage == "baseline" else "completed"
    )
    assert (
        details["candidate"]["compilation_status"]
        == {
            "baseline": "not_started",
            "compile": "failed",
            "engine": "failed",
            "correctness": "completed",
        }[stage]
    )
    json.dumps(report, allow_nan=False)


def test_trial_builds_apply_user_options_under_forced_single_engine_options(
    monkeypatch,
):
    reference = torch.fx.symbolic_trace(torch.nn.ReLU())
    seen = []

    def compile(module, **kwargs):
        seen.append(kwargs)
        return module

    monkeypatch.setattr(torch_tensorrt, "compile", compile)
    monkeypatch.setattr(
        _plugins, "register_region_plugin", lambda *args, **kwargs: torch.relu
    )
    options = {"disable_tf32": False, "optimization_level": 5, "min_block_size": 9}
    inputs = (torch.randn(7),)
    with pytest.raises(RuntimeError, match="did not produce a TensorRT engine"):
        _worker._build_reference_engine(reference, inputs, options)
    with pytest.raises(
        _worker._EvaluationError, match="not placed inside a TensorRT engine"
    ):
        _worker._build_engine(reference, object(), inputs, 1, options)
    for kwargs in seen:
        assert kwargs["disable_tf32"] is False
        assert kwargs["optimization_level"] == 5
        assert kwargs["min_block_size"] == 1
        assert kwargs["require_full_compilation"] is True
        assert kwargs["offload_module_to_cpu"] is False
        assert kwargs["ir"] == "dynamo"
        assert kwargs["strict"] is True
    assert len(seen) == 2


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


def test_worker_processes_cannot_dump_core():
    probe = (
        "import resource\n"
        "from torch_tensorrt.dynamo.region_compilation._generation import _worker\n"
        "_worker._disable_core_dumps()\n"
        "print(resource.getrlimit(resource.RLIMIT_CORE))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip().splitlines()[-1] == "(0, 0)"


@pytest.mark.parametrize(
    "home, text, scrubbed",
    [
        (
            "/",
            "No such file: '/usr/lib/kernel.py'",
            "No such file: '/usr/lib/kernel.py'",
        ),
        (
            "/home/user",
            "/home/user/.cache and /home/username/x",
            "<home>/.cache and /home/username/x",
        ),
        (
            "/home/user",
            "'/home/user' in /mnt/home/user/x",
            "'<home>' in /mnt/home/user/x",
        ),
    ],
)
def test_scrubbing_replaces_whole_paths_only(
    monkeypatch, tmp_path, home, text, scrubbed
):
    monkeypatch.setenv("HOME", home)
    assert _util.scrub(text, tmp_path) == scrubbed


def test_main_reports_unexpected_failures_as_setup_and_replaces_result_atomically(
    tmp_path, monkeypatch
):
    request = tmp_path / "request.json"
    result = tmp_path / "result.json"
    request.write_text(json.dumps({"validation_phase": "holdout"}))
    result.write_text('{"status": "passed"}')

    def fail(*args):
        raise RuntimeError("reference execution failed")

    monkeypatch.setattr(_worker, "evaluate", fail)
    monkeypatch.setattr(
        sys, "argv", ["worker", "--request", str(request), "--result", str(result)]
    )
    exits = []
    _worker.main(exit=exits.append)
    assert exits == [0]
    report = json.loads(result.read_text())
    assert report["status"] == "error"
    assert report["stage"] == "setup"
    assert report["validation_phase"] == "holdout"
    assert "reference execution failed" in report["error"]
    assert "latency_ms" not in report
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "request.json",
        "result.json",
    ]
    request.write_text("{not json")
    _worker.main(exit=exits.append)
    assert json.loads(result.read_text())["stage"] == "setup"


_MULTI_OUTPUT_KERNEL = """
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


def _run_worker(
    tmp_path, source, objective, *, incumbent=False, phase="holdout", screening=None
):
    """Evaluate a Triton kernel for (relu(x + 0.25), x * x) in a worker process."""
    pytest.importorskip("triton")

    class Reference(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x + 0.25), x * x

    reference_path = tmp_path / "reference.json"
    save_reference(Reference(), torch.randn(3, 129), path=reference_path)
    backend = _backends.get_backend("triton")
    spec = {"shape": [3, 129], "dtype": "float32", "device": "cuda:0"}
    request = {
        "reference_path": str(reference_path),
        "backend": "triton",
        "backend_version": backend.version,
        "proposal_schema_sha256": digest(backend.proposal_schema),
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
        "validation_phase": phase,
        "screening": screening,
        "objective": objective,
        "benchmark_repeats": 3,
        "diagnostics": True,
        "compile_options": DEFAULT_COMPILE_OPTIONS,
    }
    if incumbent:
        request["incumbent"] = dict(request["candidate"])
    return _run_request(tmp_path, request)


def _run_request(tmp_path, request):
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
    return json.loads(result_path.read_text())


requires_qdp = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    reason="CUDA and TensorRT QDP support are required",
)


@requires_qdp
@pytest.mark.parametrize(
    "correct, objective", [(True, "eager"), (True, "tensorrt"), (False, "candidate")]
)
def test_worker_checks_a_multi_output_kernel_in_a_real_engine(
    tmp_path, correct, objective
):
    source = _MULTI_OUTPUT_KERNEL
    if not correct:
        source = source.replace("values + 0.25", "values + 1.0")
    report = _run_worker(tmp_path, source, objective, incumbent=correct)
    details = report["diagnostics"]["candidate"]
    assert details["compilation_ms"] > 0
    assert details["compilation_status"] == "completed"
    assert details["launch"]["grid"] == [4]
    assert details["compiler_resources"]["registers_per_thread"] is None
    if correct:
        assert report["status"] == "passed", report
        assert report["measurement"] == "cuda_graph"
        assert report["tests"] == 2 + 4 + 3 + 3
        assert report["guard_check"] == "passed"
        assert report["latency_ms"] > 0
        assert report["reference_latency_ms"] > 0
        assert report["incumbent_latency_ms"] > 0
        assert report["baseline_latency_ms"] > 0
        assert len(report["benchmark"]["candidate"]["samples_ms"]) == 3
        assert report["validation_phase"] == "holdout"
    else:
        assert report["status"] == "error", report
        assert report["stage"] == "correctness"
        assert "Output 0 is wrong for" in report["error"]
        assert "case 'random 1'" in report["error"]


@requires_qdp
def test_worker_rejects_a_real_kernel_that_writes_past_its_outputs(tmp_path):
    # Without tail masks, 4 x 128 lanes cover 387 elements and write 125 past each.
    source = _MULTI_OUTPUT_KERNEL.replace(", offsets < N, other=0.0", "")
    source = source.replace(", offsets < N)", ")")
    report = _run_worker(tmp_path, source, "candidate")
    assert report["status"] == "error", report
    assert report["stage"] == "correctness"
    assert "Out-of-bounds write past output 0" in report["error"]


@requires_qdp
@pytest.mark.parametrize(
    "change, stage",
    [
        (None, None),
        (("values + 0.25", "values + 1.0"), "correctness"),
        (("values * values", "tl.no_such_function(values)"), "compile"),
    ],
    ids=["correct", "wrong", "uncompilable"],
)
def test_worker_screens_a_real_kernel_by_launching_it(tmp_path, change, stage):
    source = (
        _MULTI_OUTPUT_KERNEL
        if change is None
        else _MULTI_OUTPUT_KERNEL.replace(*change)
    )
    report = _run_worker(
        tmp_path,
        source,
        "tensorrt",
        incumbent=stage is None,
        phase="search",
        screening="launch",
    )
    details = report["diagnostics"]["candidate"]
    assert details["compilation_scope"] == "backend_and_plugin_registration"
    # A kernel that does not compile never reports a completed compilation.
    assert details["compilation_status"] == (
        "failed" if stage == "compile" else "completed"
    )
    if stage is None:
        assert report["status"] == "passed", report
        assert report["execution"] == "launch"
        assert report["tests"] == 2 + 4 + 3 + 3
        assert report["guard_check"] == "passed"
        assert report["latency_ms"] > 0 and report["incumbent_latency_ms"] > 0
        # Screening builds no TensorRT baseline: the objective waits for an engine.
        assert "baseline_latency_ms" not in report
    else:
        assert report["status"] == "error", report
        assert report["stage"] == stage
        if stage == "correctness":
            assert "Output 0 is wrong for the guard-band launch" in report["error"]


_MATMUL_KERNEL = """
import triton
import triton.language as tl


@triton.jit
def generated(
    arg0, arg1, out0, M: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, M, BK):
        a = tl.load(arg0 + rm[:, None] * M + (k + rk)[None, :])
        b = tl.load(arg1 + (k + rk)[:, None] * M + rn[None, :])
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(out0 + rm[:, None] * M + rn[None, :], acc)
"""


def _matmul_request(tmp_path, tile, block_k, num_stages, screening):
    """A worker request for a 256 x 256 float32 Triton matmul."""
    pytest.importorskip("triton")
    reference_path = tmp_path / "reference.json"
    save_reference(
        torch.mm, torch.randn(256, 256), torch.randn(256, 256), path=reference_path
    )
    backend = _backends.get_backend("triton")
    spec = {"shape": [256, 256], "dtype": "float32", "device": "cuda:0"}
    constants = {"M": 256, "BM": tile, "BN": tile, "BK": block_k}
    return {
        "reference_path": str(reference_path),
        "backend": "triton",
        "backend_version": backend.version,
        "proposal_schema_sha256": digest(backend.proposal_schema),
        "candidate": {
            "source": _MATMUL_KERNEL,
            "kernel_name": "generated",
            "constants": [{"name": k, "value": v} for k, v in constants.items()],
            "grid": [256 // tile, 256 // tile],
            "num_warps": 4,
            "num_stages": num_stages,
        },
        "inputs": [spec, spec],
        "outputs": [spec],
        "num_tests": 1,
        "check_nonfinite": False,
        "benchmark_iters": 3,
        "rtol": None,
        "atol": None,
        "validation_seed": 5,
        "validation_phase": "search",
        "screening": screening,
        "objective": "candidate",
        "benchmark_repeats": 3,
        "diagnostics": False,
        "compile_options": DEFAULT_COMPILE_OPTIONS,
    }


def _plugin_shared_memory(request, directory):
    """The shared memory of the kernel that the request's plugin would embed."""
    from torch_tensorrt.kernels._triton import compile_triton_to_ptx

    candidate = request["candidate"]
    kernel = _backends.get_backend("triton")._kernel(candidate, 2, 1, directory)
    return compile_triton_to_ptx(
        kernel,
        {"arg0": "*fp32", "arg1": "*fp32", "out0": "*fp32"},
        {item["name"]: item["value"] for item in candidate["constants"]},
        num_warps=candidate["num_warps"],
        num_stages=candidate["num_stages"],
    ).shared_mem


@requires_qdp
@pytest.mark.parametrize(
    "tile, block_k, num_stages, screening",
    [
        (128, 64, 1, "launch"),
        # Null selects Triton's default num_stages, which pipelines the operand
        # loads through additional shared memory buffers.
        (64, 64, None, "launch"),
        (64, 64, None, "engine"),
    ],
    ids=["wide-tile", "default-stages", "default-stages-engine"],
)
def test_worker_rejects_a_triton_kernel_a_plugin_launch_cannot_fit(
    tmp_path, tile, block_k, num_stages, screening
):
    # Triton's own launcher opts into this much shared memory; TensorRT does not.
    request = _matmul_request(tmp_path, tile, block_k, num_stages, screening)
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block
    if _plugin_shared_memory(request, tmp_path / "kernel") <= limit:
        pytest.skip("This GPU stages the tiles within the default shared memory")
    report = _run_request(tmp_path, request)
    assert report["status"] == "error", report
    assert report["stage"] == "compile"
    assert f"but a TensorRT plugin launch allows at most {limit}" in report["error"]


@requires_qdp
def test_worker_screens_a_triton_kernel_that_fits_a_plugin_launch(tmp_path):
    request = _matmul_request(tmp_path, 64, 64, 1, "launch")
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block
    assert _plugin_shared_memory(request, tmp_path / "kernel") <= limit
    report = _run_request(tmp_path, request)
    assert report["status"] == "passed", report
    assert report["execution"] == "launch"
