# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Capture, feedback, selection, and cache policy without agent or GPU access."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from torch_tensorrt.dynamo.region_compilation import (
    is_region_hop,
    normalize_region_scopes,
)
from torch_tensorrt.dynamo.region_compilation._generation import _factory
from torch_tensorrt.region_annotations import (
    KernelGenerator,
    RegionError,
    SearchPolicy,
    fuse_in_plugin,
)
from torch_tensorrt.region_annotations._session import RegionCompilationSession


class GeneratedRegion(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x):
        with fuse_in_plugin(generator=self.generator, name="generated_activation"):
            return torch.relu(x + 1)


def test_eager_and_unowned_export_do_not_invoke_an_agent(monkeypatch):
    generate = Mock(side_effect=AssertionError("Unexpected agent invocation"))
    monkeypatch.setattr(_factory, "generate_registration", generate)
    model = GeneratedRegion(KernelGenerator())
    inputs = (torch.randn(7),)
    expected = torch.relu(inputs[0] + 1)
    torch.testing.assert_close(model(*inputs), expected)
    exported = torch.export.export(model, inputs, strict=True)
    torch.testing.assert_close(exported.module()(*inputs), expected)
    generate.assert_not_called()


def test_strict_capture_stores_configuration_without_generating(monkeypatch):
    author = Mock(side_effect=AssertionError("Capture invoked the agent"))
    monkeypatch.setattr(_factory._providers, "propose", author)
    config = KernelGenerator(provider="codex", backend="cutile", max_attempts=2)
    with RegionCompilationSession(strict=True) as session:
        exported = torch.export.export(
            GeneratedRegion(config), (torch.ones(7),), strict=True
        )
        exported = normalize_region_scopes(exported)
        regions = [node for node in exported.graph.nodes if is_region_hop(node)]
        assert len(regions) == 1
        hints = regions[0].kwargs["hints"]
        assert hints["route"] == "plugin"
        assert session.plugins[hints["plugin_key"]] is config
    author.assert_not_called()


@pytest.mark.parametrize(
    "options,error,match",
    [
        ({}, ValueError, "exactly one"),
        (
            {"kernel": lambda: None, "generator": KernelGenerator()},
            ValueError,
            "exactly one",
        ),
        ({"generator": "codex"}, TypeError, "KernelGenerator"),
        (
            {"generator": KernelGenerator(), "grid": lambda *args: (1,)},
            ValueError,
            "launch configuration",
        ),
        (
            {"generator": KernelGenerator(), "constexprs": {}},
            ValueError,
            "launch configuration",
        ),
    ],
)
def test_manual_and_generated_kernel_options_are_exclusive(options, error, match):
    with pytest.raises(error, match=match), fuse_in_plugin(**options):
        pass


@pytest.mark.parametrize(
    "options",
    [
        {"max_attempts": True},
        {"timeout_s": float("nan")},
        {"candidate_timeout_s": 0},
        {"rtol": 0.1},
        {"rtol": -1, "atol": 0},
    ],
)
def test_invalid_generation_budgets_and_tolerances_are_rejected(options):
    with pytest.raises(ValueError):
        KernelGenerator(**options)


def _candidate(name):
    return {
        "source": f"""
import triton
import triton.language as tl

@triton.jit
def {name}(arg0, out0, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(arg0 + offsets, offsets < N, other=0.0)
    value = value + value
    tl.store(out0 + offsets, tl.where(value < 0, 0.0, value), offsets < N)
""",
        "kernel_name": name,
        "constants": [{"name": "N", "value": 7}, {"name": "BLOCK", "value": 32}],
        "grid": [1],
        "num_warps": 4,
        "num_stages": None,
    }


def _passed(latency):
    return {
        "status": "passed",
        "measurement": "cuda_graph",
        "latency_ms": latency,
        "reference_latency_ms": 1.0,
        "tests": 6,
    }


@pytest.fixture
def factory(tmp_path, monkeypatch):
    class Reference(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.relu.default(torch.ops.aten.add.Tensor(x, x))

    boundary = SimpleNamespace(
        shape=(7,), dtype=torch.float32, device=torch.device("cuda:0")
    )
    monkeypatch.setattr(_factory, "_boundaries", lambda *args: ([boundary], [boundary]))
    monkeypatch.setattr(
        _factory, "_target", lambda *args: {"name": "test-gpu", "capability": [9, 0]}
    )
    author = Mock()
    evaluate = Mock()
    finalize = Mock(return_value=_passed(0.2))

    def dispatch_evaluation(request, directory, timeout):
        function = finalize if directory.name == "holdout" else evaluate
        return function(request, directory, timeout)

    register = Mock(side_effect=lambda candidate, *args: candidate)
    monkeypatch.setattr(_factory._providers, "propose", author)
    monkeypatch.setattr(_factory, "_evaluate", dispatch_evaluation)
    monkeypatch.setattr(_factory, "make_registration", register)
    return SimpleNamespace(
        graph=torch.fx.symbolic_trace(Reference()),
        config=KernelGenerator(cache_dir=tmp_path / "cache", max_attempts=1),
        author=author,
        evaluate=evaluate,
        finalize=finalize,
        register=register,
        root=tmp_path / "cache",
    )


def _generate(factory, **changes):
    return _factory.generate_registration(
        factory.graph, replace(factory.config, **changes), region_id="activation"
    )


def test_factory_retries_with_feedback_and_retains_the_fastest_valid_kernel(factory):
    candidates = [_candidate(name) for name in ("wrong", "fast", "slower")]
    factory.author.side_effect = candidates
    factory.evaluate.side_effect = [
        {"status": "error", "error": "Output 0 disagrees with PyTorch"},
        _passed(0.1),
        _passed(0.5),
    ]
    selected = _generate(factory, max_attempts=3)
    assert selected == candidates[1]
    assert "Output 0 disagrees" in factory.author.call_args_list[1].kwargs["prompt"]
    assert '"latency_ms": 0.1' in factory.author.call_args_list[2].kwargs["prompt"]
    assert factory.evaluate.call_count == 3
    request = factory.evaluate.call_args_list[0].args[0]
    assert request["inputs"][0]["shape"] == [7]
    assert request["outputs"][0]["dtype"] == "float32"
    saved_reference = torch.load(request["reference_path"], weights_only=False)
    inputs = torch.randn(7)
    torch.testing.assert_close(saved_reference(inputs), factory.graph(inputs))
    assert all(not node.meta for node in saved_reference.graph.nodes)
    accepted = json.loads(next(factory.root.glob("*/accepted.json")).read_text())
    assert accepted["candidate"] == candidates[1]
    assert accepted["report"]["latency_ms"] == 0.1


def test_repeated_passing_candidate_reuses_evaluation_and_preserves_evidence(factory):
    candidate = _candidate("repeated")
    factory.author.return_value = candidate
    factory.evaluate.return_value = _passed(0.2)

    assert _generate(factory, max_attempts=3) == candidate

    assert factory.author.call_count == 3
    factory.evaluate.assert_called_once()
    history_path = next(factory.root.glob("*/run-*/history.json"))
    history = json.loads(history_path.read_text())
    evaluation_path = history_path.parent / "attempt-1" / "feedback.json"
    assert len(history) == 3
    assert "reused_from" not in history[0]["result"]
    for index in (1, 2):
        assert history[index]["result"]["reused_from"] == str(evaluation_path)
        assert history[index]["result"]["latency_ms"] == 0.2
        attempt = history_path.parent / f"attempt-{index + 1}"
        assert (
            json.loads((attempt / "feedback.json").read_text())
            == history[index]["result"]
        )
        assert json.loads((attempt / "candidate.json").read_text()) == candidate
    accepted = json.loads(next(factory.root.glob("*/accepted.json")).read_text())
    assert accepted["evaluation_path"] == str(evaluation_path)
    assert json.loads(evaluation_path.read_text()) == accepted["report"]
    assert "unchanged passing candidates reuse their prior evaluation" in (
        factory.author.call_args.kwargs["prompt"]
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"constants": [{"name": "N", "value": 7}, {"name": "BLOCK", "value": 64}]},
        {"grid": [2]},
        {"num_warps": 8},
        {"num_stages": 2},
    ],
)
def test_changed_launch_configuration_gets_a_fresh_evaluation(factory, changes):
    first = _candidate("tuned")
    tuned = {**first, **changes}
    factory.author.side_effect = [first, tuned]
    factory.evaluate.side_effect = [_passed(0.2), _passed(0.1)]

    assert _generate(factory, max_attempts=2) == tuned

    assert factory.evaluate.call_count == 2
    accepted = json.loads(next(factory.root.glob("*/accepted.json")).read_text())
    assert accepted["evaluation_path"].endswith("/attempt-2/feedback.json")


def test_repeated_failed_candidate_is_evaluated_again(factory):
    candidate = _candidate("retry")
    factory.author.return_value = candidate
    factory.evaluate.side_effect = [
        {"status": "error", "error": "GPU evaluation timed out"},
        _passed(0.2),
    ]

    assert _generate(factory, max_attempts=2) == candidate

    assert factory.evaluate.call_count == 2
    accepted = json.loads(next(factory.root.glob("*/accepted.json")).read_text())
    assert accepted["evaluation_path"].endswith("/attempt-2/feedback.json")


def test_cache_reuses_a_kernel_without_an_agent_and_revalidates_it(factory):
    candidate = _candidate("cached")
    factory.author.return_value = candidate
    factory.evaluate.side_effect = [_passed(0.2), _passed(0.3)]
    assert _generate(factory) == candidate
    assert _generate(factory, provider="codex") == candidate
    factory.author.assert_called_once()
    assert factory.evaluate.call_count == 2
    cache_check = next(factory.root.glob("*/run-*/cache_check.json"))
    assert json.loads(cache_check.read_text())["status"] == "passed"


def test_cache_revalidation_failure_triggers_new_generation(factory):
    old, new = _candidate("old"), _candidate("new")
    factory.author.side_effect = [old, new]
    factory.evaluate.side_effect = [
        _passed(0.2),
        {"status": "error", "error": "Cached kernel failed validation"},
        _passed(0.1),
    ]
    assert _generate(factory) == old
    assert _generate(factory) == new
    assert factory.author.call_count == 2
    assert factory.evaluate.call_count == 3
    accepted = json.loads(next(factory.root.glob("*/accepted.json")).read_text())
    assert accepted["candidate"] == new


def test_invalid_nan_proposal_cannot_discard_an_earlier_valid_winner(factory):
    best, invalid = _candidate("best"), _candidate("invalid")
    invalid["constants"][0]["value"] = float("nan")
    factory.author.side_effect = [best, invalid]
    factory.evaluate.return_value = _passed(0.2)
    assert _generate(factory, max_attempts=2) == best
    factory.evaluate.assert_called_once()
    history = json.loads(next(factory.root.glob("*/run-*/history.json")).read_text())
    assert history[1]["candidate"] is None
    assert history[1]["result"]["status"] == "error"
    assert "finite" in history[1]["result"]["error"]


def test_exhausted_failures_raise_and_preserve_evidence(factory):
    factory.author.side_effect = [_candidate("first"), _candidate("second")]
    factory.evaluate.side_effect = [
        {"status": "error", "error": "Compilation failed"},
        {"status": "error", "error": "Numerical mismatch"},
    ]
    with pytest.raises(RegionError, match="Numerical mismatch.*Artifacts:") as failure:
        _generate(factory, max_attempts=2)
    history_path = next(factory.root.glob("*/run-*/history.json"))
    assert str(history_path.parent) in str(failure.value)
    assert len(json.loads(history_path.read_text())) == 2
    assert not list(factory.root.glob("*/accepted.json"))
    factory.register.assert_not_called()


def test_provider_failure_stops_retries_and_preserves_the_best_kernel(factory):
    best = _candidate("best")
    factory.author.side_effect = [
        best,
        _factory._providers.ProviderError("Account quota exhausted"),
    ]
    factory.evaluate.return_value = _passed(0.2)
    assert _generate(factory, max_attempts=4) == best
    assert factory.author.call_count == 2
    factory.evaluate.assert_called_once()
    history = json.loads(next(factory.root.glob("*/run-*/history.json")).read_text())
    assert history[-1]["result"]["error"] == "Account quota exhausted"


def test_corrupt_cache_digest_requires_new_authoring(factory):
    old, new = _candidate("old"), _candidate("new")
    factory.author.side_effect = [old, new]
    factory.evaluate.return_value = _passed(0.2)
    assert _generate(factory) == old
    manifest = next(factory.root.glob("*/accepted.json"))
    stored = json.loads(manifest.read_text())
    stored["candidate_sha256"] = "corrupted"
    manifest.write_text(json.dumps(stored))
    assert _generate(factory) == new
    assert factory.author.call_count == 2
    assert factory.evaluate.call_count == 2
    assert json.loads(manifest.read_text())["candidate"] == new


def test_reference_semantic_changes_invalidate_the_kernel_cache(factory):
    relu, sigmoid = _candidate("relu"), _candidate("sigmoid")
    factory.author.side_effect = [relu, sigmoid]
    factory.evaluate.return_value = _passed(0.2)
    assert _generate(factory) == relu
    activation = next(
        node
        for node in factory.graph.graph.nodes
        if node.target == torch.ops.aten.relu.default
    )
    activation.target = torch.ops.aten.sigmoid.default
    factory.graph.recompile()
    assert _generate(factory) == sigmoid
    assert factory.author.call_count == 2
    assert len(list(factory.root.glob("*/accepted.json"))) == 2
    assert "aten.sigmoid.default" in factory.author.call_args.kwargs["prompt"]


def test_tensor_attributes_are_rejected_before_agent_invocation(factory):
    class WithConstant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("bias", torch.ones(7))

        def forward(self, x):
            return x + self.bias

    factory.graph = torch.fx.symbolic_trace(WithConstant())
    with pytest.raises(RegionError, match="tensor constants.*boundary inputs"):
        _generate(factory)
    factory.author.assert_not_called()
    factory.evaluate.assert_not_called()


def test_saved_reference_reuses_cache_and_reports_its_exact_validation_run(factory):
    candidate = _candidate("round_trip")
    factory.author.return_value = candidate
    factory.evaluate.return_value = _passed(0.2)
    registration, first = _factory._generate(
        factory.graph, factory.config, region_id="original"
    )
    restored = torch.load(first / "reference.pt", weights_only=False)
    replayed, second = _factory._generate(
        restored, factory.config, region_id="interactive"
    )
    assert registration == replayed == candidate
    assert first != second
    assert (first / "accepted.json").is_file()
    assert (second / "cache_check.json").is_file()
    factory.author.assert_called_once()


def test_malformed_evaluator_json_is_reported_as_a_failure(tmp_path, monkeypatch):
    def run_process(command, *, cwd, **kwargs):
        (cwd / "result.json").write_text("[]")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(_factory._providers, "run_process", run_process)
    report = _factory._evaluate({}, tmp_path, 1)
    assert report == {
        "status": "error",
        "error": "Evaluator returned an invalid report",
    }


def test_exhausted_deadline_prevents_evaluation_and_further_agent_calls(
    factory, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(_factory, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def slow_proposal(**kwargs):
        clock[0] = 2.0
        return _candidate("late")

    factory.author.side_effect = slow_proposal
    with pytest.raises(RegionError, match="budget exhausted"):
        _generate(factory, timeout_s=1, max_attempts=4)
    factory.author.assert_called_once()
    factory.evaluate.assert_not_called()
    factory.register.assert_not_called()


def test_prompt_history_bounds_source_context_and_keeps_the_best_candidate():
    history = [
        {"candidate": _candidate(f"kernel_{index}"), "result": _passed(index + 1)}
        for index in range(20)
    ]
    selected = _factory._prompt_history(history)
    assert selected == [history[0], *history[-3:]]
    assert len(history) == 20


def test_prompt_history_keeps_paired_comparison_winner_despite_old_faster_timing():
    history = [
        {"candidate": _candidate(f"kernel_{index}"), "result": _passed(index + 1)}
        for index in range(10)
    ]
    assert _factory._prompt_history(history, history[2]["candidate"]) == [
        history[2],
        *history[-3:],
    ]


def test_prompt_history_without_a_winner_keeps_recent_failures():
    history = [
        {"candidate": None, "result": {"status": "error", "error": str(index)}}
        for index in range(10)
    ]
    assert _factory._prompt_history(history) == history[-3:]
    assert _factory._prompt_history([]) == []
    assert _factory._prompt_history(history[:2]) == history[:2]


def test_generation_snapshots_user_guidance_and_invalidates_search_cache(
    factory, tmp_path
):
    guidance = tmp_path / "guidance.md"
    guidance.write_text("Prefer a block size of 32.")
    factory.author.return_value = _candidate("guided")
    factory.evaluate.return_value = _passed(0.2)
    _generate(
        factory, instruction_files=(guidance,), instructions="Preserve exact rounding."
    )
    assert "Prefer a block size of 32" in factory.author.call_args.kwargs["prompt"]
    assert "Preserve exact rounding" in factory.author.call_args.kwargs["prompt"]
    bundle = json.loads(
        next(factory.root.glob("*/run-*/instructions.json")).read_text()
    )
    guidance.write_text("Prefer a block size of 64.")
    _generate(
        factory, instruction_files=(guidance,), instructions="Preserve exact rounding."
    )
    assert factory.author.call_count == 2
    assert len(list(factory.root.glob("*/accepted.json"))) == 2
    assert "Prefer a block size of 32" in bundle["text"]


def test_search_shares_inputs_and_final_validation_uses_fresh_inputs(factory):
    factory.author.side_effect = [_candidate("first"), _candidate("second")]
    factory.evaluate.side_effect = [_passed(0.2), _passed(0.1)]
    _generate(factory, max_attempts=2)
    requests = [call.args[0] for call in factory.evaluate.call_args_list]
    assert requests[0]["validation_seed"] == requests[1]["validation_seed"]
    assert requests[1]["incumbent"] == _candidate("first")
    final = factory.finalize.call_args.args[0]
    assert final["validation_phase"] == "holdout"
    assert final["validation_seed"] != requests[0]["validation_seed"]
    assert final["candidate"] == _candidate("second")
    assert "validation_seed" not in factory.author.call_args.kwargs["prompt"]


def test_failed_final_validation_cannot_publish_a_kernel(factory):
    factory.author.return_value = _candidate("overfit")
    factory.evaluate.return_value = _passed(0.1)
    factory.finalize.return_value = {
        "status": "error",
        "stage": "correctness",
        "error": "Fresh input mismatch",
    }
    with pytest.raises(RegionError, match="final validation.*Fresh input mismatch"):
        _generate(factory)
    assert not list(factory.root.glob("*/accepted.json"))
    factory.register.assert_not_called()


def test_incumbent_comparison_retains_winner_when_improvement_is_within_noise(factory):
    factory.author.side_effect = [_candidate("stable"), _candidate("noisy")]
    factory.evaluate.side_effect = [
        _passed(0.2),
        {
            **_passed(0.1),
            "incumbent_latency_ms": 0.101,
            "benchmark": {
                "candidate": {"mad_ms": 0.002},
                "incumbent": {"mad_ms": 0.002},
            },
        },
    ]
    assert _generate(factory, max_attempts=2) == _candidate("stable")


@pytest.mark.parametrize("objective", ["eager", "tensorrt"])
def test_required_baseline_speedup_is_enforced(factory, objective):
    factory.author.return_value = _candidate("slower")
    factory.evaluate.return_value = {**_passed(1.0), "baseline_latency_ms": 1.02}
    with pytest.raises(RegionError, match="performance objective"):
        _generate(factory, objective=objective, min_speedup=1.05)
    factory.finalize.assert_not_called()
    assert not list(factory.root.glob("*/accepted.json"))


@pytest.mark.parametrize("baseline, accepted", [(1.01, False), (1.3, True)])
def test_baseline_speedup_requires_a_margin_beyond_measured_noise(baseline, accepted):
    report = {
        **_passed(1.0),
        "baseline_latency_ms": baseline,
        "benchmark": {
            "candidate": {"mad_ms": 0.1},
            "baseline": {"mad_ms": 0.1},
        },
    }
    assert (
        _factory._meets_objective(report, KernelGenerator(objective="eager"))
        == accepted
    )


def test_baseline_objective_must_still_pass_on_fresh_final_inputs(factory):
    factory.author.return_value = _candidate("search_winner")
    factory.evaluate.return_value = {**_passed(1.0), "baseline_latency_ms": 2.0}
    factory.finalize.return_value = {**_passed(1.0), "baseline_latency_ms": 1.0}
    with pytest.raises(RegionError, match="final validation.*Performance objective"):
        _generate(factory, objective="eager", min_speedup=1.05)
    assert not list(factory.root.glob("*/accepted.json"))


def test_author_can_read_only_catalog_evidence_before_proposing(factory):
    factory.author.side_effect = [
        {"action": "read", "candidate": None, "resource": "task.json", "offset": 0},
        {
            "action": "propose",
            "candidate": _candidate("read"),
            "resource": None,
            "offset": None,
        },
    ]
    factory.evaluate.return_value = _passed(0.2)
    _generate(factory)
    assert factory.author.call_count == 2
    assert "EVIDENCE RESULT" in factory.author.call_args.kwargs["prompt"]
    assert "3 evidence reads remaining" in factory.author.call_args.kwargs["prompt"]
    assert factory.author.call_args.kwargs["schema"]["properties"]["candidate"][
        "anyOf"
    ][0]["properties"]["num_warps"]
    evidence = json.loads(
        next(factory.root.glob("*/run-*/attempt-1/evidence.json")).read_text()
    )
    assert evidence["result"]["resource"] == "task.json"
    assert "reference" in evidence["result"]["text"]
    factory.evaluate.assert_called_once()


def test_evidence_reads_are_bounded_without_invoking_gpu(factory):
    factory.author.return_value = {
        "action": "read",
        "candidate": None,
        "resource": "/etc/passwd",
        "offset": 0,
    }
    with pytest.raises(RegionError, match="Evidence request limit"):
        _generate(factory, max_context_requests=1)
    assert factory.author.call_count == 2
    assert "Unknown evidence resource" in factory.author.call_args.kwargs["prompt"]
    factory.evaluate.assert_not_called()


def test_author_timeout_reserves_evaluation_time(factory):
    factory.author.return_value = _candidate("budgeted")
    factory.evaluate.return_value = _passed(0.2)
    _generate(factory, timeout_s=100, author_timeout_s=90)
    assert 0 < factory.author.call_args.kwargs["timeout_s"] <= 50


def test_custom_provider_receives_structured_catalog(factory):
    factory.author.return_value = _candidate("custom_author")
    factory.evaluate.return_value = _passed(0.2)
    _generate(factory, provider="example:author")
    catalog = factory.author.call_args.kwargs["evidence"]["catalog"]
    assert any(entry["resource"] == "task.json" for entry in catalog)
    assert all("sha256" in entry for entry in catalog)


def test_backend_owns_candidate_fields_without_a_required_source_key(
    factory, monkeypatch
):
    proposal = {"artifact": "custom-format"}
    backend = SimpleNamespace(
        name="custom",
        version="1",
        instruction_resources=(),
        proposal_schema={
            "type": "object",
            "properties": {"artifact": {"type": "string"}},
            "required": ["artifact"],
            "additionalProperties": False,
        },
        target_metadata=lambda device: {},
        validate_candidate=lambda candidate, inputs, outputs: dict(candidate),
        make_registration=factory.register,
    )
    monkeypatch.setitem(sys.modules, "test_backend", SimpleNamespace(adapter=backend))
    factory.author.return_value = proposal
    factory.evaluate.return_value = _passed(0.2)
    assert _generate(factory, backend="test_backend:adapter") == proposal
    assert _generate(factory, backend="test_backend:adapter") == proposal
    factory.author.assert_called_once()


@pytest.mark.parametrize(
    "options",
    [
        {"instructions": 42},
        {"instruction_files": "instructions.md"},
        {"author_timeout_s": 0},
        {"max_context_requests": -1},
        {"benchmark_repeats": False},
        {"objective": "unknown"},
        {"min_speedup": 0.5},
    ],
)
def test_invalid_extension_and_search_options_are_rejected(options):
    with pytest.raises((ValueError, TypeError)):
        KernelGenerator(**options)


def test_reusing_old_paired_measurements_cannot_replace_a_faster_winner(factory):
    first, second, third = [_candidate(name) for name in ("first", "second", "third")]
    factory.author.side_effect = [first, second, third, second]
    factory.evaluate.side_effect = [
        _passed(0.3),
        {**_passed(0.2), "incumbent_latency_ms": 0.3},
        {**_passed(0.1), "incumbent_latency_ms": 0.2},
    ]
    assert _generate(factory, max_attempts=4) == third
    assert factory.evaluate.call_count == 3


def test_search_worker_cannot_consume_final_validation_budget(factory, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(_factory, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    factory.author.return_value = _candidate("reserved")

    def evaluate(request, directory, timeout):
        assert timeout == pytest.approx(60)
        clock[0] += timeout
        return _passed(0.2)

    factory.evaluate.side_effect = evaluate
    assert _generate(factory, timeout_s=90, candidate_timeout_s=120) == _candidate(
        "reserved"
    )
    assert factory.finalize.call_args.args[2] == pytest.approx(30)


def test_author_stop_finalizes_the_measured_winner(factory):
    candidate = _candidate("enough")
    factory.author.side_effect = [candidate, {"action": "stop"}]
    factory.evaluate.return_value = _passed(0.2)
    assert _generate(factory, max_attempts=4) == candidate
    assert factory.author.call_count == 2
    factory.evaluate.assert_called_once()
    factory.finalize.assert_called_once()
    search = json.loads(next(factory.root.glob("*/run-*/search.json")).read_text())
    assert search["stop_reason"] == "author_stop"


def test_author_diagnostic_uses_current_winner_and_shared_evaluation_budget(factory):
    candidate = _candidate("measured")
    factory.author.side_effect = [candidate, {"action": "diagnose"}, {"action": "stop"}]
    factory.evaluate.side_effect = [
        _passed(0.2),
        {**_passed(0.3), "diagnostics": {"candidate": {"compilation_ms": 123}}},
    ]
    assert (
        _generate(
            factory,
            max_attempts=3,
            diagnostics=True,
            search=SearchPolicy(max_candidates=2),
        )
        == candidate
    )
    assert factory.evaluate.call_count == 2
    diagnostic = factory.evaluate.call_args.args[0]
    assert diagnostic["candidate"] == candidate
    assert diagnostic["diagnostics"] is True
    assert "compilation_ms" in factory.author.call_args.kwargs["prompt"]
    assert "123" in factory.author.call_args.kwargs["prompt"]
    factory.finalize.assert_called_once()


def test_disabled_diagnostics_return_feedback_without_gpu_work(factory):
    factory.author.side_effect = [{"action": "diagnose"}, _candidate("normal")]
    factory.evaluate.return_value = _passed(0.2)
    assert _generate(factory) == _candidate("normal")
    assert "Diagnostics are not enabled" in factory.author.call_args.kwargs["prompt"]
    factory.evaluate.assert_called_once()


def test_stagnation_stops_after_complete_author_rounds(factory):
    first, tied, ignored = [_candidate(name) for name in ("first", "tied", "ignored")]
    factory.author.side_effect = [first, tied, ignored]
    factory.evaluate.return_value = _passed(0.2)
    assert (
        _generate(factory, max_attempts=5, search=SearchPolicy(max_no_improvement=1))
        == first
    )
    assert factory.author.call_count == 2
    assert factory.evaluate.call_count == 2


@pytest.mark.parametrize(
    "stage,action",
    [("compile", "repair_compile"), ("correctness", "repair_correctness")],
)
def test_measured_failure_selects_targeted_repair(factory, stage, action):
    factory.author.side_effect = [_candidate("broken"), _candidate("fixed")]
    factory.evaluate.side_effect = [
        {"status": "error", "stage": stage, "error": "failed"},
        _passed(0.2),
    ]
    assert _generate(factory, max_attempts=2) == _candidate("fixed")
    assert f"SEARCH STEP: {action}" in factory.author.call_args.kwargs["prompt"]


def test_author_cannot_exceed_batch_policy(factory):
    factory.author.return_value = {
        "action": "propose",
        "candidates": [_candidate("one"), _candidate("two")],
    }
    with pytest.raises(RegionError, match="bounded batch"):
        _generate(factory)
    factory.evaluate.assert_not_called()
