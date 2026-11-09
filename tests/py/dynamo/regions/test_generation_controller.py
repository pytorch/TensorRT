# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Capture, feedback, selection, and cache policy without agent or GPU access."""

import json
import logging
import operator
import os
import re
import stat
import subprocess
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
from torch_tensorrt.dynamo.region_compilation._generation._reference import (
    load_reference,
)
from torch_tensorrt.region import (
    KernelGenerator,
    RegionError,
    SearchPolicy,
    fused_kernel,
)
from torch_tensorrt.region._session import RegionCompilationSession

from ._generation_utils import _candidate, _generate, _passed, _proposal

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Kernel generation requires a POSIX host"
)


class GeneratedRegion(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x):
        with fused_kernel(generator=self.generator, name="generated_activation"):
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
    with pytest.raises(error, match=match), fused_kernel(**options):
        pass


@pytest.mark.parametrize(
    "options,error,match",
    [
        ({"max_attempts": True}, TypeError, "max_attempts must be a positive integer"),
        ({"timeout_s": "600"}, TypeError, "timeout_s must be a finite positive"),
        ({"timeout_s": float("nan")}, ValueError, "timeout_s must be a finite"),
        ({"timeout_s": 10**400}, ValueError, "timeout_s must be a finite"),
        ({"timeout_s": True}, TypeError, "timeout_s must be a finite positive"),
        (
            {"min_speedup": True, "objective": "eager"},
            TypeError,
            "min_speedup must be a finite positive",
        ),
        ({"rtol": False, "atol": 0}, TypeError, "rtol must be a finite nonnegative"),
        ({"candidate_timeout_s": 0}, ValueError, "candidate_timeout_s"),
        ({"max_context_requests": -1}, ValueError, "nonnegative integer"),
        ({"rtol": 0.1}, ValueError, "rtol and atol must be set together"),
        ({"rtol": -1, "atol": 0}, ValueError, "rtol must be a finite nonnegative"),
        ({"min_speedup": 0.5, "objective": "eager"}, ValueError, "at least 1"),
        ({"min_speedup": 1.05}, ValueError, "requires objective='eager' or"),
        ({"author_timeout_s": 0}, ValueError, "author_timeout_s must be a finite"),
        ({"benchmark_repeats": False}, TypeError, "benchmark_repeats must be a"),
        ({"instructions": 42}, TypeError, "instructions must be a string"),
        ({"instruction_files": "guide.md"}, TypeError, "sequence of paths"),
        ({"objective": "unknown"}, ValueError, "objective must be 'candidate'"),
        ({"allow_generation": 0}, TypeError, "allow_generation must be a bool"),
        (
            {"allow_generation": False, "use_cache": False},
            ValueError,
            "allow_generation=False requires use_cache=True",
        ),
        ({"max_protocol_retries": -1}, ValueError, "max_protocol_retries"),
        ({"aligned_io": "yes"}, TypeError, "aligned_io"),
        ({"fp32_matmul_precision": "medium"}, ValueError, "'highest' or 'high'"),
        (
            {"fp32_matmul_precision": None},
            TypeError,
            "fp32_matmul_precision must be a string",
        ),
    ],
)
def test_invalid_generation_budgets_and_tolerances_are_rejected(options, error, match):
    with pytest.raises(error, match=match):
        KernelGenerator(**options)


_FLOAT_SETTINGS = (
    "timeout_s",
    "candidate_timeout_s",
    "author_timeout_s",
    "min_speedup",
    "rtol",
    "atol",
)


def test_whole_numbers_in_float_settings_are_stored_as_floats():
    # Settings enter the task's cache key as JSON, where 2 and 2.0 differ.
    whole = KernelGenerator(
        objective="eager",
        min_speedup=2,
        rtol=0,
        atol=1,
        timeout_s=600,
        candidate_timeout_s=60,
        author_timeout_s=60,
    )
    fractional = KernelGenerator(
        objective="eager",
        min_speedup=2.0,
        rtol=0.0,
        atol=1.0,
        timeout_s=600.0,
        candidate_timeout_s=60.0,
        author_timeout_s=60.0,
    )
    assert [type(getattr(whole, name)) for name in _FLOAT_SETTINGS] == [float] * 6
    assert whole == fractional and hash(whole) == hash(fractional)
    assert json.dumps([getattr(whole, name) for name in _FLOAT_SETTINGS]) == json.dumps(
        [getattr(fractional, name) for name in _FLOAT_SETTINGS]
    )
    # The defaults are whole numbers too.
    assert type(KernelGenerator().timeout_s) is float


def test_whole_number_settings_share_the_cache_key_of_their_float_values(factory):
    """The replay CLI parses these settings as floats; a cache-only compile
    must find the kernel that a compile spelling them as ints published."""
    factory.author.return_value = _candidate("published")
    report = {**_passed(0.5), "baseline_latency_ms": 2.0}
    factory.evaluate.return_value = report
    factory.finalize.return_value = report
    _generate(factory, objective="eager", min_speedup=2, rtol=0, atol=1)
    reused = _generate(
        factory,
        allow_generation=False,
        objective="eager",
        min_speedup=2.0,
        rtol=0.0,
        atol=1.0,
    )
    assert reused == _candidate("published")
    factory.author.assert_called_once()
    assert len(list(factory.root.glob("*/accepted.json"))) == 1


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
    saved_reference = load_reference(request["reference_path"])
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
        assert history[index]["result"]["reused_from"] == "attempt-1/feedback.json"
        assert history[index]["result"]["latency_ms"] == 0.2
        attempt = history_path.parent / f"attempt-{index + 1}"
        assert (
            json.loads((attempt / "feedback.json").read_text())
            == history[index]["result"]
        )
        assert json.loads((attempt / "candidate.json").read_text()) == candidate
    accepted = json.loads(next(factory.root.glob("*/accepted.json")).read_text())
    assert accepted["evaluation_path"] == "attempt-1/feedback.json"
    assert json.loads(evaluation_path.read_text()) == accepted["report"]
    assert "an unchanged candidate reuses its earlier pass" in (
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
    assert accepted["evaluation_path"] == "attempt-2/feedback.json"


@pytest.mark.parametrize(
    "failure",
    [
        {"status": "error", "error": "GPU evaluation timed out"},
        {"status": "error", "stage": "timeout", "error": "Evaluator exceeded"},
        {"status": "error", "stage": "benchmark", "error": "Graph capture failed"},
    ],
)
def test_repeated_nondeterministic_failure_is_evaluated_again(factory, failure):
    candidate = _candidate("retry")
    factory.author.return_value = candidate
    factory.evaluate.side_effect = [failure, _passed(0.2)]

    assert _generate(factory, max_attempts=2) == candidate

    assert factory.evaluate.call_count == 2
    accepted = json.loads(next(factory.root.glob("*/accepted.json")).read_text())
    assert accepted["evaluation_path"] == "attempt-2/feedback.json"


def test_cache_reuses_a_kernel_without_an_agent_and_revalidates_it(factory):
    candidate = _candidate("cached")
    factory.author.return_value = candidate
    factory.evaluate.return_value = _passed(0.2)
    assert _generate(factory) == candidate
    assert _generate(factory, provider="codex") == candidate
    factory.author.assert_called_once()
    # One search evaluation; final validation and the cache check use fresh inputs.
    factory.evaluate.assert_called_once()
    assert [
        call.args[0]["validation_phase"] for call in factory.finalize.call_args_list
    ] == ["holdout", "holdout"]
    cache_check = next(factory.root.glob("*/run-*/cache_check.json"))
    assert json.loads(cache_check.read_text())["status"] == "passed"


def test_cache_revalidation_failure_triggers_new_generation(factory):
    old, new = _candidate("old"), _candidate("new")
    factory.author.side_effect = [old, new]
    factory.evaluate.side_effect = [_passed(0.2), _passed(0.1)]
    factory.finalize.side_effect = [
        _passed(0.2),
        {"status": "error", "error": "Cached kernel failed validation"},
        _passed(0.1),
    ]
    assert _generate(factory) == old
    assert _generate(factory) == new
    assert factory.author.call_count == 2
    assert factory.evaluate.call_count == 2
    assert factory.finalize.call_count == 3
    accepted = json.loads(next(factory.root.glob("*/accepted.json")).read_text())
    assert accepted["candidate"] == new


def test_cache_only_generation_reuses_a_revalidated_kernel_without_an_agent(
    factory,
):
    factory.author.return_value = _candidate("published")
    factory.evaluate.return_value = _passed(0.2)
    assert _generate(factory) == _candidate("published")
    assert _generate(factory, allow_generation=False) == _candidate("published")
    factory.author.assert_called_once()
    factory.evaluate.assert_called_once()
    # Final validation, then the cache-only compilation's revalidation.
    assert factory.finalize.call_count == 2


@pytest.mark.parametrize(
    "cache, reason",
    [
        ("missing", "no accepted kernel is cached for this task"),
        ("corrupt", "its cached kernel entry is invalid"),
        (
            "stale",
            "its cached kernel failed revalidation (correctness: Fresh inputs "
            "disagree; see cache_check.json)",
        ),
    ],
)
def test_cache_only_generation_fails_without_a_usable_kernel(factory, cache, reason):
    factory.author.return_value = _candidate("published")
    factory.evaluate.return_value = _passed(0.2)
    if cache == "missing":
        factory.root.mkdir()
    else:
        _generate(factory)
        factory.author.reset_mock()
        factory.evaluate.reset_mock()
    if cache == "corrupt":
        manifest = next(factory.root.glob("*/accepted.json"))
        stored = json.loads(manifest.read_text())
        stored["candidate_sha256"] = "corrupted"
        manifest.write_text(json.dumps(stored))
    elif cache == "stale":
        factory.finalize.return_value = {
            "status": "error",
            "stage": "correctness",
            "error": "Fresh inputs disagree",
        }
    with pytest.raises(RegionError) as failure:
        _generate(factory, allow_generation=False)
    message = str(failure.value)
    assert "'activation' is disabled by allow_generation=False" in message
    assert reason in message
    assert "Compile once with generation allowed" in message
    factory.author.assert_not_called()
    factory.evaluate.assert_not_called()
    published = list(factory.root.glob("*/accepted.json"))
    assert len(published) == (0 if cache == "missing" else 1)


@pytest.mark.parametrize(
    "stage, error, inconclusive",
    [
        ("budget", "GPU evaluation lock timed out", True),
        ("setup", "Evaluator exited 1: CUDA out of memory", True),
        ("timeout", "Evaluator exceeded its 120s timeout", True),
        ("baseline", "TensorRT baseline build failed", True),
        ("crash", "Evaluator was killed by signal 11", False),
        ("correctness", "Output 0 is wrong for case 'random 1'", False),
    ],
)
def test_cache_only_refusal_names_the_revalidation_failure(
    factory, stage, error, inconclusive
):
    factory.author.return_value = _candidate("published")
    factory.evaluate.return_value = _passed(0.2)
    _generate(factory)
    factory.finalize.return_value = {
        "status": "error",
        "stage": stage,
        "error": f"Traceback (most recent call last):\n  ...\n{error}\n",
    }
    with pytest.raises(RegionError) as failure:
        _generate(factory, allow_generation=False)
    message = str(failure.value)
    outcome = "could not be revalidated" if inconclusive else "failed revalidation"
    assert f"its cached kernel {outcome} ({stage}: {error}; see" in message
    # Only a check that says nothing about the kernel is worth retrying as is.
    assert ("Retry once the cause is resolved" in message) == inconclusive
    assert ("Compile once with generation allowed" in message) != inconclusive
    factory.author.assert_called_once()


def test_cache_only_refusal_reports_a_remeasured_speedup_below_the_objective(
    factory,
):
    factory.author.return_value = _candidate("published")
    factory.evaluate.return_value = {**_passed(1.0), "baseline_latency_ms": 1.2}
    factory.finalize.return_value = {**_passed(1.0), "baseline_latency_ms": 1.2}
    objective = {"objective": "eager", "min_speedup": 1.1}
    _generate(factory, **objective)
    factory.finalize.return_value = {
        **_passed(1.0),
        "baseline_latency_ms": 1.2,
        "benchmark": {"baseline": {"mad_ms": 0.15}, "candidate": {"mad_ms": 0.0}},
    }
    with pytest.raises(RegionError) as failure:
        _generate(factory, allow_generation=False, **objective)
    message = str(failure.value)
    # The raw ratio, 1.2x, would contradict the refusal; report the compared one.
    assert (
        "its cached kernel passed the correctness checks, but its remeasured "
        "speedup over the eager baseline after allowing for timing noise, 1.05x "
        "(baseline 1.2 ± 0.15 ms, kernel 1 ± 0 ms), is below min_speedup=1.1"
    ) in message
    # A correct kernel is most likely slow because the GPU is busy: retry first.
    assert "Retry on a less busy GPU; if the speedup stays below min_speedup" in message


def test_cache_only_refusal_without_a_remeasured_speedup_gives_no_speed_advice(
    factory,
):
    factory.author.return_value = _candidate("published")
    factory.evaluate.return_value = {**_passed(1.0), "baseline_latency_ms": 1.2}
    factory.finalize.return_value = {**_passed(1.0), "baseline_latency_ms": 1.2}
    objective = {"objective": "eager", "min_speedup": 1.1}
    _generate(factory, **objective)
    factory.finalize.return_value = _passed(1.0)
    with pytest.raises(RegionError) as failure:
        _generate(factory, allow_generation=False, **objective)
    message = str(failure.value)
    assert (
        "its cached kernel passed revalidation without a measured eager baseline"
    ) in message
    # No speedup was measured, so none can be retried for.
    assert "less busy GPU" not in message
    assert "Compile once with generation allowed" in message


def test_cache_only_compilation_never_creates_a_missing_cache_directory(factory):
    with pytest.raises(RegionError) as failure:
        _generate(factory, allow_generation=False)
    assert (
        f"the kernel cache directory {factory.root} does not exist; check cache_dir"
    ) in str(failure.value)
    assert not factory.root.exists()
    factory.author.assert_not_called()


def test_cache_only_miss_names_what_differs_from_a_cached_kernel(factory):
    factory.author.return_value = _candidate("published")
    factory.evaluate.return_value = _passed(0.2)
    _generate(factory)
    task = next(factory.root.glob("*/accepted.json")).parent
    # A field outside the cache key, even one the cached task lacks, is no
    # reason for a miss.
    saved = next(task.glob("run-*/task.json"))
    saved.write_text(
        json.dumps(
            {k: v for k, v in json.loads(saved.read_text()).items() if k != "budget"}
        )
    )
    # Validation settings are part of the cache key; budgets are not.
    with pytest.raises(RegionError) as failure:
        _generate(factory, allow_generation=False, num_tests=9, max_attempts=7)
    assert (
        "no accepted kernel is cached for this task; a kernel is cached for the "
        "same captured computation, but its task.json differs in validation "
        f"(compare {task}/run-"
    ) in str(failure.value)
    factory.author.assert_called_once()


def test_tuning_choices_alone_of_the_search_policy_are_in_the_cache_key(factory):
    factory.author.return_value = _candidate("published")
    factory.evaluate.return_value = _passed(0.2)
    _generate(factory, search=SearchPolicy(tuning={"num_warps": (4, 8)}))
    # Other search settings do not change the key, nor do lists for tuples.
    reused = _generate(
        factory,
        allow_generation=False,
        search=SearchPolicy(tuning={"num_warps": [4, 8]}, batch_size=2, finalists=2),
    )
    assert reused == _candidate("published")
    # Changed choices do, even when the cached kernel's value stays allowed.
    with pytest.raises(RegionError) as failure:
        _generate(
            factory,
            allow_generation=False,
            search=SearchPolicy(tuning={"num_warps": (4,)}),
        )
    assert (
        "a kernel is cached for the same captured computation, but its task.json "
        "differs in search.tuning (compare"
    ) in str(failure.value)
    factory.author.assert_called_once()


def test_cache_only_miss_says_when_no_kernel_is_cached_for_the_computation(
    factory,
):
    factory.root.mkdir()
    with pytest.raises(RegionError) as failure:
        _generate(factory, allow_generation=False)
    assert (
        "no accepted kernel is cached for this task, and none of the 0 kernels "
        f"cached in {factory.root} is for this captured computation"
    ) in str(failure.value)


@pytest.mark.parametrize("switch", ["0", "1"])
def test_environment_switch_can_forbid_generation_for_every_generator(
    factory, monkeypatch, switch
):
    monkeypatch.setenv("TORCH_TENSORRT_REGION_GENERATION", switch)
    factory.author.return_value = _candidate("allowed")
    factory.evaluate.return_value = _passed(0.2)
    if switch == "1":
        assert _generate(factory) == _candidate("allowed")
        return
    with pytest.raises(
        RegionError, match="disabled by TORCH_TENSORRT_REGION_GENERATION=0"
    ):
        _generate(factory)
    factory.author.assert_not_called()
    factory.evaluate.assert_not_called()


@pytest.mark.parametrize("switch", ["false", "off", ""])
def test_unrecognized_generation_switch_fails_before_any_work(
    factory, monkeypatch, switch
):
    monkeypatch.setenv("TORCH_TENSORRT_REGION_GENERATION", switch)
    with pytest.raises(RegionError, match="must be 0 or 1"):
        _generate(factory)
    factory.author.assert_not_called()
    assert not factory.root.exists()


def test_disabled_generation_and_cache_leave_nothing_to_use(factory, monkeypatch):
    monkeypatch.setenv("TORCH_TENSORRT_REGION_GENERATION", "0")
    with pytest.raises(RegionError, match="use_cache=False leaves no cached kernel"):
        _generate(factory, use_cache=False)
    factory.author.assert_not_called()
    assert not factory.root.exists()


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
    search = json.loads(next(factory.root.glob("*/run-*/search.json")).read_text())
    assert search["stop_reason"] == "provider"


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


def _calling(target, tensors, *args, **kwargs):
    """A reference that calls ``target`` once, on its first output if several."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    value = graph.call_function(target, (x,) * tensors + args, kwargs)
    if len(target._schema.returns) > 1:
        value = graph.call_function(operator.getitem, (value, 0))
    graph.output((value,))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


@pytest.mark.parametrize(
    "graph",
    [
        # PyTorch tags these as random, but they draw no random numbers here.
        _calling(torch.ops.aten.scaled_dot_product_attention.default, 3),
        _calling(
            torch.ops.aten.scaled_dot_product_attention.default, 3, is_causal=True
        ),
        _calling(torch.ops.aten.dropout.default, 1, 0.1, False),
    ],
    ids=["attention", "causal_attention", "eval_dropout"],
)
def test_operations_tagged_random_that_draw_none_reach_the_author(factory, graph):
    factory.graph = graph
    factory.author.return_value = {
        "action": "stop",
        "candidates": None,
        "resource": None,
        "offset": None,
    }
    with pytest.raises(RegionError, match="the author stopped"):
        _generate(factory)
    target = next(node.target for node in graph.graph.nodes if node.op != "placeholder")
    assert str(target) in factory.author.call_args.kwargs["prompt"]


@pytest.mark.parametrize(
    "graph, message",
    [
        (
            _calling(
                torch.ops.aten.scaled_dot_product_attention.default, 3, dropout_p=0.1
            ),
            r"but aten.scaled_dot_product_attention.default draws them here; it "
            r"draws none when its dropout_p argument is 0 or False\. With "
            r"decompose_attention=True",
        ),
        (
            _calling(torch.ops.aten._scaled_dot_product_flash_attention.default, 3),
            r"but PyTorch tags aten._scaled_dot_product_flash_attention.default as "
            r"random\. With decompose_attention=True",
        ),
        (
            _calling(torch.ops.aten.dropout.default, 1, 0.1, True),
            r"but aten.dropout.default draws them here; it draws none when its p "
            r"or train argument is 0 or False$",
        ),
        (
            _calling(torch.ops.aten.rand_like.default, 1),
            r"but PyTorch tags aten.rand_like.default as random$",
        ),
    ],
    ids=["attention_dropout", "flash_attention", "train_dropout", "rand_like"],
)
def test_random_operations_are_named_before_agent_invocation(factory, graph, message):
    factory.graph = graph
    with pytest.raises(
        RegionError, match="requires a reference that draws no random numbers"
    ) as error:
        _generate(factory)
    assert re.search(message, str(error.value)), str(error.value)
    factory.author.assert_not_called()
    factory.evaluate.assert_not_called()


def test_saved_reference_reuses_cache_and_reports_its_exact_validation_run(factory):
    candidate = _candidate("round_trip")
    factory.author.return_value = candidate
    factory.evaluate.return_value = _passed(0.2)
    registration, first = _factory._generate(
        factory.graph, factory.config, region_id="original"
    )
    restored = load_reference(first / "reference.json")
    replayed, second = _factory._generate(
        restored, factory.config, region_id="interactive"
    )
    assert registration == replayed == candidate
    assert first != second
    assert (first / "accepted.json").is_file()
    assert (second / "cache_check.json").is_file()
    factory.author.assert_called_once()


@pytest.mark.parametrize(
    "result,error",
    [
        ("[]", "Evaluator returned an invalid report"),
        (
            '{"status": "passed", "latency_ms": 0}',
            "Evaluator returned an invalid timing",
        ),
        (
            '{"status": "passed", "latency_ms": NaN}',
            "Evaluator returned an invalid timing",
        ),
    ],
)
def test_malformed_evaluator_results_are_setup_failures(
    tmp_path, monkeypatch, result, error
):
    def run_process(command, *, cwd, **kwargs):
        (cwd / "result.json").write_text(result)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(_factory._providers, "run_process", run_process)
    report = _factory._evaluate({}, tmp_path, 1)
    assert report == {"status": "error", "stage": "setup", "error": error}


def test_disabled_cache_neither_reuses_nor_publishes_accepted_kernels(factory):
    candidate = _candidate("published")
    factory.author.return_value = candidate
    factory.evaluate.return_value = _passed(0.2)
    assert _generate(factory) == candidate
    manifest = next(factory.root.glob("*/accepted.json"))
    published = manifest.read_text()
    assert _generate(factory, use_cache=False) == candidate
    assert factory.author.call_count == 2
    assert manifest.read_text() == published
    fresh = factory.root.parent / "fresh"
    assert _generate(factory, use_cache=False, cache_dir=fresh) == candidate
    assert not list(fresh.glob("*/accepted.json"))


def test_tampered_checkpoint_is_ignored_on_resume(factory):
    saved = _candidate("saved")
    factory.author.return_value = saved
    factory.evaluate.return_value = _passed(0.2)
    factory.finalize.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        _generate(factory)
    checkpoint = next(factory.root.glob("*/checkpoint.json"))
    content = json.loads(checkpoint.read_text())
    injected = _candidate("injected")
    # Edit the state without updating its digest.
    content["state"]["finalists"] = [injected]
    checkpoint.write_text(json.dumps(content))
    factory.author.reset_mock()
    factory.evaluate.reset_mock()
    factory.finalize.side_effect = None
    assert _generate(factory, search=SearchPolicy(resume=True)) == saved
    factory.author.assert_called_once()
    evaluated = [call.args[0]["candidate"] for call in factory.evaluate.call_args_list]
    assert injected not in evaluated


def test_exhausted_deadline_prevents_evaluation_and_further_agent_calls(
    factory, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(_factory, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def slow_proposal(**kwargs):
        clock[0] = 20.0
        return _candidate("late")

    factory.author.side_effect = slow_proposal
    with pytest.raises(RegionError, match="search time ran out.*0 of 4 evaluations"):
        _generate(factory, timeout_s=10, max_attempts=4)
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


@pytest.mark.parametrize(
    "measured, reason",
    [
        (
            {
                "baseline_latency_ms": 1.0,
                "benchmark": {
                    "baseline": {"mad_ms": 0.01},
                    "candidate": {"mad_ms": 0.01},
                },
            },
            "its speedup over the eager baseline after allowing for timing noise, "
            "1.09x (baseline 1 ± 0.01 ms, kernel 0.9 ± 0.01 ms), is below "
            "min_speedup=1.5",
        ),
        ({}, "no eager baseline was measured"),
    ],
)
def test_final_validation_refusal_for_speed_reports_the_measurements(
    factory, measured, reason
):
    factory.author.return_value = _candidate("search_winner")
    # A launch screen leaves the speedup to final validation's engine.
    factory.evaluate.return_value = {**_passed(0.9), "execution": "launch"}
    factory.finalize.return_value = {**_passed(0.9), "execution": "engine", **measured}
    with pytest.raises(RegionError) as failure:
        _generate(factory, objective="eager", min_speedup=1.5)
    assert (
        "failed final validation: Performance objective not met: "
        f"{reason} (see holdout/feedback.json). Artifacts:"
    ) in str(failure.value)


def test_author_can_read_only_catalog_evidence_before_proposing(factory):
    factory.author.side_effect = [
        {"action": "read", "candidates": None, "resource": "task.json", "offset": 0},
        _proposal(_candidate("read")),
    ]
    factory.evaluate.return_value = _passed(0.2)
    _generate(factory)
    assert factory.author.call_count == 2
    assert "EVIDENCE RESULT" in factory.author.call_args.kwargs["prompt"]
    assert "3 evidence reads remaining" in factory.author.call_args.kwargs["prompt"]
    schema = factory.author.call_args.kwargs["schema"]
    assert set(schema["required"]) == {"action", "candidates", "resource", "offset"}
    assert schema["properties"]["candidates"]["anyOf"][0]["items"]["properties"][
        "num_warps"
    ]
    evidence = json.loads(
        next(factory.root.glob("*/run-*/attempt-1/evidence.json")).read_text()
    )
    assert evidence["result"]["resource"] == "task.json"
    assert "reference" in evidence["result"]["text"]
    factory.evaluate.assert_called_once()


def test_a_read_without_a_resource_name_is_answered_with_an_error(factory):
    factory.author.side_effect = [
        {"action": "read", "candidates": None, "resource": None, "offset": 0},
        _proposal(_candidate("after_bad_read")),
    ]
    factory.evaluate.return_value = _passed(0.2)
    _generate(factory)
    evidence = json.loads(
        next(factory.root.glob("*/run-*/attempt-1/evidence.json")).read_text()
    )
    assert evidence["result"] == {"error": "Evidence resource must be a catalog name"}
    factory.evaluate.assert_called_once()


def test_evidence_reads_are_bounded_without_invoking_gpu(factory):
    factory.author.return_value = {
        "action": "read",
        "candidates": None,
        "resource": "/etc/passwd",
        "offset": 0,
    }
    with pytest.raises(RegionError, match="when no evidence reads remain"):
        _generate(factory, max_context_requests=1)
    # One read, then a read beyond the limit and its protocol retry.
    assert factory.author.call_count == 3
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
            search=SearchPolicy(max_candidates=3),
        )
        == candidate
    )
    assert factory.evaluate.call_count == 2
    search = json.loads(next(factory.root.glob("*/run-*/search.json")).read_text())
    assert search["evaluations"] == 2
    diagnostic = factory.evaluate.call_args.args[0]
    assert diagnostic["candidate"] == candidate
    assert diagnostic["diagnostics"] is True
    assert "compilation_ms" in factory.author.call_args.kwargs["prompt"]
    assert "123" in factory.author.call_args.kwargs["prompt"]
    factory.finalize.assert_called_once()


def test_diagnostics_cannot_take_the_last_candidate_evaluation(factory):
    first, second = _candidate("first"), _candidate("second")
    factory.author.side_effect = [first, {"action": "diagnose"}, second]
    factory.evaluate.return_value = _passed(0.2)
    _generate(
        factory,
        max_attempts=2,
        diagnostics=True,
        search=SearchPolicy(max_candidates=2),
    )
    # The refused diagnosis leaves the evaluation for the round's proposal.
    assert [call.args[0]["candidate"] for call in factory.evaluate.call_args_list] == [
        first,
        second,
    ]
    evidence = json.loads(
        next(factory.root.glob("*/run-*/attempt-2/evidence.json")).read_text()
    )
    assert "last candidate evaluation" in evidence["result"]["error"]


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


@pytest.mark.parametrize(
    "reply",
    [
        _candidate("bare"),
        {"action": "propose", "candidate": _candidate("singular"), "offset": None},
    ],
)
def test_provider_replies_must_use_the_candidates_envelope(factory, monkeypatch, reply):
    monkeypatch.setattr(_factory._providers, "propose", Mock(return_value=reply))
    with pytest.raises(RegionError, match="action must be|bounded batch"):
        _generate(factory)
    factory.evaluate.assert_not_called()


def test_task_records_budget_and_compile_options_used_by_every_request(factory):
    from torch_tensorrt.dynamo._settings import CompilationSettings

    factory.author.return_value = _candidate("configured")
    factory.evaluate.return_value = _passed(0.2)
    settings = CompilationSettings(disable_tf32=True, optimization_level=5)
    _factory.generate_registration(
        factory.graph,
        replace(factory.config, timeout_s=300),
        region_id="activation",
        settings=settings,
    )
    run = next(factory.root.glob("*/run-*"))
    task = json.loads((run / "task.json").read_text())
    assert task["version"] == 5
    assert task["budget"] == {
        "max_attempts": 1,
        "timeout_s": 300,
        "candidate_timeout_s": 300,
        "author_timeout_s": 300,
        "max_context_requests": 4,
        "max_protocol_retries": 1,
        "use_cache": True,
    }
    assert task["compile_options"] == {
        name: getattr(settings, name) for name in _factory._COMPILE_OPTIONS
    }
    assert task["compile_options"]["disable_tf32"] is True
    assert task["compile_options"]["optimization_level"] == 5
    template = json.loads((run / "evaluation-template.json").read_text())
    search = factory.evaluate.call_args.args[0]
    assert template == {
        key: value for key, value in search.items() if key != "candidate"
    }
    assert set(template) == {
        "reference_path",
        "samples_path",
        "backend",
        "backend_version",
        "proposal_schema_sha256",
        "inputs",
        "outputs",
        "objective",
        "diagnostics",
        "num_tests",
        "benchmark_iters",
        "benchmark_repeats",
        "check_nonfinite",
        "rtol",
        "atol",
        "validation_seed",
        "validation_phase",
        "screening",
        "check_baseline",
        "compile_options",
        "codegen",
        "fp32_matmul_precision",
        "auto_fit",
    }
    assert template["validation_phase"] == "search"
    assert template["screening"] == "launch"
    # Only the tensorrt objective has a TensorRT baseline to check.
    assert template["check_baseline"] is False
    assert template["samples_path"] is None and "samples" not in task
    assert template["compile_options"] == task["compile_options"]
    final = factory.finalize.call_args.args[0]
    assert final["compile_options"] == task["compile_options"]


def test_default_compile_options_come_from_compilation_settings(factory):
    from torch_tensorrt.dynamo._settings import CompilationSettings

    defaults = CompilationSettings()
    factory.author.return_value = _candidate("defaults")
    factory.evaluate.return_value = _passed(0.2)
    _generate(factory)
    assert factory.evaluate.call_args.args[0]["compile_options"] == {
        name: getattr(defaults, name) for name in _factory._COMPILE_OPTIONS
    }


@pytest.mark.parametrize("use_cache", [True, False])
def test_identical_regions_in_one_compilation_share_one_validated_kernel(
    factory, use_cache
):
    factory.author.return_value = _candidate("shared")
    factory.evaluate.return_value = _passed(0.2)
    config = replace(factory.config, use_cache=use_cache)
    memo = {}
    for layer in range(3):
        registration = _factory.generate_registration(
            factory.graph, config, region_id=f"layer{layer}", memo=memo
        )
        assert registration == _candidate("shared")
    factory.author.assert_called_once()
    factory.evaluate.assert_called_once()
    factory.finalize.assert_called_once()
    assert factory.register.call_count == 3
    assert len(list(factory.root.glob("*/run-*"))) == 1


@pytest.mark.parametrize("failure", ["deleted instruction file", "missing backend"])
def test_configuration_failures_name_the_region(factory, failure):
    # KernelGenerator rejects missing files and built-in tuning errors when it is
    # constructed; these failures can only appear once generation starts.
    if failure == "missing backend":
        config = replace(factory.config, backend="missing_generation_backend:Adapter")
        cause = ModuleNotFoundError
    else:
        guidance = factory.root.parent / "guidance.md"
        guidance.write_text("Prefer wide tiles.")
        config = replace(factory.config, instruction_files=(guidance,))
        guidance.unlink()
        cause = FileNotFoundError
    with pytest.raises(
        RegionError, match="generation for 'activation' failed"
    ) as error:
        _factory.generate_registration(factory.graph, config, region_id="activation")
    assert isinstance(error.value.__cause__, cause)
    factory.author.assert_not_called()


def test_target_probe_failures_name_the_region_and_region_errors_pass_through(
    factory, monkeypatch
):
    failure = RuntimeError("cuTile toolchain is unavailable")
    monkeypatch.setattr(_factory, "_target", Mock(side_effect=failure))
    with pytest.raises(RegionError, match="'activation' failed: cuTile") as error:
        _generate(factory)
    assert error.value.__cause__ is failure
    original = RegionError("Kernel generation backend requires cuda-tile")
    monkeypatch.setattr(_factory, "_target", Mock(side_effect=original))
    with pytest.raises(RegionError) as error:
        _generate(factory)
    assert error.value is original
    factory.author.assert_not_called()


def test_waiting_for_a_peers_generation_lock_keeps_the_full_budget(
    factory, monkeypatch
):
    factory.author.return_value = _candidate("published")
    factory.evaluate.return_value = _passed(0.2)
    assert _generate(factory, timeout_s=90, candidate_timeout_s=40) == (
        _candidate("published")
    )
    clock = [0.0]
    monkeypatch.setattr(_factory, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    class BusyLock:
        """A peer held the generation lock for most of our timeout."""

        def __init__(self, path, timeout):
            self.timeout = timeout

        def __enter__(self):
            clock[0] += 0.9 * self.timeout

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(_factory, "FileLock", BusyLock)
    assert _generate(factory, timeout_s=90, candidate_timeout_s=40) == (
        _candidate("published")
    )
    factory.author.assert_called_once()
    # The cache revalidation still receives a full candidate slice.
    assert factory.evaluate.call_args.args[2] == 40


@pytest.mark.parametrize(
    "returncode,stage,action", [(-11, "crash", "repair"), (1, "setup", "finalize")]
)
def test_worker_exit_status_separates_kernel_crashes_from_setup_failures(
    tmp_path, monkeypatch, returncode, stage, action
):

    # The worker reports Python failures itself; a signal means native code,
    # usually the candidate's kernel, crashed the process.
    def run_process(command, *, cwd, timeout_s, **kwargs):
        return subprocess.CompletedProcess(command, returncode, "", "fault\x1b[0m")

    monkeypatch.setattr(_factory._providers, "run_process", run_process)
    report = _factory._evaluate({}, tmp_path, 5)
    assert report["stage"] == stage and report["error"].endswith(": fault")
    entry = {"candidate": None, "result": report, "action": "author"}
    assert _factory._next_action([entry], False, 0, KernelGenerator()) == action


def test_worker_timeout_keeps_its_partial_output_and_stage(tmp_path, monkeypatch):
    def run_process(command, *, cwd, timeout_s, **kwargs):
        raise _factory._providers.ProcessTimeoutError(
            timeout_s, "building engine\x1b[0m", "benchmarking candidate\x07"
        )

    monkeypatch.setattr(_factory._providers, "run_process", run_process)
    report = _factory._evaluate({}, tmp_path, 5)
    assert report == {
        "status": "error",
        "stage": "timeout",
        "error": "Evaluator exceeded its 5s timeout: benchmarking candidate",
    }
    log = (tmp_path / "worker.log").read_text()
    assert "building engine" in log and "benchmarking candidate" in log
    assert "\x1b" not in log and "\x07" not in log


_SECRET = "fake-secret-0123456789"
# What a candidate printing os.environ while it compiles writes to the log.
_PRINTED = f"ENV environ({{'REGION_TEST_API_KEY': '{_SECRET}'}})"
_REDACTED = "'REGION_TEST_API_KEY': '[redacted]'"
# The fixture scripts _evaluate; these tests run the real one.
_EVALUATE = _factory._evaluate


@pytest.mark.parametrize("outcome", ["timeout", "crash", "report"])
def test_worker_logs_and_reports_are_redacted(tmp_path, monkeypatch, outcome):
    monkeypatch.setenv("REGION_TEST_API_KEY", _SECRET)
    fitted = dict(_candidate("fitted"), source=f"# {_SECRET}\n")

    def run_process(command, *, cwd, timeout_s, **kwargs):
        if outcome == "timeout":
            raise _factory._providers.ProcessTimeoutError(timeout_s, _PRINTED, _PRINTED)
        if outcome == "report":
            report = {
                "status": "error",
                "stage": "compile",
                "error": _PRINTED,
                "fit_log": [_PRINTED],
                "fitted_candidate": fitted,
            }
            (cwd / "result.json").write_text(json.dumps(report))
        return subprocess.CompletedProcess(command, -11, _PRINTED, _PRINTED)

    monkeypatch.setattr(_factory._providers, "run_process", run_process)
    report = _EVALUATE({}, tmp_path, 5)
    log = (tmp_path / "worker.log").read_text()
    for text in (log, report["error"], *report.get("fit_log", ())):
        assert _SECRET not in text and _REDACTED in text
    if outcome == "report":
        # A fitted candidate is adopted only with the author's exact source.
        assert report["fitted_candidate"] == fitted


def test_credentials_that_a_candidate_prints_never_reach_the_author(
    factory, monkeypatch
):
    monkeypatch.setenv("REGION_TEST_API_KEY", _SECRET)

    def run_process(command, *, cwd, timeout_s, **kwargs):
        (cwd / "result.json").write_text(json.dumps(_passed(0.2)))
        return subprocess.CompletedProcess(command, 0, _PRINTED, "")

    monkeypatch.setattr(_factory._providers, "run_process", run_process)
    factory.evaluate.side_effect = _EVALUATE
    candidate = _candidate("printing")
    factory.author.side_effect = [
        candidate,
        {
            "action": "read",
            "candidates": None,
            "resource": "attempt-1/worker.log",
            "offset": 0,
        },
        {"action": "stop"},
    ]
    assert _generate(factory, max_attempts=2) == candidate
    run = next(factory.root.glob("*/run-*"))
    log = (run / "attempt-1" / "worker.log").read_text()
    page = json.loads((run / "attempt-2" / "evidence.json").read_text())["result"]
    prompt = factory.author.call_args_list[2].kwargs["prompt"]
    assert "EVIDENCE RESULT" in prompt
    for text in (log, page["text"], prompt):
        assert _SECRET not in text and _REDACTED in text


def _listed_evidence(prompt):
    listing = prompt.split('"catalog" for the full listing.\n', 1)[1]
    lines = listing.split("\nAUTHOR BUDGET", 1)[0].splitlines()
    return [line.split(" (")[0] for line in lines]


def test_prompts_list_recent_evidence_and_the_catalog_pages_all_of_it(factory):
    candidates = [_candidate(f"kernel_{index}") for index in range(8)]
    factory.author.side_effect = [
        *candidates[:7],
        {"action": "read", "candidates": None, "resource": "catalog", "offset": 0},
        candidates[7],
    ]
    factory.evaluate.side_effect = [
        {"status": "error", "stage": "correctness", "error": f"mismatch {index}"}
        for index in range(7)
    ] + [_passed(0.2)]
    assert _generate(factory, max_attempts=8) == candidates[7]
    prompts = [call.kwargs["prompt"] for call in factory.author.call_args_list]
    assert "candidate evaluations: 8; author rounds including this one: 8." in (
        prompts[0]
    )
    assert "candidate evaluations: 1; author rounds including this one: 1." in (
        prompts[7]
    )
    # Inlined instruction resources are readable but not listed again.
    assert _listed_evidence(prompts[0]) == ["task.json"]
    # Seven attempts registered 21 resources after the task; 20 are listed.
    listed = _listed_evidence(prompts[7])
    assert len(listed) == 20
    assert listed[0] == "attempt-1/candidate.json"
    assert listed[-1] == "attempt-7/source"
    page = json.loads(
        next(factory.root.glob("*/run-*/attempt-8/evidence.json")).read_text()
    )["result"]
    assert "SKILL.md (" in page["text"]
    assert "attempt-1/feedback.json (" in page["text"]
    assert "EVIDENCE RESULT" in prompts[8]


def test_author_stop_without_a_kernel_says_why(factory):
    factory.author.return_value = {"action": "stop"}
    with pytest.raises(
        RegionError,
        match=r"the author stopped \(0 of 4 evaluations used\); "
        "no candidate was evaluated",
    ):
        _generate(factory, max_attempts=4)


def test_cache_directories_are_private(tmp_path):
    from torch_tensorrt.dynamo.region_compilation._generation import _util

    task = _util.private_directory(_util.private_directory(tmp_path / "cache") / "task")
    for directory in (tmp_path / "cache", task):
        assert directory.stat().st_mode & 0o077 == 0, directory


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership")
def test_cache_directory_of_another_user_is_refused(monkeypatch, tmp_path):
    from torch_tensorrt.dynamo.region_compilation._generation import _util

    other = os.getuid() + 1
    monkeypatch.setattr(_util.os, "getuid", lambda: other)
    with pytest.raises(RegionError, match="belongs to another user"):
        _util.private_directory(tmp_path / "cache")


def test_writable_cache_directory_is_reported_once_and_left_unchanged(tmp_path, caplog):
    from torch_tensorrt.dynamo.region_compilation._generation import _util

    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o775)
    with caplog.at_level(logging.WARNING, logger=_util.__name__):
        for _ in range(3):
            _util.private_directory(shared)
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == _util.__name__
    ]
    assert len(warnings) == 1 and "writable by other users" in warnings[0]
    # Its owner may share it on purpose, so its permissions are left alone.
    assert stat.S_IMODE(shared.stat().st_mode) == 0o775


def test_generation_keeps_cache_task_and_run_directories_private(factory):
    factory.author.return_value = _candidate("published")
    factory.evaluate.return_value = _passed(0.2)
    # Under this umask, plain mkdir would leave the directories readable.
    umask = os.umask(0o022)
    try:
        _generate(factory)
    finally:
        os.umask(umask)
    (task,) = (path for path in factory.root.iterdir() if path.is_dir())
    (run,) = task.glob("run-*")
    for directory in (factory.root, task, run):
        assert directory.stat().st_mode & 0o077 == 0, directory


def test_evaluator_keeps_the_candidate_directory_off_sys_path(tmp_path, monkeypatch):
    # The worker runs in the candidate's directory, where a planted module must
    # not shadow one that the worker imports.
    commands = []

    def run_process(command, *, cwd, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(_factory._providers, "run_process", run_process)
    _factory._evaluate({}, tmp_path, 1)
    (command,) = commands
    (tmp_path / "planted.py").write_text("")
    # PYTHONSAFEPATH would keep the directory off sys.path even without the flag.
    environment = {
        name: value for name, value in os.environ.items() if name != "PYTHONSAFEPATH"
    }
    probe = subprocess.run(
        [*command[: command.index("-m")], "-c", "import planted"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert "ModuleNotFoundError: No module named 'planted'" in probe.stderr


def test_evaluator_tells_the_worker_its_time_limit_and_controller(
    tmp_path, monkeypatch
):
    # The worker ends itself if this process dies without killing it.
    calls = []

    def run_process(command, *, cwd, timeout_s, **kwargs):
        calls.append((command, timeout_s))
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(_factory._providers, "run_process", run_process)
    _factory._evaluate({}, tmp_path, 42.5)
    ((command, timeout),) = calls
    assert timeout == float(command[command.index("--timeout") + 1]) == 42.5
    assert command[command.index("--parent") + 1] == str(os.getpid())
