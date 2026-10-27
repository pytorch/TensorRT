# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Search queues and resume decisions must use bounded, current measurements."""

import json
from types import SimpleNamespace

import pytest

from torch_tensorrt.dynamo.region_compilation._generation import _factory
from torch_tensorrt.region import RegionError, SearchPolicy

from .test_kernel_generation import _candidate, _generate, _passed
from .test_kernel_generation import factory as factory  # noqa: F401


@pytest.fixture
def controller(request, monkeypatch):
    base = request.getfixturevalue("factory")

    def evaluate(request, directory, timeout):
        function = (
            base.finalize
            if request.get("validation_phase") == "holdout"
            else base.evaluate
        )
        return function(request, directory, timeout)

    monkeypatch.setattr(_factory, "_evaluate", evaluate)
    return base


def _batch(*candidates):
    return {"action": "propose", "candidates": list(candidates)}


def test_batch_tuning_is_interleaved_and_stops_at_the_candidate_cap(controller):
    first, second = _candidate("first"), _candidate("second")
    first["num_warps"] = second["num_warps"] = 16
    controller.author.return_value = _batch(first, second)
    controller.evaluate.side_effect = [_passed(0.4), _passed(0.3), _passed(0.2)]
    selected = _generate(
        controller,
        max_attempts=3,
        search=SearchPolicy(
            batch_size=2,
            max_candidates=3,
            max_variants=2,
            tuning={"num_warps": [4, 8]},
        ),
    )
    trials = [call.args[0]["candidate"] for call in controller.evaluate.call_args_list]
    assert [(item["kernel_name"], item["num_warps"]) for item in trials] == [
        ("first", 4),
        ("second", 4),
        ("first", 8),
    ]
    assert selected == dict(first, num_warps=8)
    controller.author.assert_called_once()
    controller.finalize.assert_called_once()
    search = json.loads(next(controller.root.glob("*/run-*/search.json")).read_text())
    assert search["evaluations"] == search["max_candidates"] == 3


def test_duplicate_tuned_variants_reuse_current_proof(controller):
    controller.author.return_value = _candidate("same")
    controller.evaluate.return_value = _passed(0.2)
    _generate(
        controller,
        search=SearchPolicy(max_candidates=2, tuning={"num_warps": [4, 4]}),
    )
    controller.evaluate.assert_called_once()
    history = json.loads(next(controller.root.glob("*/run-*/history.json")).read_text())
    assert "reused_from" in history[1]["result"]


def test_improving_correct_candidates_can_progress_toward_the_baseline_objective(
    controller,
):
    candidates = [_candidate(name) for name in ("slow", "improving", "winner")]
    controller.author.side_effect = candidates
    controller.evaluate.side_effect = [
        {**_passed(latency), "baseline_latency_ms": 1.0} for latency in (2.0, 1.5, 0.5)
    ]
    controller.finalize.return_value = {**_passed(0.4), "baseline_latency_ms": 1.0}
    selected = _generate(
        controller,
        objective="eager",
        min_speedup=1,
        max_attempts=3,
        search=SearchPolicy(max_no_improvement=1),
    )
    assert selected == candidates[-1]
    assert controller.author.call_count == controller.evaluate.call_count == 3
    for call in controller.author.call_args_list[1:]:
        assert "SEARCH STEP: optimize" in call.kwargs["prompt"]
    controller.finalize.assert_called_once()


def test_interrupted_batch_resumes_pending_and_remeasures_historical_winner(
    controller,
):
    historical, pending = _candidate("historical"), _candidate("pending")
    policy = SearchPolicy(batch_size=2, max_candidates=2, resume=True)
    controller.author.return_value = _batch(historical, pending)
    controller.evaluate.side_effect = [_passed(0.001), KeyboardInterrupt()]
    with pytest.raises(KeyboardInterrupt):
        _generate(controller, search=policy)
    checkpoint = next(controller.root.glob("*/checkpoint.json"))
    state = json.loads(checkpoint.read_text())["state"]
    assert state["finalists"] == [historical]
    assert state["pending"] == [pending]
    first_seed = controller.evaluate.call_args_list[0].args[0]["validation_seed"]

    controller.author.reset_mock()
    controller.evaluate.reset_mock()
    controller.evaluate.side_effect = [
        {"status": "error", "stage": "correctness", "error": "fresh check failed"},
        _passed(0.3),
    ]
    selected = _generate(controller, search=policy)
    assert selected == pending
    controller.author.assert_not_called()
    requests = [call.args[0] for call in controller.evaluate.call_args_list]
    assert [request["candidate"] for request in requests] == [historical, pending]
    assert requests[0]["validation_seed"] != first_seed
    assert requests[0]["validation_seed"] == requests[1]["validation_seed"]
    assert "incumbent" not in requests[1]
    assert controller.finalize.call_args.args[0]["candidate"] == pending
    accepted = json.loads(next(controller.root.glob("*/accepted.json")).read_text())
    assert accepted["report"]["latency_ms"] == 0.3
    history = json.loads(
        (
            next(controller.root.glob("*/run-*/resume.json")).parent / "history.json"
        ).read_text()
    )
    assert any(entry.get("historical") for entry in history)
    assert not checkpoint.exists()


def test_failed_resume_cannot_publish_historical_timing(controller):
    candidate = _candidate("interrupted_finalist")
    policy = SearchPolicy(max_candidates=1, resume=True)
    controller.author.return_value = candidate
    controller.evaluate.return_value = _passed(0.001)
    controller.finalize.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        _generate(controller, search=policy)
    controller.author.reset_mock()
    controller.finalize.reset_mock()
    controller.evaluate.return_value = {
        "status": "error",
        "stage": "correctness",
        "error": "does not reproduce",
    }
    with pytest.raises(RegionError, match="does not reproduce"):
        _generate(controller, search=policy)
    controller.author.assert_not_called()
    controller.finalize.assert_not_called()
    assert not list(controller.root.glob("*/accepted.json"))


def test_resume_can_be_enabled_after_an_unexpected_interruption(controller):
    candidate = _candidate("saved_by_default")
    controller.author.return_value = candidate
    controller.evaluate.return_value = _passed(0.2)
    controller.finalize.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        _generate(controller)
    checkpoint = next(controller.root.glob("*/checkpoint.json"))
    controller.author.reset_mock()
    controller.evaluate.reset_mock()
    controller.finalize.side_effect = None
    assert _generate(controller, search=SearchPolicy(resume=True)) == candidate
    controller.author.assert_not_called()
    controller.evaluate.assert_called_once()
    assert not checkpoint.exists()


def test_budget_failure_keeps_unmeasured_proposal_pending(controller):
    candidate = _candidate("waiting_for_gpu")
    controller.author.return_value = candidate
    controller.evaluate.return_value = {
        "status": "error",
        "stage": "budget",
        "error": "GPU evaluation lock timed out",
    }
    with pytest.raises(RegionError, match="GPU evaluation lock timed out"):
        _generate(controller, search=SearchPolicy(resume=True))
    state = json.loads(next(controller.root.glob("*/checkpoint.json")).read_text())[
        "state"
    ]
    assert state["pending"] == [candidate]
    assert state["finalists"] == []


def test_finalist_alternative_gets_a_fresh_holdout_and_reserved_time(
    controller, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(_factory, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    first, second = _candidate("fastest"), _candidate("alternative")
    controller.author.return_value = _batch(first, second)

    def evaluate(request, directory, timeout):
        clock[0] += 10
        return _passed(0.1 if request["candidate"] == first else 0.2)

    def holdout(request, directory, timeout):
        assert timeout == pytest.approx(35)
        clock[0] += timeout
        if request["candidate"] == first:
            return {
                "status": "error",
                "stage": "correctness",
                "error": "holdout mismatch",
            }
        return _passed(0.3)

    controller.evaluate.side_effect = evaluate
    controller.finalize.side_effect = holdout
    selected = _generate(
        controller,
        timeout_s=90,
        candidate_timeout_s=120,
        search=SearchPolicy(batch_size=2, finalists=2),
    )
    assert selected == second
    finals = controller.finalize.call_args_list
    assert [call.args[1].name for call in finals] == ["holdout", "holdout-2"]
    seeds = [call.args[0]["validation_seed"] for call in finals]
    assert seeds[0] != seeds[1]
    assert all(call.args[0]["validation_phase"] == "holdout" for call in finals)
    controller.register.assert_called_once()
    accepted = json.loads(next(controller.root.glob("*/accepted.json")).read_text())
    assert accepted["candidate"] == second
    assert accepted["report"]["latency_ms"] == 0.2
    assert accepted["validation"]["latency_ms"] == 0.3
    assert clock[0] == 90


def test_batch_author_logs_remain_available_as_next_attempt_evidence(controller):
    first, second = _candidate("first"), _candidate("second")

    def author(**kwargs):
        if controller.author.call_count == 1:
            (kwargs["workdir"] / "provider.log").write_text("compiler hint from author")
            return _batch(first, second)
        assert "attempt-1/provider.log" in kwargs["prompt"]
        return {"action": "stop"}

    controller.author.side_effect = author
    controller.evaluate.return_value = _passed(0.2)
    _generate(controller, max_attempts=2, search=SearchPolicy(batch_size=2))
    assert controller.author.call_count == 2
