# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Search queues and resume decisions must use bounded, current measurements."""

import json
import os
from types import SimpleNamespace

import pytest

from torch_tensorrt.dynamo.region_compilation._generation import _factory
from torch_tensorrt.region import RegionError, SearchPolicy

from ._generation_utils import _candidate, _generate, _passed

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Kernel generation requires a POSIX host"
)


def _batch(*candidates):
    return {"action": "propose", "candidates": list(candidates)}


def test_batch_tuning_is_interleaved_and_stops_at_the_candidate_cap(factory):
    first, second = _candidate("first"), _candidate("second")
    first["num_warps"] = second["num_warps"] = 16
    factory.author.return_value = _batch(first, second)
    factory.evaluate.side_effect = [_passed(0.4), _passed(0.3), _passed(0.2)]
    selected = _generate(
        factory,
        max_attempts=3,
        search=SearchPolicy(
            batch_size=2,
            max_candidates=3,
            max_variants=2,
            tuning={"num_warps": [4, 8]},
        ),
    )
    trials = [call.args[0]["candidate"] for call in factory.evaluate.call_args_list]
    assert [(item["kernel_name"], item["num_warps"]) for item in trials] == [
        ("first", 4),
        ("second", 4),
        ("first", 8),
    ]
    assert selected == dict(first, num_warps=8)
    factory.author.assert_called_once()
    factory.finalize.assert_called_once()
    search = json.loads(next(factory.root.glob("*/run-*/search.json")).read_text())
    assert search["evaluations"] == search["max_candidates"] == 3


def test_duplicate_tuned_variants_reuse_current_proof(factory):
    factory.author.return_value = _candidate("same")
    factory.evaluate.return_value = _passed(0.2)
    _generate(
        factory,
        search=SearchPolicy(max_candidates=2, tuning={"num_warps": [4, 4]}),
    )
    factory.evaluate.assert_called_once()
    history = json.loads(next(factory.root.glob("*/run-*/history.json")).read_text())
    assert "reused_from" in history[1]["result"]


def test_improving_correct_candidates_can_progress_toward_the_baseline_objective(
    factory,
):
    candidates = [_candidate(name) for name in ("slow", "improving", "winner")]
    factory.author.side_effect = candidates
    factory.evaluate.side_effect = [
        {**_passed(latency), "baseline_latency_ms": 1.0} for latency in (2.0, 1.5, 0.5)
    ]
    factory.finalize.return_value = {**_passed(0.4), "baseline_latency_ms": 1.0}
    selected = _generate(
        factory,
        objective="eager",
        min_speedup=1,
        max_attempts=3,
        search=SearchPolicy(max_no_improvement=1),
    )
    assert selected == candidates[-1]
    assert factory.author.call_count == factory.evaluate.call_count == 3
    for call in factory.author.call_args_list[1:]:
        assert "SEARCH STEP: optimize" in call.kwargs["prompt"]
    factory.finalize.assert_called_once()


def test_search_requests_carry_the_screening_policy(factory):
    factory.author.return_value = _candidate("screened")
    factory.evaluate.return_value = _passed(0.2)
    _generate(factory)
    assert factory.evaluate.call_args.args[0]["screening"] == "launch"
    _generate(factory, use_cache=False, search=SearchPolicy(screening="engine"))
    assert factory.evaluate.call_args.args[0]["screening"] == "engine"


def test_tensorrt_screens_build_the_baseline_until_a_search_gets_past_it(factory):
    factory.author.side_effect = [_candidate(name) for name in ("a", "b", "c")]
    screened = {**_passed(0.5), "execution": "launch"}
    factory.evaluate.side_effect = [
        {"status": "error", "stage": "source", "error": "rejected first"},
        screened,
        screened,
    ]
    factory.finalize.return_value = {**_passed(0.4), "baseline_latency_ms": 1.0}
    _generate(factory, objective="tensorrt", max_attempts=3)
    requests = [call.args[0] for call in factory.evaluate.call_args_list]
    # The worker checks the source before it builds the baseline.
    assert [request["check_baseline"] for request in requests] == [True, True, False]


def test_a_tensorrt_baseline_failure_ends_a_screened_search_at_once(factory):
    factory.author.return_value = _candidate("screened")
    factory.evaluate.return_value = {
        "status": "error",
        "stage": "baseline",
        "error": "TensorRT baseline contains eager computation",
    }
    with pytest.raises(RegionError, match="no proposal can fix"):
        _generate(factory, objective="tensorrt", max_attempts=4)
    factory.author.assert_called_once()
    request = factory.evaluate.call_args.args[0]
    assert request["screening"] == "launch" and request["check_baseline"] is True


@pytest.mark.parametrize(
    "screened_baseline, final_baseline, accepted",
    [(0.9, 2.0, True), (2.0, 1.0, False)],
    ids=["screen-slower-engine-faster", "screen-faster-engine-slower"],
)
def test_screened_speedups_are_decided_on_final_engine_measurements(
    factory, screened_baseline, final_baseline, accepted
):
    candidate = _candidate("screened")
    factory.author.return_value = candidate
    factory.evaluate.return_value = {
        **_passed(1.0),
        "execution": "launch",
        "baseline_latency_ms": screened_baseline,
    }
    factory.finalize.return_value = {
        **_passed(1.0),
        "execution": "engine",
        "baseline_latency_ms": final_baseline,
    }
    if accepted:
        assert _generate(factory, objective="eager", min_speedup=1.05) == candidate
    else:
        with pytest.raises(
            RegionError, match="final validation.*Performance objective"
        ):
            _generate(factory, objective="eager", min_speedup=1.05)
    factory.finalize.assert_called_once()
    run = next(factory.root.glob("*/run-*"))
    # The screen's timing neither rejects nor accepts the candidate.
    assert (
        json.loads((run / "attempt-1" / "feedback.json").read_text())["meets_objective"]
        is None
    )
    assert bool(list(factory.root.glob("*/accepted.json"))) == accepted


def test_interrupted_batch_resumes_pending_and_remeasures_historical_winner(
    factory,
):
    historical, pending = _candidate("historical"), _candidate("pending")
    policy = SearchPolicy(batch_size=2, max_candidates=2, resume=True)
    factory.author.return_value = _batch(historical, pending)
    factory.evaluate.side_effect = [_passed(0.001), KeyboardInterrupt()]
    with pytest.raises(KeyboardInterrupt):
        _generate(factory, search=policy)
    checkpoint = next(factory.root.glob("*/checkpoint.json"))
    state = json.loads(checkpoint.read_text())["state"]
    assert state["finalists"] == [historical]
    assert state["pending"] == [pending]
    first_seed = factory.evaluate.call_args_list[0].args[0]["validation_seed"]

    factory.author.reset_mock()
    factory.evaluate.reset_mock()
    factory.evaluate.side_effect = [
        {"status": "error", "stage": "correctness", "error": "fresh check failed"},
        _passed(0.3),
    ]
    selected = _generate(factory, search=policy)
    assert selected == pending
    factory.author.assert_not_called()
    requests = [call.args[0] for call in factory.evaluate.call_args_list]
    assert [request["candidate"] for request in requests] == [historical, pending]
    assert requests[0]["validation_seed"] != first_seed
    assert requests[0]["validation_seed"] == requests[1]["validation_seed"]
    assert "incumbent" not in requests[1]
    assert factory.finalize.call_args.args[0]["candidate"] == pending
    accepted = json.loads(next(factory.root.glob("*/accepted.json")).read_text())
    assert accepted["report"]["latency_ms"] == 0.3
    history = json.loads(
        (
            next(factory.root.glob("*/run-*/resume.json")).parent / "history.json"
        ).read_text()
    )
    assert any(entry.get("historical") for entry in history)
    assert not checkpoint.exists()


def test_failed_resume_cannot_publish_historical_timing(factory):
    candidate = _candidate("interrupted_finalist")
    policy = SearchPolicy(max_candidates=1, resume=True)
    factory.author.return_value = candidate
    factory.evaluate.return_value = _passed(0.001)
    factory.finalize.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        _generate(factory, search=policy)
    factory.author.reset_mock()
    factory.finalize.reset_mock()
    factory.evaluate.return_value = {
        "status": "error",
        "stage": "correctness",
        "error": "does not reproduce",
    }
    with pytest.raises(RegionError, match="does not reproduce"):
        _generate(factory, search=policy)
    factory.author.assert_not_called()
    factory.finalize.assert_not_called()
    assert not list(factory.root.glob("*/accepted.json"))


def test_resume_can_be_enabled_after_an_unexpected_interruption(factory):
    candidate = _candidate("saved_by_default")
    factory.author.return_value = candidate
    factory.evaluate.return_value = _passed(0.2)
    factory.finalize.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        _generate(factory)
    checkpoint = next(factory.root.glob("*/checkpoint.json"))
    factory.author.reset_mock()
    factory.evaluate.reset_mock()
    factory.finalize.side_effect = None
    assert _generate(factory, search=SearchPolicy(resume=True)) == candidate
    factory.author.assert_not_called()
    factory.evaluate.assert_called_once()
    assert not checkpoint.exists()


def test_budget_failure_keeps_unmeasured_proposal_pending(factory):
    candidate = _candidate("waiting_for_gpu")
    factory.author.return_value = candidate
    factory.evaluate.return_value = {
        "status": "error",
        "stage": "budget",
        "error": "GPU evaluation lock timed out",
    }
    with pytest.raises(RegionError, match="GPU evaluation lock timed out"):
        _generate(factory, search=SearchPolicy(resume=True))
    state = json.loads(next(factory.root.glob("*/checkpoint.json")).read_text())[
        "state"
    ]
    assert state["pending"] == [candidate]
    assert state["finalists"] == []


def test_finalists_never_shorten_the_winners_final_validation(factory, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(_factory, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    first, second = _candidate("fastest"), _candidate("alternative")
    factory.author.return_value = _batch(first, second)

    def evaluate(request, directory, timeout):
        clock[0] += 10
        return _passed(0.1 if request["candidate"] == first else 0.2)

    def holdout(request, directory, timeout):
        clock[0] += timeout
        if request["candidate"] == first:
            return {
                "status": "error",
                "stage": "correctness",
                "error": "holdout mismatch",
            }
        return _passed(0.3)

    factory.evaluate.side_effect = evaluate
    factory.finalize.side_effect = holdout
    selected = _generate(
        factory,
        timeout_s=90,
        candidate_timeout_s=40,
        search=SearchPolicy(batch_size=2, finalists=2),
    )
    assert selected == second
    finals = factory.finalize.call_args_list
    assert [call.args[1].name for call in finals] == ["holdout", "holdout-2"]
    # The winner gets a full candidate slice; the alternative gets what is left.
    assert [call.args[2] for call in finals] == [40, 30]
    seeds = [call.args[0]["validation_seed"] for call in finals]
    assert seeds[0] != seeds[1]
    assert all(call.args[0]["validation_phase"] == "holdout" for call in finals)
    factory.register.assert_called_once()
    accepted = json.loads(next(factory.root.glob("*/accepted.json")).read_text())
    assert accepted["candidate"] == second
    assert accepted["report"]["latency_ms"] == 0.2
    assert accepted["validation"]["latency_ms"] == 0.3
    assert clock[0] == 90


@pytest.mark.parametrize("finalists", [1, 2])
def test_final_validation_reserves_a_slice_per_finalist(
    factory, monkeypatch, finalists
):
    clock = [0.0]
    monkeypatch.setattr(_factory, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    names = iter(range(10))
    factory.author.side_effect = lambda **kwargs: _candidate(f"k{next(names)}")

    def evaluate(request, directory, timeout):
        assert timeout == 100
        clock[0] += timeout
        return _passed(1 / clock[0])

    factory.evaluate.side_effect = evaluate
    _generate(
        factory,
        timeout_s=600,
        candidate_timeout_s=100,
        max_attempts=10,
        search=SearchPolicy(finalists=finalists),
    )
    # Each finalist reserves one candidate slice out of the search.
    assert factory.evaluate.call_count == 6 - finalists
    assert factory.finalize.call_args.args[2] == 100


@pytest.mark.parametrize("execution", ["engine", "launch"])
@pytest.mark.parametrize(
    "failure, retained",
    [
        ({"status": "error", "stage": "timeout", "error": "slow"}, True),
        ({"status": "error", "stage": "budget", "error": "late"}, True),
        # After an engine screen the search built the same engine, so only a
        # flaky build fails here.
        ({"status": "error", "stage": "compile", "error": "flaky"}, "engine"),
        ({"status": "error", "stage": "engine", "error": "flaky"}, "engine"),
        ({"status": "error", "stage": "correctness", "error": "wrong"}, False),
        ({"status": "error", "stage": "crash", "error": "signal 11"}, False),
        ({**_passed(0.2), "validation_phase": "holdout"}, False),
    ],
    ids=["timeout", "budget", "compile", "engine", "correctness", "crash", "slow"],
)
def test_only_a_disproving_final_check_drops_the_finalist(
    factory, failure, retained, execution
):
    candidate = _candidate("finalist")
    factory.author.return_value = candidate
    factory.evaluate.return_value = {
        **_passed(1.0),
        "baseline_latency_ms": 2.0,
        "execution": execution,
    }
    factory.finalize.return_value = failure
    with pytest.raises(RegionError, match="final validation"):
        _generate(factory, objective="eager", search=SearchPolicy(resume=True))
    state = json.loads(next(factory.root.glob("*/checkpoint.json")).read_text())
    kept = retained is True or retained == execution
    assert state["state"]["finalists"] == ([candidate] if kept else [])


def test_evaluation_cut_short_by_the_deadline_stays_pending(factory, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(_factory, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    candidate = _candidate("needs_more_time")
    factory.author.return_value = candidate

    def evaluate(request, directory, timeout):
        clock[0] += timeout
        return {"status": "error", "stage": "timeout", "error": "Evaluator exceeded"}

    factory.evaluate.side_effect = evaluate
    with pytest.raises(RegionError, match="search time ran out"):
        _generate(
            factory,
            timeout_s=90,
            candidate_timeout_s=120,
            max_attempts=2,
            search=SearchPolicy(resume=True),
        )
    # The 60 s search slice was shorter than candidate_timeout_s.
    assert factory.evaluate.call_args.args[2] == 60
    history = json.loads(next(factory.root.glob("*/run-*/history.json")).read_text())
    assert [entry["result"]["stage"] for entry in history] == ["budget"]
    state = json.loads(next(factory.root.glob("*/checkpoint.json")).read_text())
    assert state["state"]["pending"] == [candidate]


def test_full_candidate_timeout_is_the_candidates_failure(factory):
    slow, fixed = _candidate("slow"), _candidate("fixed")
    factory.author.side_effect = [slow, fixed]
    factory.evaluate.side_effect = [
        {"status": "error", "stage": "timeout", "error": "Evaluator exceeded"},
        _passed(0.2),
    ]
    assert _generate(factory, max_attempts=2, candidate_timeout_s=10) == fixed
    assert factory.evaluate.call_args_list[0].args[2] == 10
    assert "SEARCH STEP: repair" in factory.author.call_args.kwargs["prompt"]


def test_short_slices_launch_neither_author_nor_worker(factory, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(_factory, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def author(**kwargs):
        clock[0] += 3.5
        return _candidate("late")

    factory.author.side_effect = author
    with pytest.raises(RegionError, match="search time ran out"):
        _generate(factory, timeout_s=6, max_attempts=4)
    factory.author.assert_called_once()
    factory.evaluate.assert_not_called()


def test_default_candidate_cap_counts_tuning_variants(factory):
    factory.author.side_effect = [_candidate("first"), _candidate("second")]
    factory.evaluate.side_effect = [_passed(latency) for latency in (4, 3, 2, 1)]
    _generate(
        factory,
        max_attempts=2,
        search=SearchPolicy(max_variants=2, tuning={"num_warps": (4, 8, 16)}),
    )
    assert factory.author.call_count == 2
    search = json.loads(next(factory.root.glob("*/run-*/search.json")).read_text())
    assert search["max_candidates"] == search["evaluations"] == 4


def test_next_step_follows_the_whole_round_not_the_last_variant(factory):
    good, bad = _candidate("good"), _candidate("bad")
    factory.author.side_effect = [_batch(good, bad), {"action": "stop"}]
    factory.evaluate.side_effect = [
        _passed(0.5),
        {"status": "error", "stage": "compile", "error": "ptxas failed"},
    ]
    _generate(factory, max_attempts=2, search=SearchPolicy(batch_size=2))
    assert "SEARCH STEP: optimize" in factory.author.call_args.kwargs["prompt"]


@pytest.mark.parametrize("stage", ["source", "compile", "engine", "correctness"])
def test_identical_deterministic_failure_reuses_its_report(factory, stage):
    factory.author.return_value = _candidate("broken")
    factory.evaluate.return_value = {
        "status": "error",
        "stage": stage,
        "error": "deterministic failure",
    }
    with pytest.raises(RegionError, match="deterministic failure"):
        _generate(factory, max_attempts=3)
    assert factory.author.call_count == 3
    factory.evaluate.assert_called_once()
    run = next(factory.root.glob("*/run-*"))
    history = json.loads((run / "history.json").read_text())
    first = "attempt-1/feedback.json"
    assert [entry["result"].get("reused_from") for entry in history] == [
        None,
        first,
        first,
    ]
    assert json.loads((run / "search.json").read_text())["evaluations"] == 1


def test_author_timeout_on_a_shortened_slice_ends_the_search(factory):
    # The search deadline leaves the author less than author_timeout_s; timing
    # out there means the remaining rounds would only get shorter slices.
    factory.author.side_effect = _factory._providers.ProcessTimeoutError(30, "", "")
    with pytest.raises(RegionError, match="search time ran out"):
        _generate(
            factory,
            max_attempts=4,
            timeout_s=100,
            candidate_timeout_s=40,
            author_timeout_s=60,
        )
    factory.author.assert_called_once()
    run = next(factory.root.glob("*/run-*"))
    history = json.loads((run / "history.json").read_text())
    assert [entry["result"]["stage"] for entry in history] == ["budget"]
    assert json.loads((run / "search.json").read_text())["stop_reason"] == "budget"


def test_proposals_past_the_candidate_cap_are_recorded_and_kept_pending(factory):
    measured, unmeasured = _candidate("measured"), _candidate("unmeasured")
    factory.author.return_value = _batch(measured, measured, unmeasured)
    factory.evaluate.return_value = _passed(0.2)
    _generate(factory, search=SearchPolicy(batch_size=3, max_candidates=1))
    factory.evaluate.assert_called_once()
    run = next(factory.root.glob("*/run-*"))
    history = json.loads((run / "history.json").read_text())
    # A repeat reuses its measurement for free; the next proposal is refused.
    assert [
        (entry["candidate"]["kernel_name"], entry["result"].get("stage"))
        for entry in history
    ] == [("measured", None), ("measured", None), ("unmeasured", "budget")]
    assert "reused_from" in history[1]["result"]
    assert json.loads((run / "search.json").read_text())["stop_reason"] == "budget"


def test_incumbent_timing_failure_keeps_every_measured_contender(factory):
    fast, slower = _candidate("fast"), _candidate("slower")
    factory.author.side_effect = [fast, slower]
    factory.evaluate.side_effect = [
        _passed(0.1),
        {
            **_passed(0.3),
            "incumbent_error": "compile: incumbent build ran out of memory",
        },
    ]
    factory.finalize.side_effect = [
        {"status": "error", "stage": "correctness", "error": "fresh inputs disagree"},
        _passed(0.3),
    ]
    # A failed incumbent arm in another worker says nothing about the incumbent's
    # own result: it stays the winner, and the candidate stays an alternative.
    assert _generate(factory, max_attempts=2, search=SearchPolicy(finalists=2)) == (
        slower
    )
    assert factory.evaluate.call_args.args[0]["incumbent"] == fast
    assert [call.args[0]["candidate"] for call in factory.finalize.call_args_list] == [
        fast,
        slower,
    ]


def test_no_kernel_error_reports_this_invocation_not_resumed_history(factory):
    factory.author.return_value = _candidate("old")
    factory.evaluate.return_value = {
        "status": "error",
        "stage": "correctness",
        "error": "old failure",
    }
    policy = SearchPolicy(resume=True)
    with pytest.raises(RegionError, match="old failure"):
        _generate(factory, search=policy)
    factory.author.return_value = {"action": "stop"}
    with pytest.raises(RegionError) as failure:
        _generate(factory, search=policy)
    message = str(failure.value)
    assert "the author stopped (0 of 1 evaluations used)" in message
    assert "no candidate was evaluated" in message
    assert "old failure" not in message


def test_batch_author_logs_remain_available_as_next_attempt_evidence(factory):
    first, second = _candidate("first"), _candidate("second")

    def author(**kwargs):
        if factory.author.call_count == 1:
            (kwargs["workdir"] / "provider.log").write_text("compiler hint from author")
            return _batch(first, second)
        assert "attempt-1/provider.log" in kwargs["prompt"]
        return {"action": "stop"}

    factory.author.side_effect = author
    factory.evaluate.return_value = _passed(0.2)
    _generate(factory, max_attempts=2, search=SearchPolicy(batch_size=2))
    assert factory.author.call_count == 2
