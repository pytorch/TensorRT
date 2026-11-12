# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Tuning sessions on the fake CPU workload, with worker jobs run in this process.

Decisions, stops and resumes; the noise and head-to-head runs; code edits;
failures and their causes; planner failures; prepared kernels; exports; and
the pinned workload and environment.
"""

import json
import logging
import os
import shutil
from pathlib import Path

import pytest
from torch_tensorrt.autotune import (
    Budget,
    Goal,
    Limits,
    NoiseControl,
    Planner,
    Proposal,
    RuleBasedPlanner,
    TuningError,
    TuningSession,
    WorkloadFactory,
)
from torch_tensorrt.autotune import __main__ as cli
from torch_tensorrt.autotune import (
    _worker,
)

FACTORY = f"{Path(__file__).with_name('_workloads.py')}:create_fake"
# The fake workload sleeps, so its samples spread widely on a loaded machine;
# these tests script decisions, not stability, which has its own tests below.
SINGLE_RUN = NoiseControl(
    noise_runs=0, head_to_head_pairs=0, quiet_gpu_s=0, max_spread=10
)


@pytest.fixture
def jobs(monkeypatch):
    """Run worker jobs in this process; ``jobs`` lists ``(directory, request)`` of each.

    A test can change what the jobs report:

    * ``jobs.slow_best_in_final_check``: the best config looks 3x slower in
      the final check;
    * ``jobs.slow_head_to_head_candidates``: the candidate looks 3x slower in
      head-to-head runs;
    * ``jobs.inaccurate_export``: the export's rebuild misses the accuracy
      bounds;
    * ``jobs.export_times_out``: the export worker times out;
    * ``jobs.unstable``: the job directories whose measurement is unstable;
    * ``jobs.source``: the Torch-TensorRT source provenance, fixed so that
      edits to the checkout while a test runs do not count as a changed
      environment;
    * ``jobs.memory``: when set, the GPU memory reading of every measurement
      (``None`` for unknown), so a test does not depend on whether this
      process can see a GPU.
    """

    class Jobs(list):
        slow_best_in_final_check = False
        slow_head_to_head_candidates = False
        inaccurate_export = False
        export_times_out = False
        unstable: set = set()
        source: dict = {"commit": "0" * 40, "uncommitted_sha256": None}
        memory: object = ...  # ... keeps what the job measured

    recorded = Jobs()
    monkeypatch.setattr(
        _worker, "source_provenance", lambda package: dict(recorded.source)
    )

    def execute(self, request, directory):
        name = directory.relative_to(self.directory).as_posix()
        recorded.append((name, request))
        try:
            if recorded.export_times_out and name == "export":
                # What the session writes for a worker it killed.
                result = {
                    "status": "failed",
                    "stage": "timeout",
                    "error": "The worker exceeded trial_timeout_s=1200",
                    "exit_status": -9,
                    "signal": "SIGKILL",
                    "synthesized": True,
                }
            else:
                result = _worker.run_job(request)
        except Exception as error:  # noqa: BLE001 - like the worker's own handler
            result = {"status": "failed", "stage": "setup", "error": str(error)}
        slow = (
            recorded.slow_best_in_final_check
            and directory.parent.name == "final_check"
            and directory.name.endswith("best")
        ) or (
            recorded.slow_head_to_head_candidates
            and directory.parent.name == "head_to_head"
            and directory.name.endswith("candidate")
        )
        if slow and result.get("status") == "ok":
            result["latency_ms"] = {k: v * 3 for k, v in result["latency_ms"].items()}
            result["throughput"] /= 3
        if (
            recorded.inaccurate_export
            and name == "export"
            and result.get("status") == "ok"
        ):
            result["accuracy"] = {**result["accuracy"], "max_abs_error": 0.5}
        if recorded.memory is not ... and "memory" in result:
            result["memory"] = {
                "process_gpu_mb": recorded.memory,
                "torch_peak_mb": recorded.memory,
            }
        if name in recorded.unstable and result.get("status") == "ok":
            result["timing_stability"] = {
                "attempts": 3,
                "spreads": [0.2, 0.2, 0.2],
                "gpu_wait_s": 0.0,
                "gpu_quiet": False,
                "stable": False,
            }
        _worker.write_json(Path(request["result_path"]), result)
        return result

    monkeypatch.setattr(TuningSession, "_run_worker_process", execute)
    return recorded


def session(directory, kwargs=None, factory=FACTORY, **options):
    """A session of the fake workload made with ``kwargs``, with quick test settings."""
    # The fake workload sleeps, so its timings jitter by a few percent on a
    # busy machine; scripted speedups are 2x or more, noise never 20%.
    options.setdefault("goal", Goal(min_improvement=0.2))
    options.setdefault("budget", Budget(max_trials=3))
    options.setdefault("final_check_pairs", 1)
    # These tests script the worker sequence one run per trial; noise runs and
    # head-to-head runs are covered on their own below.
    options.setdefault("noise_control", SINGLE_RUN)
    return TuningSession(
        WorkloadFactory(factory, kwargs or {}), directory=directory, **options
    )


def decisions(directory):
    """The decision of every trial recorded in ``directory``, in order."""
    return [
        json.loads(path.read_text())["decision"]
        for path in sorted(Path(directory).glob("trials/*/decision.json"))
    ]


def test_session_accepts_measured_gains_and_keeps_the_rest(tmp_path, jobs):
    result = session(
        tmp_path / "s", {"speedups": {"runtime.cudagraphs=True": 0.5}}
    ).run()
    # precision (bottleneck hint) first, then runtime: only the CUDA-graph change pays.
    assert decisions(tmp_path / "s") == ["baseline", "rejected", "accepted", "rejected"]
    assert result.decision == "accepted"
    assert result.config == {"runtime.cudagraphs": True}
    assert result.winner["trial"] == "0002" and result.baseline["trial"] == "0000"
    assert result.stop_reason == "trial budget of 3 used"
    assert result.final_check["passed"]
    assert result.final_check["median_gain"] == pytest.approx(0.5, abs=0.1)
    assert sorted(path.name for path in result.artifact.iterdir()) == [
        "config.json",
        "fake.txt",
    ]
    assert json.loads((result.artifact / "config.json").read_text()) == result.config
    report = result.report()
    for section in ("## Before / after", "## Config", "## Trials", "## Profiles"):
        assert section in report
    assert "accepted trial 0002" in report and "Winner (0002)" in report
    summary = json.loads((tmp_path / "s" / "report.json").read_text())
    assert summary["winner"]["trial"] == "0002"
    assert [name for name, _ in jobs] == [
        "reference",
        "trials/0000",
        "trials/0001",
        "trials/0002",
        "trials/0003",
        "final_check/0-baseline",
        "final_check/0-best",
        "export",
    ]
    context = json.loads(
        (tmp_path / "s" / "trials" / "0003" / "context.json").read_text()
    )
    assert context["best"]["trial"] == "0002"
    assert [entry["decision"] for entry in context["history"]] == [
        "rejected",
        "accepted",
    ]


def test_inaccurate_and_broken_candidates_are_rejected_with_reasons(tmp_path, jobs):
    session(
        tmp_path / "s",
        {
            "speedups": {"engine.precision=fp16": 0.3, "runtime.cudagraphs=True": 0.4},
            "inaccurate": ["engine.precision=fp16"],
            "broken": ["runtime.cudagraphs=True"],
        },
        budget=Budget(max_trials=2),
    ).run()
    first = json.loads((tmp_path / "s/trials/0001/decision.json").read_text())
    second = json.loads((tmp_path / "s/trials/0002/decision.json").read_text())
    result = json.loads((tmp_path / "s/trials/0002/result.json").read_text())
    assert first["decision"] == "rejected"
    assert first["reason"].startswith("accuracy: max_abs_error=0.5")
    assert second == {
        "decision": "rejected",
        "reason": "build failed: RuntimeError: cannot build ['runtime.cudagraphs=True']",
    }
    assert result["status"] == "failed" and result["stage"] == "build"


def test_session_stops_at_the_target(tmp_path, jobs):
    result = session(
        tmp_path / "s",
        {"speedups": {"engine.precision=fp16": 0.25}},
        goal=Goal(target=1.0, min_improvement=0.2),
        budget=Budget(max_trials=5),
    ).run()
    assert result.stop_reason == "goal target reached"
    assert decisions(tmp_path / "s") == ["baseline", "accepted"]


def test_session_stops_after_patience_trials_without_an_accepted_candidate(
    tmp_path, jobs
):
    result = session(tmp_path / "s", budget=Budget(max_trials=5, patience=2)).run()
    assert result.stop_reason == "no accepted candidate in the last 2 trials"
    assert result.decision == "baseline kept"
    assert result.final_check is None
    assert decisions(tmp_path / "s") == ["baseline", "rejected", "rejected"]
    assert "baseline kept (no candidate was accepted)" in result.report()


class ScriptedPlanner(Planner):
    """Return the given proposals in turn, then stop; keep the contexts it saw."""

    name = "scripted"

    def __init__(self, proposals):
        self.proposals = list(proposals)
        self.contexts = []

    def propose(self, context, workdir):
        self.contexts.append(context)
        if not self.proposals:
            return Proposal(stop=True, rationale="done")
        return self.proposals.pop(0)


def test_invalid_proposals_are_recorded_and_end_the_search(tmp_path, jobs):
    planner = ScriptedPlanner(
        [
            Proposal(changes={"engine.turbo": True}),
            Proposal(changes={"engine.optimization_level": 4}),
            Proposal(changes={"workload.variant": "a"}),
        ]
    )
    result = session(tmp_path / "s", planner=planner, budget=Budget(max_trials=5)).run()
    assert result.stop_reason == "the planner made 3 invalid proposals in a row"
    reasons = [
        json.loads(path.read_text())["reason"]
        for path in sorted((tmp_path / "s/trials").glob("*/decision.json"))[1:]
    ]
    assert reasons[0].startswith("proposal rejected: Unknown knob 'engine.turbo'")
    assert reasons[1].startswith(
        "proposal rejected: Knob engine.optimization_level does not allow 4"
    )
    assert reasons[2].startswith("proposal rejected: this config was already measured")
    # Invalid proposals spend no measurement.
    assert [name for name, _ in jobs if name.startswith("trials/")] == ["trials/0000"]


def test_a_best_config_that_fails_the_final_check_keeps_the_baseline(tmp_path, jobs):
    jobs.slow_best_in_final_check = True
    result = session(
        tmp_path / "s",
        {"speedups": {"engine.precision=fp16": 0.5}},
        budget=Budget(max_trials=1),
    ).run()
    assert decisions(tmp_path / "s") == ["baseline", "accepted"]
    assert not result.final_check["passed"]
    assert result.decision == "baseline kept"
    # The best config found (trial 0001) lost, so the baseline is the winner.
    assert result.config == {} and result.winner["trial"] == "0000"
    assert "did NOT pass" in result.report()


def test_resume_reuses_recorded_trials_and_extends_the_budget(tmp_path, jobs):
    kwargs = {"speedups": {"engine.precision=fp16": 0.5}}
    session(tmp_path / "s", kwargs, budget=Budget(max_trials=1)).run()
    first = [name for name, _ in jobs]
    jobs.clear()
    result = session(tmp_path / "s", kwargs, budget=Budget(max_trials=3)).run()
    # The workload is described again, but recorded trials are not re-measured.
    assert first[:2] == ["reference", "trials/0000"]
    assert [name for name, _ in jobs][:3] == ["describe", "trials/0002", "trials/0003"]
    assert decisions(tmp_path / "s") == ["baseline", "accepted", "rejected", "rejected"]
    assert result.config == {"engine.precision": "fp16"}
    resumed = TuningSession.resume(tmp_path / "s")
    assert resumed.workload_factory.kwargs == kwargs
    assert resumed.budget == Budget(max_trials=3)


def test_an_interrupted_trial_is_measured_again(tmp_path, jobs):
    session(tmp_path / "s", budget=Budget(max_trials=1)).run()
    (tmp_path / "s/trials/0001/decision.json").unlink()
    jobs.clear()
    session(tmp_path / "s", budget=Budget(max_trials=1)).run()
    assert [name for name, _ in jobs][:2] == ["describe", "trials/0001"]


def test_a_directory_holds_one_session(tmp_path, jobs):
    session(tmp_path / "s", budget=Budget(max_trials=0)).run()
    with pytest.raises(TuningError, match="differs in goal"):
        session(tmp_path / "s", goal=Goal(min_improvement=0.1)).run()
    with pytest.raises(TuningError, match="differs in workload_factory"):
        session(tmp_path / "s", {"milliseconds": 1.0}).run()


def test_invalid_baselines_and_references_end_the_session(tmp_path, jobs):
    with pytest.raises(TuningError, match="baseline is not valid.*build failed"):
        session(
            tmp_path / "a",
            {
                "broken": ["workload.variant=b"],
                "baseline_changes": {"workload.variant": "b"},
            },
        ).run()
    with pytest.raises(TuningError, match="no reference today"):
        session(tmp_path / "b", {"reference_fails": True}).run()
    with pytest.raises(TuningError, match="accuracy"):
        session(
            tmp_path / "c",
            {"inaccurate": ["workload.variant=c"]},
            baseline={"workload.variant": "c"},
        ).run()
    # Memory above the limit fails it, and so does memory that is unknown (no
    # GPU, or no reading): set both readings, whatever this machine has.
    jobs.memory = 64.0
    with pytest.raises(TuningError, match="memory 64.0 MB exceeds max_memory_mb=1"):
        session(tmp_path / "d", limits=Limits(max_memory_mb=1.0)).run()
    jobs.memory = None
    with pytest.raises(TuningError, match="memory None MB exceeds max_memory_mb=1"):
        session(tmp_path / "e", limits=Limits(max_memory_mb=1.0)).run()


def test_check_result_needs_a_gain_beyond_the_quartiles(tmp_path):
    check = session(tmp_path / "s", goal=Goal()).check_result
    bounds = {"top1": (">=", 0.9)}

    def result(median, spread, top1=1.0):
        latency = {"median": median, "p25": median - spread, "p75": median + spread}
        return {
            "status": "ok",
            "accuracy": {"top1": top1},
            "accuracy_after_timing": {"top1": top1},
            "latency_ms": latency,
            "throughput": 1000 / median,
            "items": 1,
        }

    best = result(10.0, 0.1)
    assert check(result(9.0, 0.1), best, bounds)[0]
    # A 10% gain with overlapping quartiles is noise.
    assert not check(result(9.0, 1.5), best, bounds)[0]
    # Separated, but below min_improvement (2%).
    assert not check(result(9.9, 0.01), result(10.0, 0.01), bounds)[0]
    accepted, reason = check(result(5.0, 0.1, top1=0.5), best, bounds)
    assert (
        not accepted
        and reason
        == "accuracy: top1=0.5 (needs >= 0.9); after timing: top1=0.5 (needs >= 0.9)"
    )


def test_status_shows_a_failed_baseline_being_measured_again_as_failed(
    tmp_path, jobs, capsys
):
    with pytest.raises(TuningError, match="baseline is not valid"):
        session(tmp_path / "s", {"broken": ["workload.variant=a"]}).run()
    # A resume that measures the baseline again first removes its result;
    # its old decision stands until the new measurement is decided.
    (tmp_path / "s/trials/0000/result.json").unlink()
    capsys.readouterr()
    assert cli.main(["status", str(tmp_path / "s")]) == 0
    assert "0000   failed" in capsys.readouterr().out


def with_noise_runs(directory, kwargs=None, **options):
    """A session with two noise runs and two head-to-head pairs per candidate."""
    options.setdefault(
        "noise_control",
        NoiseControl(noise_runs=2, head_to_head_pairs=2, quiet_gpu_s=0, max_spread=10),
    )
    return session(directory, kwargs, **options)


def test_noise_runs_set_the_noise_threshold_and_head_to_head_runs_check_gains(
    tmp_path, jobs
):
    result = with_noise_runs(
        tmp_path / "s",
        {"speedups": {"runtime.cudagraphs=True": 0.5}},
        budget=Budget(max_trials=2),
    ).run()
    assert [name for name, _ in jobs] == [
        "reference",
        "trials/0000",
        "noise/0",
        "noise/1",
        "trials/0001",
        "trials/0002",
        # Interleaved pairs, alternating which side runs first.
        "trials/0002/head_to_head/0-best",
        "trials/0002/head_to_head/0-candidate",
        "trials/0002/head_to_head/1-candidate",
        "trials/0002/head_to_head/1-best",
        "final_check/0-baseline",
        "final_check/0-best",
        "export",
    ]
    noise = json.loads((tmp_path / "s" / "noise" / "summary.json").read_text())
    assert len(noise["metrics"]) == 3
    # The fake workload sleeps, so its spread depends on the machine's load;
    # the noise threshold is the goal's 20% or twice that spread, whichever is
    # larger.
    spread = noise["spread"]
    assert spread == pytest.approx(
        (max(noise["metrics"]) - min(noise["metrics"])) / sorted(noise["metrics"])[1]
    )
    assert noise["noise_threshold"] == pytest.approx(max(0.2, 2 * spread))
    assert result.noise["noise_threshold"] == noise["noise_threshold"]
    decision = json.loads(
        (tmp_path / "s" / "trials" / "0002" / "decision.json").read_text()
    )
    assert decision["decision"] == "accepted"
    assert decision["reason"].startswith("won head-to-head: median paired gain")
    assert len(decision["head_to_head"]["gains"]) == 2
    assert min(decision["head_to_head"]["gains"]) > 0.3
    assert result.decision == "accepted"
    table = (tmp_path / "s" / "trials.tsv").read_text().splitlines()
    assert table[0].split("\t")[:4] == ["trial", "parent", "decision", "metric"]
    assert [row.split("\t")[2] for row in table[1:]] == [
        "baseline",
        "rejected",
        "accepted",
    ]
    paired_gain = float(table[3].split("\t")[4].rstrip("%")) / 100
    assert paired_gain == pytest.approx(0.5, abs=0.1)


def test_head_to_head_rejects_a_gain_that_does_not_hold_in_paired_runs(tmp_path, jobs):
    jobs.slow_head_to_head_candidates = True
    result = with_noise_runs(
        tmp_path / "s",
        {"speedups": {"runtime.cudagraphs=True": 0.5}},
        budget=Budget(max_trials=2),
    ).run()
    decision = json.loads(
        (tmp_path / "s" / "trials" / "0002" / "decision.json").read_text()
    )
    assert decision["decision"] == "rejected"
    assert decision["reason"].startswith("lost head-to-head")
    assert decision["first_measurement"].startswith("latency_ms improved")
    assert result.decision == "baseline kept"


def test_unstable_measurements_are_never_accepted(tmp_path, jobs):
    jobs.unstable = {"trials/0002"}
    with_noise_runs(
        tmp_path / "s",
        {"speedups": {"runtime.cudagraphs=True": 0.5}},
        budget=Budget(max_trials=2),
    ).run()
    decision = json.loads(
        (tmp_path / "s" / "trials" / "0002" / "decision.json").read_text()
    )
    assert decision["decision"] == "rejected"
    assert decision["reason"].startswith("unstable measurement: interquartile spread")
    assert "the GPU was busy" in decision["reason"]


def test_resume_reuses_the_noise_runs(tmp_path, jobs):
    with_noise_runs(tmp_path / "s", budget=Budget(max_trials=1)).run()
    first = [name for name, _ in jobs]
    assert first.count("noise/0") == 1
    jobs.clear()
    with_noise_runs(tmp_path / "s", budget=Budget(max_trials=2)).run()
    assert not any(name.startswith("noise/") for name, _ in jobs)


def test_the_worker_times_again_when_the_samples_are_unstable():
    from torch_tensorrt.autotune._knobs import knobs_by_name, resolve
    from torch_tensorrt.autotune._measure import measure_candidate
    from torch_tensorrt.autotune._settings import Timing

    from ._workloads import FAKE_KNOBS, FakeWorkload

    values = resolve({}, knobs_by_name(FAKE_KNOBS))
    timing = Timing(warmup=0, rounds=5, calls=1, profile_calls=0)
    settings = {"quiet_gpu_s": 0, "max_spread": 0.1}
    # The accuracy check takes call 0; calls 1-5 (the first attempt) wobble.
    # 20 ms calls keep sleep jitter on a busy machine well inside the 10% limit.
    workload = FakeWorkload(milliseconds=20.0, wobble_calls=6)
    timed_again = measure_candidate(
        workload,
        values,
        workload.reference(),
        timing,
        noise_control={**settings, "retries": 2},
    )
    assert timed_again["timing_stability"]["attempts"] == 2
    assert timed_again["timing_stability"]["stable"]
    assert timed_again["timing_stability"]["spreads"][0] > 0.1
    workload = FakeWorkload(milliseconds=20.0, wobble_calls=6)
    once = measure_candidate(
        workload,
        values,
        workload.reference(),
        timing,
        noise_control={**settings, "retries": 0},
    )
    assert once["timing_stability"]["attempts"] == 1
    assert not once["timing_stability"]["stable"]


def test_file_edits_are_checked_stored_measured_and_accepted(tmp_path, jobs):
    planner = ScriptedPlanner(
        [
            Proposal(
                edits={"speed": [{"search": "SPEED = 1.0", "replace": "SPEED = 0.4"}]},
                rationale="halve the time",
            ),
            # Outside the fence: refused before anything runs.
            Proposal(
                edits={"speed": [{"search": "fenced factor", "replace": "factor"}]}
            ),
            Proposal(edits={"nope": [{"search": "x", "replace": "y"}]}),
            Proposal(
                edits={"speed": [{"search": "SPEED = 0.4", "replace": "SPEED = ("}]}
            ),
        ]
    )
    result = session(
        tmp_path / "s",
        {"editable_file": True},
        planner=planner,
        budget=Budget(max_trials=4),
        guidance="Try code before knobs.",
    ).run()
    root = tmp_path / "s"
    assert decisions(root) == ["baseline", "accepted", "invalid", "invalid", "invalid"]
    reasons = [
        json.loads((root / "trials" / f"{n:04d}" / "decision.json").read_text())[
            "reason"
        ]
        for n in (2, 3, 4)
    ]
    assert "not inside an EDITABLE region" in reasons[0]
    assert "Unknown editable file 'nope'" in reasons[1]
    assert "does not compile" in reasons[2]
    sha = result.config["edit.speed"]
    assert (root / "edited_files" / f"{sha}.py").read_text().count("SPEED = 0.4") == 1
    assert result.decision == "accepted"
    assert result.final_check["median_gain"] == pytest.approx(0.6, abs=0.1)
    # The planner saw the editable file, then the best config's edited version,
    # and the guidance.
    first, second = planner.contexts[0], planner.contexts[1]
    assert first["editable_files"][0]["editable_blocks"] == ["SPEED = 1.0\n"]
    assert second["editable_files"][0]["editable_blocks"] == ["SPEED = 0.4\n"]
    assert second["editable_files"][0]["edited"] == sha
    assert first["guidance"] == "Try code before knobs."


def test_a_retry_counts_once_the_gpu_went_quiet(monkeypatch):
    from torch_tensorrt.autotune import _measure
    from torch_tensorrt.autotune._knobs import knobs_by_name, resolve
    from torch_tensorrt.autotune._settings import Timing

    from ._workloads import FAKE_KNOBS, FakeWorkload

    gates = []

    def gate(max_wait_s):
        gates.append(max_wait_s)
        if len(gates) == 1:
            return {"waited_s": 7.0, "quiet": False}
        return {"waited_s": 1.0, "quiet": True}

    monkeypatch.setattr(_measure, "wait_for_quiet_gpu", gate)
    values = resolve({}, knobs_by_name(FAKE_KNOBS))
    timing = Timing(warmup=0, rounds=5, calls=1, profile_calls=0)
    settings = {"quiet_gpu_s": 10, "max_spread": 10}
    workload = FakeWorkload(milliseconds=2.0)
    result = _measure.measure_candidate(
        workload,
        values,
        workload.reference(),
        timing,
        noise_control={**settings, "retries": 2},
    )
    stability = result["timing_stability"]
    assert stability["attempts"] == 2
    assert stability["gpu_quiet"] is True and stability["stable"]
    # quiet_gpu_s bounds the whole wait; once spent, an attempt still looks briefly.
    assert gates == [10, 3.0]
    gates.clear()
    workload = FakeWorkload(milliseconds=2.0)
    once = _measure.measure_candidate(
        workload,
        values,
        workload.reference(),
        timing,
        noise_control={**settings, "retries": 0},
    )
    assert once["timing_stability"]["gpu_quiet"] is False
    assert not once["timing_stability"]["stable"]


def test_noise_runs_are_redone_when_their_settings_change(tmp_path, jobs):
    session(tmp_path / "s", budget=Budget(max_trials=1)).run()
    assert not any(name.startswith("noise/") for name, _ in jobs)
    jobs.clear()
    with_noise_runs(tmp_path / "s", budget=Budget(max_trials=1)).run()
    names = [name for name, _ in jobs]
    assert names.count("noise/0") == names.count("noise/1") == 1
    summary = json.loads((tmp_path / "s/noise/summary.json").read_text())
    assert summary["requested"] == 2 and summary["spread"] is not None


def test_edit_changes_must_name_a_declared_file_and_a_stored_edited_file(
    tmp_path, jobs
):
    planner = ScriptedPlanner(
        [
            Proposal(changes={"edit.speed": "0" * 64}),
            Proposal(changes={"edit.nope": ""}),
        ]
    )
    session(tmp_path / "s", {"editable_file": True}, planner=planner).run()
    root = tmp_path / "s"
    assert decisions(root) == ["baseline", "invalid", "invalid"]
    reasons = [
        json.loads((root / "trials" / f"{n:04d}" / "decision.json").read_text())[
            "reason"
        ]
        for n in (1, 2)
    ]
    assert "names no stored edited file" in reasons[0]
    assert "Unknown editable file 'edit.nope'" in reasons[1]


def test_edits_that_restore_the_original_select_the_original(tmp_path, jobs):
    planner = ScriptedPlanner(
        [
            Proposal(
                edits={"speed": [{"search": "SPEED = 1.0", "replace": "SPEED = 0.4"}]}
            ),
            Proposal(
                edits={"speed": [{"search": "SPEED = 0.4", "replace": "SPEED = 1.0"}]}
            ),
        ]
    )
    session(tmp_path / "s", {"editable_file": True}, planner=planner).run()
    root = tmp_path / "s"
    assert decisions(root) == ["baseline", "accepted", "invalid"]
    reason = json.loads((root / "trials/0002/decision.json").read_text())["reason"]
    assert "already measured" in reason


def test_a_resumed_session_forgets_earlier_invalid_proposals(tmp_path, jobs):
    bad = [Proposal(changes={"engine.turbo": True}) for _ in range(3)]
    first = session(tmp_path / "s", planner=ScriptedPlanner(bad)).run()
    assert first.stop_reason == "the planner made 3 invalid proposals in a row"
    asked = ScriptedPlanner([])
    second = session(tmp_path / "s", planner=asked).run()
    assert len(asked.contexts) == 1
    assert second.stop_reason == "the planner stopped: done"


def test_an_edited_guidance_md_survives_a_resume(tmp_path, jobs):
    planner = ScriptedPlanner([])
    session(tmp_path / "s", planner=planner, guidance="Try A.").run()
    guidance = tmp_path / "s" / "guidance.md"
    assert guidance.read_text() == "Try A."
    guidance.write_text("Try B instead.")
    session(tmp_path / "s", planner=ScriptedPlanner([]), guidance="Try A.").run()
    assert guidance.read_text() == "Try B instead."
    session(tmp_path / "s", planner=ScriptedPlanner([]), guidance="Try C.").run()
    assert guidance.read_text() == "Try C."


# ------------------------------------------------- failures and their causes


def test_failure_reasons_name_the_first_error_line_and_the_signal():
    from torch_tensorrt.autotune._report import failure_reason

    multiline = {"stage": "build", "error": "RuntimeError: no tactic\nfor layer 3"}
    assert failure_reason(multiline) == "build failed: RuntimeError: no tactic"
    syntax = '  File "kernels.py", line 1\n    def (:\n        ^\nSyntaxError: invalid syntax'
    assert (
        failure_reason({"stage": "setup", "error": syntax})
        == "setup failed: SyntaxError: invalid syntax"
    )
    timeout = {"stage": "timeout", "error": "The worker exceeded", "signal": "SIGKILL"}
    assert failure_reason(timeout) == (
        "timeout failed: The worker exceeded (signal SIGKILL)"
    )
    crash = {
        "stage": "crash",
        "error": "The worker was killed by SIGSEGV",
        "signal": "SIGSEGV",
        "log_error": "Fatal Python error: Segmentation fault",
    }
    assert failure_reason(crash) == (
        "crash failed: The worker was killed by SIGSEGV; log: Fatal Python error: "
        "Segmentation fault"
    )
    assert failure_reason({"stage": "build"}) == "build failed: ?"


def test_worker_logs_give_the_first_error_and_the_stack_dump(tmp_path):
    from torch_tensorrt.autotune._session import _read_worker_log

    log = tmp_path / "worker.log"
    log.write_text(
        "\n".join(
            [
                "W1108 22:35:09.678000 1 torch/x.py:1] an error-ish deprecation",
                "[11/09/2026-10:42:49] [TRT] [W] a warning about errors",
                "Building the engine",
                "ERROR:torch_tensorrt [TensorRT Conversion Context]:Error Code 2",
                "Fatal Python error: Segmentation fault",
                "",
                "Current thread 0x00007f (most recent call first):",
                '  File "workload.py", line 3 in build',
                '  File "_measure.py", line 9 in measure_candidate',
                "",
                "Extension modules: numpy.core._multiarray_umath (total: 1)",
            ]
        )
        + "\n"
    )
    found = _read_worker_log(log)
    assert found["log_error"].startswith("ERROR:torch_tensorrt")
    assert found["log_stack"].startswith("Fatal Python error: Segmentation fault")
    assert found["log_stack"].rstrip().endswith("line 9 in measure_candidate")
    assert "Extension modules" not in found["log_tail"]
    # A native library's own crash handler, which faulthandler calls after
    # writing the dump, can print more: the dump still counts, without it.
    log.write_text(
        "\n".join(
            [
                "Building the engine",
                "Fatal Python error: Segmentation fault",
                "",
                "Current thread 0x00007f (most recent call first):",
                '  File "workload.py", line 3 in build',
                "",
                "Extension modules: numpy.core._multiarray_umath (total: 1)",
                "*** SIGSEGV received at time=1762790000 on cpu 3 ***",
                "PC: @     0x7f0000001000  (unknown)  native_crash_handler",
            ]
        )
        + "\n"
    )
    found = _read_worker_log(log)
    assert found["log_stack"].startswith("Fatal Python error: Segmentation fault")
    assert found["log_stack"].rstrip().endswith("line 3 in build")
    assert "SIGSEGV received" not in found["log_stack"]
    log.write_text("Building the engine\nstill building\n")
    assert _read_worker_log(log) == {
        "log_error": None,
        "log_stack": None,
        "log_tail": "Building the engine\nstill building",
    }


def thread(header, *functions):
    """A faulthandler block: ``header`` and frames, innermost first."""
    return [
        f"{header} (most recent call first):",
        *(f'  File "/src/{name}.py", line 7 in {name}' for name in functions),
        "",
    ]


# Idle threads, as a progress bar's monitor or a thread pool leave them; with
# twelve of them the dump's first 2000 characters hold no other thread.
IDLE = [
    line
    for index in range(12)
    for line in thread(
        f"Thread 0x00007f00000000{index:02d}",
        "wait",
        "idle",
        "run",
        "_bootstrap_inner",
        "_bootstrap",
    )
]
MAIN = ("build", "measure_candidate", "run", "main", "_run_code", "_run_module_as_main")


@pytest.mark.parametrize(
    "dump,first",
    [
        # A fault in the main thread, which faulthandler lists last.
        (
            [
                "Fatal Python error: Segmentation fault",
                "",
                *IDLE,
                *thread("Current thread 0x00007f00000000ff", *MAIN),
            ],
            "Current thread 0x00007f00000000ff",
        ),
        # A fault in a background thread: that thread, wherever it is listed.
        (
            [
                "Fatal Python error: Aborted",
                "",
                *IDLE[:28],
                *thread("Current thread 0x00007f00000000ee", "launch", "worker"),
                *IDLE[28:],
                *thread("Thread 0x00007f00000000ff", *MAIN),
            ],
            "Current thread 0x00007f00000000ee",
        ),
        # A fault in a thread Python does not know: the main thread.
        (
            [
                "Fatal Python error: Segmentation fault",
                "",
                *IDLE,
                *thread("Thread 0x00007f00000000ff", *MAIN),
            ],
            "Thread 0x00007f00000000ff",
        ),
        # The dump before a timeout kill is about the main thread, even when
        # another thread received the signal.
        (
            [
                *IDLE[:7],
                *thread("Current thread 0x00007f00000000ee", "idle", "run"),
                *IDLE[7:],
                *thread("Thread 0x00007f00000000ff", *MAIN),
            ],
            "Thread 0x00007f00000000ff",
        ),
        # More than 100 threads: the main thread is cut, the current one kept.
        (
            [
                "Fatal Python error: Segmentation fault",
                "",
                *thread("Current thread 0x00007f00000000ee", "launch"),
                *IDLE,
                "...",
            ],
            "Current thread 0x00007f00000000ee",
        ),
    ],
)
def test_the_stack_dump_starts_with_the_thread_that_failed(tmp_path, dump, first):
    from torch_tensorrt.autotune._session import _error_excerpt, _read_worker_log

    log = tmp_path / "worker.log"
    log.write_text("\n".join(["Building the engine", *dump]) + "\n")
    found = _read_worker_log(log)
    blocks = found["log_stack"].split("\n\n")
    if dump[0].startswith("Fatal Python error"):
        assert blocks.pop(0) == dump[0]
    assert blocks[0].startswith(first + " (most recent call first):")
    assert "in idle" not in blocks[0]
    # The planner's excerpt starts there too.
    excerpt = _error_excerpt({"status": "failed", **found})
    assert excerpt.startswith(found["log_stack"][:200])
    # The other threads follow, in faulthandler's order, as space allows.
    assert len(found["log_stack"]) == 2000 and len(blocks) > 2


# ------------------------------------------------------- planner failures


class FailingPlanner(Planner):
    """A planner that raises, or returns ``returns`` (not a well-formed Proposal)."""

    name = "failing"

    def __init__(self, returns=None):
        self.returns = returns

    def propose(self, context, workdir):
        if self.returns is not None:
            return self.returns
        raise RuntimeError("the planner broke")


def test_a_failing_planner_is_recorded_and_a_rule_based_planner_stands_in(
    tmp_path, jobs
):
    result = session(
        tmp_path / "s",
        {"speedups": {"engine.precision=fp16": 0.5}},
        planner=FailingPlanner(),
        budget=Budget(max_trials=1),
    ).run()
    trial = tmp_path / "s/trials/0001"
    proposal = json.loads((trial / "proposal.json").read_text())
    assert proposal["planner"] == "rule-based"
    assert proposal["planner_error"] == "RuntimeError: the planner broke"
    assert proposal["note"] == (
        "the failing planner failed (RuntimeError: the planner broke); the "
        "rule-based planner proposed instead"
    )
    # The rule-based planner's choice for the precision hint is measured as usual.
    assert proposal["changes"] == {"engine.precision": "fp16"}
    assert decisions(tmp_path / "s") == ["baseline", "accepted"]
    assert result.decision == "accepted"
    assert "the planner broke" in (trial / "planner/error.txt").read_text()
    events = [
        json.loads(line)
        for line in (tmp_path / "s/events.jsonl").read_text().splitlines()
    ]
    assert [e["error"] for e in events if e["event"] == "planner_failed"] == [
        "RuntimeError: the planner broke"
    ]
    assert "[the failing planner failed" in result.report()


def test_a_planner_that_returns_no_proposal_has_failed(tmp_path, jobs):
    session(
        tmp_path / "s",
        planner=FailingPlanner(returns={"engine.precision": "fp16"}),
        budget=Budget(max_trials=1),
    ).run()
    proposal = json.loads((tmp_path / "s/trials/0001/proposal.json").read_text())
    assert proposal["planner_error"] == (
        "TypeError: FailingPlanner.propose returned dict, not a Proposal"
    )
    assert proposal["planner"] == "rule-based"


@pytest.mark.parametrize(
    "proposal,problem",
    [
        (Proposal(changes=None), "changes must map knob names to values"),
        (Proposal(changes={1: True}), "changes must map knob names to values"),
        (
            Proposal(changes={"engine.precision": {"fp16"}}),
            "it is not JSON data (Object of type set is not JSON serializable)",
        ),
        (
            Proposal(edits={"speed": ["SPEED = 1"]}),
            "edits must map editable file names to lists of {search, replace} edits",
        ),
        (
            Proposal(changes={"engine.precision": "fp16"}, rationale=None),
            "rationale, planner and note must be text",
        ),
    ],
)
def test_a_malformed_proposal_is_a_planner_failure(tmp_path, jobs, proposal, problem):
    session(
        tmp_path / "s",
        planner=FailingPlanner(returns=proposal),
        budget=Budget(max_trials=1),
    ).run()
    recorded = json.loads((tmp_path / "s/trials/0001/proposal.json").read_text())
    assert recorded["planner_error"] == (
        f"TypeError: FailingPlanner.propose returned a malformed Proposal: {problem}"
    )
    assert recorded["planner"] == "rule-based"
    assert decisions(tmp_path / "s") == ["baseline", "rejected"]


def test_the_search_stops_once_the_stand_in_has_nothing_left(tmp_path, jobs):
    result = session(
        tmp_path / "s", planner=FailingPlanner(), budget=Budget(max_trials=10)
    ).run()
    # Every option of the fake workload was tried once, by the stand-in.
    assert decisions(tmp_path / "s") == ["baseline"] + ["rejected"] * 5
    assert result.stop_reason == (
        "the failing planner failed (RuntimeError: the planner broke) and the "
        "rule-based planner that stood in stopped: Every exposed option has been tried"
    )


def test_the_search_stops_when_the_stand_in_fails_too(tmp_path, jobs, monkeypatch):
    def broken(self, context, workdir):
        raise ValueError("no knobs")

    monkeypatch.setattr(RuleBasedPlanner, "propose", broken)
    result = session(tmp_path / "s", planner=FailingPlanner()).run()
    assert result.stop_reason == (
        "the failing planner failed (RuntimeError: the planner broke) and the "
        "rule-based planner that stood in stopped: it failed too (ValueError: no "
        "knobs)"
    )
    assert decisions(tmp_path / "s") == ["baseline"]


def test_per_value_requirements_hold_in_a_session_and_its_resume(tmp_path, jobs):
    kwargs = {"requires": {"workload.variant": {"c": {"runtime.cudagraphs": True}}}}
    planner = ScriptedPlanner(
        [
            Proposal(changes={"workload.variant": "b"}),
            Proposal(changes={"workload.variant": "c"}),
            Proposal(changes={"workload.variant": "c", "runtime.cudagraphs": True}),
        ]
    )
    session(tmp_path / "s", kwargs, planner=planner).run()
    assert decisions(tmp_path / "s") == ["baseline", "rejected", "invalid", "rejected"]
    invalid = json.loads((tmp_path / "s/trials/0002/decision.json").read_text())
    assert invalid["reason"] == (
        "proposal rejected: workload.variant='c' requires runtime.cudagraphs=True"
    )
    third = json.loads((tmp_path / "s/trials/0003/trial.json").read_text())
    assert third["proposed"] == {"workload.variant": "c"}
    session_json = json.loads((tmp_path / "s/session.json").read_text())
    (variant,) = [
        knob
        for knob in session_json["workload"]["knobs"]
        if knob["name"] == "workload.variant"
    ]
    assert variant["requires"] == [
        {"value": "c", "requires": {"runtime.cudagraphs": True}}
    ]
    # The pinned workload, per-value requirements included, matches on resume.
    result = session(tmp_path / "s", kwargs, planner=ScriptedPlanner([])).run()
    assert result.stop_reason == "the planner stopped: done"


# -------------------------------------------------------- prepared kernels

PREPARED = "kernels/kernels.fused=generated"
OTHER = "kernels/kernels.other=generated"


def with_kernels(directory, kwargs=None, **options):
    """A session of the fake workload with the prepared knob ``kernels.fused``."""
    return session(directory, {"kernel_knob": True, **(kwargs or {})}, **options)


def with_two_kernels(directory, kwargs=None, **options):
    """A session with the prepared knobs ``kernels.fused`` and ``kernels.other``.

    ``kwargs["kernels"]`` may give either region's settings.
    """
    kwargs = dict(kwargs or {})
    regions = kwargs.pop("kernels", {})
    kernels = {name: regions.get(name, {}) for name in ("fused", "other")}
    return session(directory, {"kernels": kernels, **kwargs}, **options)


def read(path):
    return json.loads(Path(path).read_text())


def events(directory):
    return [
        json.loads(line)
        for line in (Path(directory) / "events.jsonl").read_text().splitlines()
    ]


def test_a_prepared_kernel_is_built_once_pinned_and_shown_to_the_planner(
    tmp_path, jobs
):
    planner = ScriptedPlanner(
        [Proposal(changes={"kernels.fused": "generated"}, rationale="use it")]
    )
    result = with_kernels(
        tmp_path / "s",
        {"speedups": {"kernels.fused=generated": 0.5}},
        planner=planner,
        budget=Budget(max_trials=2, prepare_timeout_s=60),
    ).run()
    root = tmp_path / "s"
    # After the baseline (and its noise runs), before the search.
    assert [name for name, _ in jobs] == [
        "reference",
        "trials/0000",
        PREPARED,
        "trials/0001",
        "final_check/0-baseline",
        "final_check/0-best",
        "export",
    ]
    # One worker builds the value without timing it, under its own limit.
    request = read(root / PREPARED / "request.json")
    assert request["kind"] == "prepare" and request["timeout_s"] == 60
    assert request["config"] == {"kernels.fused": "generated"}
    assert "allow_generation" not in request
    prepared = read(root / PREPARED / "result.json")
    assert prepared["status"] == "ok" and "latency_ms" not in prepared
    (kernel,) = prepared["kernels"]
    assert (kernel["region"], kernel["origin"], kernel["speedup"]) == (
        "fused",
        "new",
        2.0,
    )
    summary = read(root / "kernels/summary.json")
    (entry,) = summary["preparations"]
    assert entry["status"] == "ok" and entry["reason"] is None
    assert entry["regions"] == [kernel] and entry["region_speedup"] == 2.0
    # One pin per prepared generation task.
    assert summary["pins"] == [
        {
            "task_sha256": kernel["task_sha256"],
            "region": "fused",
            "kernel_sha256": kernel["kernel_sha256"],
            "values": ["kernels.fused=generated"],
        }
    ]
    assert summary["warnings"] == [] and entry["warnings"] == []
    # The planner sees each prepared value, its regions and their measurements.
    (shown,) = planner.contexts[0]["kernels"]
    assert shown == {
        "knob": "kernels.fused",
        "value": "generated",
        "status": "ok",
        "reason": None,
        "prepared_config": {"kernels.fused": "generated"},
        "build_s": prepared["build_s"],
        "region_speedup": 2.0,
        "used_by_best": False,
        "warnings": [],
        "regions": [{key: kernel[key] for key in kernel if key != "manifest"}],
    }
    knob = next(k for k in planner.contexts[0]["knobs"] if k["name"] == "kernels.fused")
    assert knob["prepare"] == ["generated"]
    # Every later run that uses the value only loads kernels, and runs the
    # prepared one; the baseline's runs are not pinned.
    for name in ("trials/0001", "final_check/0-best", "export"):
        assert read(root / name / "request.json")["allow_generation"] is False
        (ran,) = read(root / name / "result.json")["kernels"]
        assert ran["kernel_sha256"] == kernel["kernel_sha256"]
        assert ran["origin"] == "cached"
    assert "allow_generation" not in read(root / "final_check/0-baseline/request.json")
    assert read(root / "trials/0000/result.json")["kernels"] == []
    assert "TORCH_TENSORRT_REGION_GENERATION" not in os.environ
    assert result.decision == "accepted"
    assert result.config == {"kernels.fused": "generated"}
    # The report lists the prepared value and the kernel the winner runs.
    report = result.report()
    assert "## Kernels" in report
    assert (
        f"| `kernels.fused=generated` | ok | yes | `fused#0` | "
        f"`{kernel['kernel_sha256'][:12]}` | new | 0.1 | 0.2 (eager) | 2.00x | 1.5 |"
    ) in report
    (reported,) = read(root / "report.json")["kernels"]
    assert reported == {**entry, "winner_uses": True}
    # trials.tsv keeps its columns.
    header = (root / "trials.tsv").read_text().splitlines()[0].split("\t")
    assert header[:4] == ["trial", "parent", "decision", "metric"]
    assert "kernels" not in header
    assert [e["status"] for e in events(root) if e["event"] == "prepared"] == ["ok"]


def test_a_value_that_could_not_be_prepared_is_never_measured(tmp_path, jobs):
    kwargs = {"broken": ["kernels.fused=generated"]}
    planner = ScriptedPlanner([Proposal(changes={"kernels.fused": "generated"})])
    with_kernels(tmp_path / "s", kwargs, planner=planner).run()
    root = tmp_path / "s"
    assert decisions(root) == ["baseline", "invalid"]
    why = (
        "kernels.fused='generated', which could not be prepared (build failed: "
        "RuntimeError: cannot build ['kernels.fused=generated']; see "
        "kernels/kernels.fused=generated)"
    )
    assert read(root / "trials/0001/decision.json")["reason"] == (
        f"proposal rejected: the config uses {why}"
    )
    # Only the preparation built it, and no trial was spent.
    assert [name for name, _ in jobs if name.startswith(("trials/", "kernels/"))] == [
        "trials/0000",
        PREPARED,
    ]
    (shown,) = planner.contexts[0]["kernels"]
    assert shown["status"] == "failed" and shown["regions"] == []
    assert shown["reason"].startswith("build failed: RuntimeError: cannot build")
    report = (root / "report.md").read_text()
    assert "| `kernels.fused=generated` | failed: build failed: RuntimeError" in report


def test_a_config_that_needs_a_kernel_never_prepared_is_an_invalid_proposal(
    tmp_path, jobs
):
    # The optimization level is part of a region's generation task, so at
    # level 5 the region needs a kernel that was only prepared at level 3;
    # with generation off, its build misses the kernel cache.
    planner = ScriptedPlanner(
        [
            Proposal(
                changes={"kernels.fused": "generated", "engine.optimization_level": 5}
            ),
            Proposal(changes={"kernels.fused": "generated"}),
        ]
    )
    with_kernels(
        tmp_path / "s",
        {
            "kernel_cache": str(tmp_path / "cache.json"),
            "speedups": {"kernels.fused=generated": 0.5},
        },
        planner=planner,
        budget=Budget(max_trials=1, patience=1),
    ).run()
    root = tmp_path / "s"
    # It spent neither a trial nor patience: the next proposal still ran.
    assert decisions(root) == ["baseline", "invalid", "accepted"]
    first = read(root / "trials/0001/result.json")
    (missing,) = first["unprepared"]
    assert (first["status"], first["stage"]) == ("failed", "kernels")
    assert first["error"] == (
        "The config needs region kernels that were never prepared: region "
        f"'fused#0' (generation task {missing['task_sha256'][:12]}). Its settings "
        "differ from those its prepared values were prepared in "
        '(kernels.fused=generated in {"kernels.fused": "generated"}), and runs that '
        "use prepared values only load kernels from the cache"
    )
    assert "region 'fused#0' is disabled by TORCH_TENSORRT_REGION_GENERATION=0" in (
        first["build_error"]
    )
    assert read(root / "trials/0001/decision.json") == {
        "decision": "invalid",
        "reason": f"proposal rejected: {first['error']}",
    }
    # The planner learns why from the history.
    (entry,) = planner.contexts[1]["history"]
    assert (entry["decision"], entry["status"], entry["stage"]) == (
        "invalid",
        "failed",
        "kernels",
    )
    assert entry["error"].startswith("The config needs region kernels that were")
    row = (root / "trials.tsv").read_text().splitlines()[2].split("\t")
    assert (row[0], row[2], row[-2]) == ("0001", "invalid", "kernels")


def test_a_run_whose_kernel_changed_since_its_preparation_fails_at_stage_kernels(
    tmp_path, jobs
):
    cache = tmp_path / "cache.json"

    class Replacing(ScriptedPlanner):
        """Replace every cached kernel before proposing, as a new search would."""

        def propose(self, context, workdir):
            cache.write_text(json.dumps(dict.fromkeys(read(cache), "f" * 64)))
            return super().propose(context, workdir)

    with_kernels(
        tmp_path / "s",
        {"kernel_cache": str(cache)},
        planner=Replacing([Proposal(changes={"kernels.fused": "generated"})]),
        budget=Budget(max_trials=1),
    ).run()
    root = tmp_path / "s"
    (prepared,) = read(root / PREPARED / "result.json")["kernels"]
    result = read(root / "trials/0001/result.json")
    assert result["status"] == "failed" and result["stage"] == "kernels"
    assert result["error"] == (
        "The run did not use the prepared kernels: region 'fused#0' ran kernel "
        f"ffffffffffff (cached) for generation task {prepared['task_sha256'][:12]}, "
        f"not the kernel {prepared['kernel_sha256'][:12]} that "
        "kernels.fused=generated prepared for it"
    )
    # What it measured stays on record; the decision names the stage.
    assert "latency_ms" in result
    decision = read(root / "trials/0001/decision.json")
    assert decision["decision"] == "rejected"
    assert decision["reason"].startswith(
        "kernels failed: The run did not use the prepared kernels"
    )


def test_pinned_kernels_are_matched_by_generation_task():
    from torch_tensorrt.autotune._session import _pinned_kernel_problems

    def kernel(region_id, task, sha, origin="cached"):
        return {
            "region_id": region_id,
            "task_sha256": task,
            "kernel_sha256": sha,
            "origin": origin,
        }

    pins = {"t": {"kernel_sha256": "k", "values": ["kernels.attn=generated"]}}
    ok = {"status": "ok"}
    # A region's name or index does not matter, and a task no preparation
    # made is not checked.
    assert not _pinned_kernel_problems(
        pins,
        {**ok, "kernels": [kernel("region#3", "t", "k"), kernel("region#0", "u", "m")]},
    )
    # A prepared task the run did not register is fine: the config may not
    # compile that region.
    assert not _pinned_kernel_problems(pins, {**ok, "kernels": []})
    (problem,) = _pinned_kernel_problems(
        pins, {**ok, "kernels": [kernel("attn#0", "t", "x", "new")]}
    )
    assert problem == (
        "region 'attn#0' ran kernel x (new) for generation task t, not the kernel "
        "k that kernels.attn=generated prepared for it"
    )
    # A failed run keeps its own failure.
    assert not _pinned_kernel_problems(
        pins, {"status": "failed", "kernels": [kernel("attn#0", "t", "x")]}
    )


def test_regions_with_one_name_are_told_apart_by_their_generation_tasks(tmp_path, jobs):
    # Unnamed regions are all called "region": here the baseline's region and
    # the region the prepared value adds.
    planner = ScriptedPlanner([Proposal(changes={"kernels.other": "generated"})])
    with_two_kernels(
        tmp_path / "s",
        {"kernels": {"fused": {"region": "region"}, "other": {"region": "region"}}},
        planner=planner,
        baseline={"kernels.fused": "generated"},
        budget=Budget(max_trials=1),
    ).run()
    root = tmp_path / "s"
    # The baseline's region is not the prepared value's own.
    (entry,) = read(root / "kernels/summary.json")["preparations"]
    assert [kernel["region_id"] for kernel in entry["regions"]] == ["region#1"]
    trial = read(root / "trials/0001/result.json")
    assert trial["status"] == "ok"
    assert sorted(kernel["region_id"] for kernel in trial["kernels"]) == [
        "region#0",
        "region#1",
    ]


def test_a_config_may_leave_out_a_region_its_prepared_value_also_changed(
    tmp_path, jobs
):
    # kernels.other needs optimization level 5, which also changes the task
    # of the baseline's region "fused", so its preparation generates both.
    planner = ScriptedPlanner(
        [
            Proposal(
                changes={
                    "kernels.other": "generated",
                    "engine.optimization_level": 5,
                    "kernels.fused": "native",
                }
            )
        ]
    )
    with_two_kernels(
        tmp_path / "s",
        {"requires": {"kernels.other": {"engine.optimization_level": 5}}},
        planner=planner,
        baseline={"kernels.fused": "generated"},
        budget=Budget(max_trials=1),
    ).run()
    root = tmp_path / "s"
    (entry,) = read(root / "kernels/summary.json")["preparations"]
    assert sorted(kernel["region"] for kernel in entry["regions"]) == [
        "fused",
        "other",
    ]
    # A config without "fused" runs: only the tasks a run registers count.
    trial = read(root / "trials/0001/result.json")
    assert trial["status"] == "ok"
    assert [kernel["region"] for kernel in trial["kernels"]] == ["other"]


def test_two_prepared_values_that_share_a_task_pin_its_final_kernel(tmp_path, jobs):
    # The region of kernels.other computes what "fused" computes, so both
    # values prepare one generation task, and the second preparation replaced
    # the cached kernel, as a failed revalidation followed by a search would.
    planner = ScriptedPlanner([Proposal(changes={"kernels.fused": "generated"})])
    with_two_kernels(
        tmp_path / "s",
        {
            "kernels": {"other": {"task": "fused"}},
            "kernel_cache": str(tmp_path / "cache.json"),
            "regenerates": ["kernels.other=generated"],
        },
        planner=planner,
        budget=Budget(max_trials=1),
    ).run()
    root = tmp_path / "s"
    (first,) = read(root / PREPARED / "result.json")["kernels"]
    (final,) = read(root / OTHER / "result.json")["kernels"]
    assert first["task_sha256"] == final["task_sha256"]
    assert first["kernel_sha256"] != final["kernel_sha256"]
    summary = read(root / "kernels/summary.json")
    values = ["kernels.fused=generated", "kernels.other=generated"]
    assert summary["pins"] == [
        {
            "task_sha256": final["task_sha256"],
            "region": "other",
            "kernel_sha256": final["kernel_sha256"],
            "values": values,
        }
    ]
    (warning,) = summary["warnings"]
    assert warning["values"] == values
    assert warning["replaced"] == [
        {"kernel_sha256": first["kernel_sha256"], "values": values[:1]}
    ]
    assert "changed while knob values were prepared" in warning["message"]
    # Both values show the final kernel, and the warning, to the planner.
    fused, other = planner.contexts[0]["kernels"]
    assert fused["regions"][0]["kernel_sha256"] == final["kernel_sha256"]
    assert fused["regions"][0]["replaced_kernel_sha256"] == first["kernel_sha256"]
    assert fused["warnings"] == other["warnings"] == [warning["message"]]
    # A run of kernels.fused=generated loads the final kernel, as pinned.
    trial = read(root / "trials/0001/result.json")
    assert trial["status"] == "ok"
    assert trial["kernels"][0]["kernel_sha256"] == final["kernel_sha256"]
    assert f"- warning: {warning['message']}" in (root / "report.md").read_text()
    assert [
        e["task_sha256"] for e in events(root) if e["event"] == "kernel_changed"
    ] == [final["task_sha256"]]


def test_the_time_budget_is_checked_before_each_preparation(
    tmp_path, jobs, monkeypatch
):
    from torch_tensorrt.autotune import _session

    # Each preparation takes 1000 s of session time; the budget is 500 s.
    elapsed = _session._Progress.elapsed
    monkeypatch.setattr(
        _session._Progress,
        "elapsed",
        lambda self: elapsed(self)
        + 1000.0 * sum(1 for _, request in jobs if request["kind"] == "prepare"),
    )
    result = with_two_kernels(
        tmp_path / "s", budget=Budget(max_trials=5, max_time_s=500)
    ).run()
    assert [name for name, _ in jobs if name.startswith("kernels/")] == [PREPARED]
    first, second = read(tmp_path / "s/kernels/summary.json")["preparations"]
    assert first["status"] == "ok"
    assert (second["status"], second["reason"]) == (
        "skipped",
        "the search runs no further trial (time budget of 500 s used)",
    )
    assert result.stop_reason == "time budget of 500 s used"


def test_a_goal_the_baseline_meets_starts_no_preparation(tmp_path, jobs):
    result = with_kernels(
        tmp_path / "s", goal=Goal(min_improvement=0.2, target=1e9)
    ).run()
    assert result.stop_reason == "goal target reached"
    assert not any(name.startswith("kernels/") for name, _ in jobs)
    (entry,) = read(tmp_path / "s/kernels/summary.json")["preparations"]
    assert (entry["status"], entry["reason"]) == (
        "skipped",
        "the search runs no further trial (goal target reached)",
    )


def test_a_resume_reuses_successful_preparations(tmp_path, jobs):
    with_kernels(tmp_path / "s", budget=Budget(max_trials=1)).run()
    assert PREPARED in [name for name, _ in jobs]
    jobs.clear()
    with_kernels(tmp_path / "s", budget=Budget(max_trials=2)).run()
    assert not any(name.startswith("kernels/") for name, _ in jobs)
    (entry,) = read(tmp_path / "s/kernels/summary.json")["preparations"]
    assert entry["status"] == "ok"
    assert [
        e["reused"] for e in events(tmp_path / "s") if e["event"] == "prepared"
    ] == [
        False,
        True,
    ]
    # A preparation stopped before its result is prepared again.
    (tmp_path / "s" / PREPARED / "result.json").unlink()
    jobs.clear()
    with_kernels(tmp_path / "s", budget=Budget(max_trials=3)).run()
    assert [name for name, _ in jobs if name.startswith("kernels/")] == [PREPARED]


def test_a_resume_prepares_a_failed_preparation_again(tmp_path, jobs):
    # Like a failed baseline, since its cause may be gone; but not once the
    # budget leaves no trial that could use it.
    kwargs = {"broken": ["kernels.fused=generated"]}
    for trials, prepared in ((1, True), (2, True), (2, False)):
        jobs.clear()
        with_kernels(tmp_path / "s", kwargs, budget=Budget(max_trials=trials)).run()
        assert (PREPARED in [name for name, _ in jobs]) is prepared
        (entry,) = read(tmp_path / "s/kernels/summary.json")["preparations"]
        assert entry["status"] == "failed"


def test_no_preparation_starts_once_the_budget_leaves_no_trial(tmp_path, jobs):
    with_kernels(tmp_path / "s", budget=Budget(max_trials=0)).run()
    assert not any(name.startswith("kernels/") for name, _ in jobs)
    (entry,) = read(tmp_path / "s/kernels/summary.json")["preparations"]
    assert entry["status"] == "skipped"
    assert entry["reason"] == (
        "the search runs no further trial (trial budget of 0 used)"
    )
    jobs.clear()
    with_kernels(tmp_path / "s", budget=Budget(max_trials=1)).run()
    assert PREPARED in [name for name, _ in jobs]


def test_a_value_the_best_config_uses_is_prepared_even_without_trials_left(
    tmp_path, jobs
):
    kwargs = {"speedups": {"kernels.fused=generated": 0.5}}
    planner = ScriptedPlanner([Proposal(changes={"kernels.fused": "generated"})])
    with_kernels(
        tmp_path / "s", kwargs, planner=planner, budget=Budget(max_trials=1)
    ).run()
    # As if this preparation had been interrupted; the budget is used up.
    (tmp_path / "s" / PREPARED / "result.json").unlink()
    jobs.clear()
    result = with_kernels(tmp_path / "s", kwargs, budget=Budget(max_trials=1)).run()
    assert [name for name, _ in jobs][:2] == ["describe", PREPARED]
    # The final check and the export of the best config can still be pinned.
    assert result.decision == "accepted" and result.final_check["passed"]
    assert result.export_summary["status"] == "exported"


def test_a_value_the_baseline_sets_is_not_prepared(tmp_path, jobs):
    with_kernels(
        tmp_path / "s",
        baseline={"kernels.fused": "generated"},
        budget=Budget(max_trials=1),
    ).run()
    assert not (tmp_path / "s/kernels").exists()
    # The baseline's own build made its kernel; nothing pins the runs.
    assert read(tmp_path / "s/trials/0000/result.json")["kernels"][0]["origin"] == "new"
    assert "allow_generation" not in read(tmp_path / "s/trials/0001/request.json")


def test_status_lists_the_prepared_values(tmp_path, jobs, capsys):
    with_kernels(tmp_path / "s", budget=Budget(max_trials=1)).run()
    with_kernels(
        tmp_path / "f",
        {"broken": ["kernels.fused=generated"]},
        budget=Budget(max_trials=1),
    ).run()
    capsys.readouterr()
    assert cli.main(["status", str(tmp_path / "s")]) == 0
    assert (
        "  prepared kernels.fused=generated: ok; 1 region kernels, region speedup "
        "2.00x or more\n"
    ) in capsys.readouterr().out
    assert cli.main(["status", str(tmp_path / "f")]) == 0
    assert (
        "  prepared kernels.fused=generated: failed; build failed: RuntimeError: "
        "cannot build ['kernels.fused=generated']\n"
    ) in capsys.readouterr().out


# ------------------------------------------------------------------ export


def test_an_export_that_misses_the_bounds_is_not_the_artifact(tmp_path, jobs):
    jobs.inaccurate_export = True
    result = session(
        tmp_path / "s",
        {"speedups": {"engine.precision=fp16": 0.5}},
        budget=Budget(max_trials=1),
    ).run()
    assert result.decision == "accepted" and result.artifact is None
    summary = result.export_summary
    assert summary["status"] == "export_rejected"
    assert summary["reason"].startswith(
        "the rebuilt candidate missed the accuracy bounds (max_abs_error=0.5"
    )
    assert summary == json.loads((tmp_path / "s/export/summary.json").read_text())
    root = result.directory
    # Only an export that passed makes an artifact; the config is kept aside.
    assert not (root / "artifact").exists() and "kept" not in summary
    assert json.loads((root / "export/config.json").read_text()) == result.config
    assert (root / "export/files/fake.txt").is_file()
    report_text = result.report()
    assert "Artifact: not exported (export_rejected: the rebuilt" in report_text
    assert f"(also in `{root / 'export' / 'config.json'}`)" in report_text
    report = json.loads((root / "report.json").read_text())
    assert report["export"]["status"] == "export_rejected"


@pytest.mark.parametrize(
    "option,status,reason",
    [
        (
            "unexportable",
            "export_unsupported",
            "export failed: NotImplementedError: this build cannot be saved",
        ),
        (
            "export_breaks",
            "export_failed",
            "export failed: RuntimeError: the disk is full",
        ),
    ],
)
def test_exports_that_cannot_be_made_are_recorded(
    tmp_path, jobs, option, status, reason
):
    result = session(
        tmp_path / "s",
        {"speedups": {"engine.precision=fp16": 0.5}, option: ["engine.precision=fp16"]},
        budget=Budget(max_trials=1),
    ).run()
    # The session finishes; only the artifact is missing.
    assert result.decision == "accepted" and result.artifact is None
    assert result.export_summary["status"] == status
    assert result.export_summary["reason"] == reason
    assert f"Artifact: not exported ({status}: {reason})" in result.report()


def test_a_validated_export_replaces_the_previous_artifact(tmp_path, jobs):
    kwargs = {"speedups": {"engine.precision=fp16": 0.5}}
    first = session(tmp_path / "s", kwargs, budget=Budget(max_trials=1)).run()
    assert first.export_summary["status"] == "exported"
    assert first.export_summary["files"] == ["fake.txt"]
    (first.artifact / "stale.bin").write_text("an earlier export")
    second = session(tmp_path / "s", kwargs, budget=Budget(max_trials=1)).run()
    assert sorted(path.name for path in second.artifact.iterdir()) == [
        "config.json",
        "fake.txt",
    ]
    assert "exported: the rebuilt candidate met the accuracy bounds" in second.report()
    assert not (second.directory / "export/previous_artifact").exists()
    assert not (second.directory / "export/files").exists()


def test_an_export_that_fails_keeps_the_artifact_that_passed(tmp_path, jobs):
    kwargs = {"speedups": {"engine.precision=fp16": 0.5}}
    first = session(tmp_path / "s", kwargs, budget=Budget(max_trials=1)).run()
    artifact = first.artifact
    assert first.export_summary["status"] == "exported"
    # A resume with the same winner, whose export worker times out this time.
    jobs.export_times_out = True
    second = session(tmp_path / "s", kwargs, budget=Budget(max_trials=1)).run()
    assert second.config == first.config and second.artifact is None
    summary = second.export_summary
    assert summary["status"] == "export_failed"
    assert summary["reason"].startswith("timeout failed: The worker exceeded")
    assert summary["kept"] == {
        "artifact": str(artifact),
        "config": first.config,
        "files": ["fake.txt"],
    }
    assert sorted(path.name for path in artifact.iterdir()) == [
        "config.json",
        "fake.txt",
    ]
    assert f"`{artifact}` keeps an earlier export of this config" in second.report()


def test_an_artifact_of_an_earlier_winner_is_named_as_such(tmp_path, jobs):
    kwargs = {"speedups": {"engine.precision=fp16": 0.5}}
    first = session(tmp_path / "s", kwargs, budget=Budget(max_trials=1)).run()
    assert first.config == {"engine.precision": "fp16"}
    # As if an earlier winner had been exported, and this one is rejected.
    _worker.write_json(first.artifact / "config.json", {})
    jobs.inaccurate_export = True
    second = session(tmp_path / "s", kwargs, budget=Budget(max_trials=1)).run()
    assert second.export_summary["status"] == "export_rejected"
    assert second.export_summary["kept"]["config"] == {}
    assert (first.artifact / "fake.txt").is_file()
    assert "keeps an earlier export of another config (see its config.json)" in (
        second.report()
    )


def test_an_artifact_swap_stopped_halfway_is_undone(tmp_path, jobs):
    kwargs = {"speedups": {"engine.precision=fp16": 0.5}}
    first = session(tmp_path / "s", kwargs, budget=Budget(max_trials=1)).run()
    # Stopped after moving the old artifact aside, before the new one moved in.
    aside = first.directory / "export" / "previous_artifact"
    first.artifact.rename(aside)
    jobs.export_times_out = True
    second = session(tmp_path / "s", kwargs, budget=Budget(max_trials=1)).run()
    assert second.export_summary["kept"]["config"] == first.config
    assert sorted(path.name for path in first.artifact.iterdir()) == [
        "config.json",
        "fake.txt",
    ]
    assert not aside.exists()


# -------------------------------------------------------------- provenance


def copy_workload(directory):
    """Copy the fake workload's files to ``directory``; return the copy's factory."""
    directory.mkdir()
    for name in ("_workloads.py", "_fake_editable.py"):
        shutil.copy(Path(__file__).with_name(name), directory / name)
    return f"{directory / '_workloads.py'}:create_fake"


def test_a_changed_workload_source_refuses_the_resume(tmp_path, jobs):
    factory = copy_workload(tmp_path / "workload")
    session(tmp_path / "s", factory=factory, budget=Budget(max_trials=0)).run()
    pinned = json.loads((tmp_path / "s/session.json").read_text())["workload"]
    assert sorted(Path(path).name for path in pinned["workload_files"]) == [
        "_fake_editable.py",
        "_workloads.py",
    ]
    helper = tmp_path / "workload" / "helper.py"
    helper.write_text("X = 1\n")
    with pytest.raises(
        TuningError,
        match=r"accuracy defaults or source files changed.*helper.py was added",
    ):
        session(tmp_path / "s", factory=factory).run()
    helper.unlink()
    with open(tmp_path / "workload" / "_workloads.py", "a") as stream:
        stream.write("# edited\n")
    with pytest.raises(TuningError, match=r"_workloads.py changed"):
        session(tmp_path / "s", factory=factory).run()


def test_changed_torch_tensorrt_source_refuses_the_resume(tmp_path, jobs):
    session(tmp_path / "s", budget=Budget(max_trials=0)).run()
    path = tmp_path / "s/session.json"
    session_json = json.loads(path.read_text())
    source = session_json["environment"]["torch_tensorrt_source"]
    assert source == {"commit": "0" * 40, "uncommitted_sha256": None}
    source["uncommitted_sha256"] = "f" * 64
    path.write_text(json.dumps(session_json))
    with pytest.raises(
        TuningError,
        match=r"Torch-TensorRT source or GPU changed.*torch_tensorrt_source"
        r"\.uncommitted_sha256 was 'f+', now None",
    ):
        session(tmp_path / "s").run()


def test_a_source_that_cannot_be_pinned_is_named_and_warned_about(
    tmp_path, jobs, caplog
):
    why = "git rev-parse failed: fatal: detected dubious ownership in repository"
    jobs.source = {"unavailable": why}
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.autotune._session"):
        session(tmp_path / "s", budget=Budget(max_trials=0)).run()
    assert f"The Torch-TensorRT source cannot be pinned ({why})" in caplog.text
    path = tmp_path / "s/session.json"
    session_json = json.loads(path.read_text())
    assert session_json["environment"]["torch_tensorrt_source"] == {"unavailable": why}
    # Once git reads the checkout, nothing tells what changed in between.
    jobs.source = {"commit": "1" * 40, "uncommitted_sha256": None}
    with pytest.raises(
        TuningError,
        match=r"torch_tensorrt_source\.commit was None, now '1+'.*"
        r"torch_tensorrt_source\.unavailable was 'git rev-parse failed",
    ):
        session(tmp_path / "s").run()
    # An installed package has no source record to compare.
    session_json["environment"]["torch_tensorrt_source"] = None
    path.write_text(json.dumps(session_json))
    with pytest.raises(
        TuningError, match=r"torch_tensorrt_source was None, now \{'commit': '1+'"
    ):
        session(tmp_path / "s").run()


def test_a_directory_of_another_format_is_refused(tmp_path, jobs, capsys):
    session(tmp_path / "s", budget=Budget(max_trials=0)).run()
    path = tmp_path / "s/session.json"
    session_json = json.loads(path.read_text())
    assert (session_json["format"], session_json["version"]) == (
        "torch_tensorrt.autotune session",
        1,
    )
    # As another program, or another version of the Auto-Tuner, would leave it.
    del session_json["format"]
    path.write_text(json.dumps(session_json))
    refused = "is not an Auto-Tuner session of this version"
    with pytest.raises(TuningError, match=refused):
        session(tmp_path / "s", budget=Budget(max_trials=0)).run()
    with pytest.raises(TuningError, match=refused):
        TuningSession.resume(tmp_path / "s")
    assert cli.main(["status", str(tmp_path / "s")]) == 1
    assert refused in capsys.readouterr().err
    # resume prints the refusal and exits 1, without a traceback.
    assert cli.main(["resume", str(tmp_path / "s")]) == 1
    assert refused in capsys.readouterr().err
    with pytest.raises(TuningError, match="it has no session.json"):
        TuningSession.resume(tmp_path / "elsewhere")


def test_reference_outputs_of_other_code_are_not_reused(tmp_path, jobs):
    session(tmp_path / "s", budget=Budget(max_trials=0)).run()
    root = tmp_path / "s"
    # As if the first run had stopped before its first checkpoint, after the
    # reference was saved, and the workload's code had changed since.
    session_json = json.loads((root / "session.json").read_text())
    del session_json["workload"], session_json["environment"]
    (root / "session.json").write_text(json.dumps(session_json))
    reference = json.loads((root / "reference/result.json").read_text())
    reference["workload"]["workload_files"]["/elsewhere/workload.py"] = "0" * 64
    (root / "reference/result.json").write_text(json.dumps(reference))
    with pytest.raises(TuningError, match=r"/elsewhere/workload.py was removed"):
        session(root).run()


def test_the_report_ends_a_stop_reason_with_one_period(tmp_path, jobs):
    result = session(
        tmp_path / "s",
        planner=ScriptedPlanner(
            [Proposal(stop=True, rationale="Nothing left to try.")]
        ),
    ).run()
    assert "Search stopped: the planner stopped: Nothing left to try.\n" in (
        result.report()
    )
