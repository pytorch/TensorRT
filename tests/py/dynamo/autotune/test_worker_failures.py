# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Sessions whose workers crash or time out, run in real worker processes, and resumed."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from torch_tensorrt.autotune import (
    Budget,
    Goal,
    NoiseControl,
    Planner,
    Proposal,
    TuningError,
    TuningSession,
    WorkloadFactory,
)

from ._workloads import CRASH_ENV

WORKLOADS = Path(__file__).with_name("_workloads.py")
FACTORY = f"{WORKLOADS}:create_fake"
SINGLE_RUN = NoiseControl(
    noise_runs=0, head_to_head_pairs=0, quiet_gpu_s=0, max_spread=10
)
# Workers see no git repository, so edits to this checkout while a test runs
# do not count as a changed environment between a run and its resume.
NO_GIT = {"GIT_DIR": os.devnull}

pytestmark = pytest.mark.skipif(
    not hasattr(signal, "SIGUSR1"), reason="needs POSIX signals"
)


class ScriptedPlanner(Planner):
    """Propose the given knob changes in turn, then stop; keep the contexts it saw."""

    name = "scripted"

    def __init__(self, *changes):
        self.changes = list(changes)
        self.contexts = []

    def propose(self, context, workdir):
        self.contexts.append(context)
        if not self.changes:
            return Proposal(stop=True, rationale="script done")
        return Proposal(changes=self.changes.pop(0), rationale="scripted")


def session(
    directory,
    kwargs=None,
    *,
    timeout=120.0,
    prepare_timeout=7200.0,
    max_trials=1,
    **options,
):
    """A session of the fake workload whose workers are real processes."""
    options.setdefault("goal", Goal(min_improvement=0.2))
    options.setdefault("noise_control", SINGLE_RUN)
    options.setdefault("export", False)
    options.setdefault("env", NO_GIT)
    return TuningSession(
        WorkloadFactory(FACTORY, kwargs or {}),
        directory=directory,
        budget=Budget(
            max_trials=max_trials,
            trial_timeout_s=timeout,
            prepare_timeout_s=prepare_timeout,
        ),
        **options,
    )


def read(path):
    """The JSON in ``path``."""
    return json.loads(Path(path).read_text())


def resumed_history(directory, kwargs, **options):
    """Resume with one more trial; return the history the planner then saw."""
    planner = ScriptedPlanner()
    result = session(directory, kwargs, max_trials=2, planner=planner, **options).run()
    assert result.stop_reason == "the planner stopped: script done"
    return planner.contexts[0]["history"]


def test_a_timed_out_trial_keeps_its_cause_across_a_resume(tmp_path):
    # Background threads (a progress bar's monitor, a thread pool) come first
    # in the worker's stack dump; the main thread, where it hung, last.
    kwargs = {"hangs": ["engine.precision=fp16"], "threads": 4}
    directory = tmp_path / "s"
    session(
        directory,
        kwargs,
        timeout=20.0,
        planner=ScriptedPlanner({"engine.precision": "fp16"}),
    ).run()
    trial = directory / "trials" / "0001"
    result = read(trial / "result.json")
    assert result["synthesized"] and result["stage"] == "timeout"
    assert result["exit_status"] == -9 and result["signal"] == "SIGKILL"
    # Asked for its stacks before the kill, the worker logged where it hung.
    assert "in build" in result["log_tail"]
    # Then the first error line of the log, if it has one.
    reason = "timeout failed: The worker exceeded trial_timeout_s=20 (signal SIGKILL)"
    assert read(trial / "decision.json")["reason"].startswith(reason)
    row = (directory / "trials.tsv").read_text().splitlines()[2].split("\t")
    assert row[0] == "0001" and row[-2:] == ["timeout", "SIGKILL"]
    assert "failed (timeout, SIGKILL)" in (directory / "report.md").read_text()

    recorded = trial.joinpath("worker.log").stat().st_mtime_ns
    history = resumed_history(directory, kwargs, timeout=20.0)
    # The trial is not run again, and its cause reaches the planner.
    assert trial.joinpath("worker.log").stat().st_mtime_ns == recorded
    assert history[0]["status"] == "failed" and history[0]["stage"] == "timeout"
    assert history[0]["signal"] == "SIGKILL" and history[0]["exit_status"] == -9
    assert history[0]["error"].startswith(reason.removeprefix("timeout failed: "))
    # The planner sees where the main thread hung, not the idle threads.
    main_thread = history[0]["error_tail"].split("\n\n")[0]
    assert "in build" in main_thread and "in idle" not in main_thread
    log = trial.joinpath("worker.log").read_text()
    assert log.count("(most recent call first):") == 5


def test_a_crashed_trial_keeps_its_cause_across_a_resume(tmp_path):
    kwargs = {"crashes": ["engine.precision=fp16"], "threads": 4}
    directory = tmp_path / "s"
    session(
        directory, kwargs, planner=ScriptedPlanner({"engine.precision": "fp16"})
    ).run()
    trial = directory / "trials" / "0001"
    result = read(trial / "result.json")
    assert result["synthesized"] and result["stage"] == "crash"
    assert result["exit_status"] == -signal.SIGSEGV and result["signal"] == "SIGSEGV"
    # faulthandler names the fault and the stack in the log.
    assert result["log_stack"].startswith("Fatal Python error: Segmentation fault")
    assert "in crash" in result["log_stack"] and result["log_error"]
    decision = read(trial / "decision.json")
    assert decision["reason"] == (
        f"crash failed: The worker was killed by SIGSEGV (exit status "
        f"{-signal.SIGSEGV}) before writing a result; log: {result['log_error']}"
    )
    report = read(directory / "report.json")
    assert report["trials"][1]["signal"] == "SIGSEGV"
    assert report["trials"][1]["stage"] == "crash"

    history = resumed_history(directory, kwargs)
    assert history[0]["status"] == "failed" and history[0]["stage"] == "crash"
    assert history[0]["signal"] == "SIGSEGV"
    assert history[0]["error"] == decision["reason"].removeprefix("crash failed: ")
    # The planner sees the faulting thread from the fault inward, not the
    # idle threads faulthandler lists before it.
    fatal, faulted = history[0]["error_tail"].split("\n\n")[:2]
    assert fatal == "Fatal Python error: Segmentation fault"
    assert faulted.startswith("Current thread 0x")
    assert "in crash" in faulted and "in idle" not in faulted


def test_a_preparation_has_its_own_time_limit_and_pinned_workers_only_load(
    tmp_path,
):
    # The preparation hangs: it is stopped after prepare_timeout_s, not
    # trial_timeout_s, and the value is then refused.
    kwargs = {"kernel_knob": True, "hangs": ["kernels.fused=generated"]}
    planner = ScriptedPlanner({"kernels.fused": "generated"})
    session(tmp_path / "hung", kwargs, prepare_timeout=20.0, planner=planner).run()
    prepared = read(tmp_path / "hung/kernels/kernels.fused=generated/result.json")
    assert prepared["synthesized"] and prepared["stage"] == "timeout"
    assert prepared["error"] == "The worker exceeded prepare_timeout_s=20"
    reason = read(tmp_path / "hung/trials/0001/decision.json")["reason"]
    assert reason.startswith(
        "proposal rejected: the config uses kernels.fused='generated', which could "
        "not be prepared (timeout failed: The worker exceeded prepare_timeout_s=20"
    )
    # A real worker of a run that uses the prepared value runs with kernel
    # generation off, so the fake kernel says it was loaded from the cache.
    planner = ScriptedPlanner({"kernels.fused": "generated"})
    session(tmp_path / "s", {"kernel_knob": True}, planner=planner).run()
    prepared = read(tmp_path / "s/kernels/kernels.fused=generated/result.json")
    trial = read(tmp_path / "s/trials/0001/result.json")
    assert prepared["kernels"][0]["origin"] == "new"
    assert trial["status"] == "ok" and trial["kernels"][0]["origin"] == "cached"
    assert (
        trial["kernels"][0]["kernel_sha256"] == prepared["kernels"][0]["kernel_sha256"]
    )


def test_a_crashed_preparation_is_prepared_again_on_resume(tmp_path):
    kwargs = {"kernel_knob": True}
    directory = tmp_path / "s"
    prepared = directory / "kernels" / "kernels.fused=generated"
    # The preparation's worker dies of SIGSEGV, as in a native library, so
    # the value cannot be used this time.
    session(
        directory,
        kwargs,
        planner=ScriptedPlanner({"kernels.fused": "generated"}),
        env={**NO_GIT, CRASH_ENV: "kernels.fused=generated"},
    ).run()
    crashed = read(prepared / "result.json")
    assert crashed["synthesized"] and crashed["stage"] == "crash"
    assert crashed["signal"] == "SIGSEGV"
    assert read(directory / "trials/0001/decision.json")["decision"] == "invalid"
    # Resumed once the cause is gone, it is prepared again, and runs use it.
    session(
        directory,
        kwargs,
        max_trials=2,
        planner=ScriptedPlanner({"kernels.fused": "generated"}),
    ).run()
    assert read(prepared / "result.json")["status"] == "ok"
    trial = read(directory / "trials/0002/result.json")
    assert trial["status"] == "ok" and trial["kernels"][0]["origin"] == "cached"


# Stands in for the worker (the session runs it as its Python): writes a
# result or not, then ends as FAKE_WORKER_END says.
FAKE_WORKER = """
import json, os, resource, signal, sys, time

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
request = json.load(open(sys.argv[-1]))
end = os.environ["FAKE_WORKER_END"]
if end.endswith("its result"):
    with open(request["result_path"], "w") as stream:
        json.dump({"status": "ok", "latency_ms": {"median": 1.0}}, stream)
if end == "crashes after writing its result":
    os.kill(os.getpid(), signal.SIGSEGV)
elif end == "hangs after writing its result":
    # Like the worker: log the stacks on SIGUSR1 and carry on.
    signal.signal(signal.SIGUSR1, lambda *_: print("stacks", flush=True))
    time.sleep(600)
elif end == "exits on SIGUSR1 after writing its result":
    signal.signal(signal.SIGUSR1, lambda *_: sys.exit(0))
    time.sleep(600)
elif end == "is killed":
    os.kill(os.getpid(), signal.SIGKILL)
elif end == "exits with status 3":
    sys.exit(3)
"""


@pytest.mark.parametrize(
    "end,expected",
    [
        (
            "crashes after writing its result",
            {"status": "ok", "exit_status": -signal.SIGSEGV, "signal": "SIGSEGV"},
        ),
        (
            "hangs after writing its result",
            {
                "status": "ok",
                "exit_status": -signal.SIGKILL,
                "signal": "SIGKILL",
                "exit_note": "killed after trial_timeout_s=5, having written its result",
            },
        ),
        (
            "exits on SIGUSR1 after writing its result",
            {
                "status": "ok",
                "exit_status": 0,
                "signal": None,
                "exit_note": "exceeded trial_timeout_s=5, then exited before it was "
                "killed",
            },
        ),
        (
            "is killed",
            {
                "status": "failed",
                "stage": "crash",
                "error": "The worker was killed by SIGKILL (exit status -9) before "
                "writing a result; the session did not send it (an out-of-memory "
                "kill?)",
                "exit_status": -signal.SIGKILL,
                "signal": "SIGKILL",
                "synthesized": True,
            },
        ),
        (
            "exits with status 3",
            {
                "status": "failed",
                "stage": "crash",
                "error": "The worker exited with status 3 without writing a result",
                "exit_status": 3,
                "signal": None,
                "synthesized": True,
            },
        ),
    ],
)
def test_how_a_worker_ended_is_recorded(tmp_path, end, expected):
    worker = tmp_path / "worker.py"
    worker.write_text(FAKE_WORKER)
    # exec keeps the process ID, so the session's signals reach the worker.
    python = tmp_path / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{worker}" "$@"\n')
    python.chmod(0o755)
    runner = session(
        tmp_path / "s", timeout=5.0, python=str(python), env={"FAKE_WORKER_END": end}
    )
    result = runner._run_worker({"kind": "measure"}, runner.directory / "job")
    assert {key: result.get(key) for key in expected} == expected
    assert read(runner.directory / "job" / "result.json") == result
    if result["status"] == "ok":
        # A result the worker wrote is kept, with how the worker ended.
        assert result["latency_ms"] == {"median": 1.0}
        assert "synthesized" not in result
        assert ("exit_note" in result) == ("exit_note" in expected)


def test_a_failed_baseline_is_measured_again_on_resume(tmp_path):
    directory = tmp_path / "s"
    crashing = {**NO_GIT, CRASH_ENV: "workload.variant=a"}
    with pytest.raises(
        TuningError, match="baseline is not valid.*crash failed.*SIGSEGV"
    ):
        session(directory, max_trials=0, env=crashing).run()
    baseline = directory / "trials" / "0000"
    first = baseline.joinpath("worker.log").stat().st_mtime_ns
    # Still crashing: measured again, refused again.
    with pytest.raises(TuningError, match="baseline is not valid.*SIGSEGV"):
        session(directory, max_trials=0, env=crashing).run()
    second = baseline.joinpath("worker.log").stat().st_mtime_ns
    assert second > first
    # Once the cause is gone, the session goes on from a validated baseline.
    result = session(directory, max_trials=0).run()
    assert read(baseline / "decision.json")["decision"] == "baseline"
    assert result.stop_reason == "trial budget of 0 used"
    events = [
        json.loads(line)["event"]
        for line in (directory / "events.jsonl").read_text().splitlines()
    ]
    assert events.count("baseline_retry") == 2


CONTROLLER = """
import os, signal, sys
from pathlib import Path

from torch_tensorrt.autotune import (
    Budget, Goal, NoiseControl, Planner, Proposal, TuningSession, WorkloadFactory,
)
from torch_tensorrt.autotune import _session


class Invalid(Planner):
    name = "invalid"

    def propose(self, context, workdir):
        return Proposal(changes={"engine.turbo": True})


write_json = _session.write_json


def stop_before_the_decision(path, value):
    # The moment a kill would leave trial.json without decision.json.
    if Path(path).name == "decision.json" and Path(path).parent.name == "0001":
        os.kill(os.getpid(), signal.SIGKILL)
    write_json(path, value)


_session.write_json = stop_before_the_decision
TuningSession(
    WorkloadFactory(sys.argv[2]),
    directory=sys.argv[1],
    goal=Goal(min_improvement=0.2),
    budget=Budget(max_trials=2, trial_timeout_s=120),
    planner=Invalid(),
    noise_control=NoiseControl(noise_runs=0, head_to_head_pairs=0, quiet_gpu_s=0, max_spread=10),
    export=False,
    env={"GIT_DIR": os.devnull},
).run()
"""


def test_an_invalid_proposal_stopped_before_its_decision_is_kept_on_resume(tmp_path):
    directory = tmp_path / "s"
    script = tmp_path / "controller.py"
    script.write_text(CONTROLLER)
    killed = subprocess.run(
        [sys.executable, "-P", str(script), str(directory), FACTORY],
        capture_output=True,
        timeout=600,
    )
    assert killed.returncode == -signal.SIGKILL, killed.stderr.decode()[-2000:]
    trial = directory / "trials" / "0001"
    assert read(trial / "trial.json")["decision"]["decision"] == "invalid"
    assert not (trial / "decision.json").exists()

    history = resumed_history(directory, {})
    # Kept as invalid: never measured, and its decision file is restored.
    assert not (trial / "request.json").exists()
    assert read(trial / "decision.json")["reason"].startswith(
        "proposal rejected: Unknown knob 'engine.turbo'"
    )
    assert [entry["status"] for entry in history] == ["invalid"]
    table = (directory / "trials.tsv").read_text().splitlines()
    assert [row.split("\t")[2] for row in table[1:]] == ["baseline", "invalid"]
