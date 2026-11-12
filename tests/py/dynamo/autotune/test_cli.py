# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``python -m torch_tensorrt.autotune`` command line, on the CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from torch_tensorrt.autotune import (
    Budget,
    Goal,
    LLMPlanner,
    NoiseControl,
    RuleBasedPlanner,
    TuningSession,
)
from torch_tensorrt.autotune import __main__ as cli
from torch_tensorrt.autotune import (
    _worker,
)

WORKLOADS = Path(__file__).with_name("_workloads.py")
FACTORY = f"{WORKLOADS}:create_fake"
# One fresh run per trial and no wait for an idle GPU: the fake workload
# sleeps, so its samples spread widely on a loaded machine, and the scripted
# speedup (2x) dwarfs that noise.
QUICK = [
    "--min-improvement=0.2",
    "--noise-runs=0",
    "--head-to-head-pairs=0",
    "--final-check-pairs=1",
    "--quiet-gpu-s=0",
    "--max-spread=10",
]
FASTER_FP16 = json.dumps({"speedups": {"engine.precision=fp16": 0.5}})


def parse(*argv):
    """The arguments of ``run`` with ``argv``."""
    return cli._parser().parse_args(["run", *argv])


def build(tmp_path, *argv):
    """The session ``run`` would tune, without running it."""
    return cli._session_from_args(
        parse("--workload", FACTORY, "--session", str(tmp_path / "s"), *argv)
    )


@pytest.fixture
def workers(monkeypatch):
    """Run worker jobs in this process; list the directory of each job.

    The Torch-TensorRT source provenance is fixed, so that edits to the
    checkout while a test runs do not count as a changed environment.
    """
    ran = []
    monkeypatch.setattr(
        _worker,
        "source_provenance",
        lambda package: {"commit": "0" * 40, "uncommitted_sha256": None},
    )

    def execute(self, request, directory):
        ran.append(directory.relative_to(self.directory).as_posix())
        try:
            result = _worker.run_job(request)
        except Exception as error:  # noqa: BLE001 - like the worker's own handler
            result = {"status": "failed", "stage": "setup", "error": str(error)}
        _worker.write_json(Path(request["result_path"]), result)
        return result

    monkeypatch.setattr(TuningSession, "_run_worker_process", execute)
    return ran


# ------------------------------------------------------------ arguments


def test_run_defaults(tmp_path):
    args = parse("--workload", "w.py:f", "--session", "s")
    assert (args.workload, args.session, args.args) == ("w.py:f", "s", {})
    assert (args.planner, args.provider, args.llm_model) == (
        "rule-based",
        "claude",
        None,
    )
    assert args.accuracy == [] and args.baseline is None and args.guidance is None

    session = build(tmp_path)
    assert session.workload_factory.kwargs == {}
    assert session.directory == (tmp_path / "s").resolve()
    assert isinstance(session.planner, RuleBasedPlanner)
    # The other defaults are those of the Python API.
    default = TuningSession(session.workload_factory, directory=tmp_path / "s")
    for name in (
        "goal",
        "limits",
        "budget",
        "noise_control",
        "final_check_pairs",
        "baseline_changes",
        "guidance",
    ):
        assert getattr(session, name) == getattr(default, name), name


def test_run_options_reach_the_session(tmp_path):
    guidance = tmp_path / "guidance.md"
    guidance.write_text("# Try the cheap options first\n")
    session = build(
        tmp_path,
        "--planner=llm",
        "--provider=codex",
        "--llm-model=some-model",
        "--metric=throughput",
        "--target=500",
        "--min-improvement=0.05",
        "--max-trials=3",
        "--max-time-s=60",
        "--patience=2",
        "--trial-timeout-s=30",
        "--prepare-timeout-s=600",
        "--max-memory-mb=1024",
        "--guidance",
        str(guidance),
        "--noise-runs=1",
        "--head-to-head-pairs=4",
        "--final-check-pairs=3",
        "--quiet-gpu-s=0",
        "--max-spread=0.1",
        "--retries=0",
    )
    assert isinstance(session.planner, LLMPlanner)
    assert (session.planner.provider, session.planner.model) == (
        "codex",
        "some-model",
    )
    assert session.goal == Goal(metric="throughput", target=500, min_improvement=0.05)
    assert session.limits.max_memory_mb == 1024
    assert session.budget == Budget(
        max_trials=3,
        max_time_s=60,
        patience=2,
        trial_timeout_s=30,
        prepare_timeout_s=600,
    )
    assert session.noise_control == NoiseControl(
        noise_runs=1, head_to_head_pairs=4, quiet_gpu_s=0, retries=0, max_spread=0.1
    )
    assert session.final_check_pairs == 3
    assert session.guidance == "# Try the cheap options first\n"


def test_accuracy_limits_repeat(tmp_path):
    session = build(
        tmp_path,
        "--accuracy",
        "max_abs_error",
        "<=",
        "0.01",
        "--accuracy",
        "cosine_similarity",
        ">=",
        "0.999",
        # A later limit of the same metric replaces the earlier one.
        "--accuracy",
        "max_abs_error",
        "<=",
        "1e-3",
    )
    assert session.limits.accuracy == {
        "max_abs_error": ("<=", 1e-3),
        "cosine_similarity": (">=", 0.999),
    }


def test_json_options(tmp_path):
    session = build(
        tmp_path,
        "--args",
        '{"milliseconds": 4.0, "broken": ["workload.variant=b"]}',
        "--baseline",
        '{"engine.precision": "fp16"}',
    )
    assert session.workload_factory.kwargs == {
        "milliseconds": 4.0,
        "broken": ["workload.variant=b"],
    }
    assert session.baseline_changes == {"engine.precision": "fp16"}


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--args", "{milliseconds: 4}"], "argument --args: invalid JSON"),
        (["--args", "[1, 2]"], "argument --args: expected a JSON object"),
        (["--baseline", '"fp16"'], "argument --baseline: expected a JSON object"),
        (["--accuracy", "max_abs_error", "<=", "small"], "needs a number"),
        (["--accuracy", "max_abs_error", "~", "0.1"], "comparator"),
        (["--max-trials", "-1"], "max_trials must be a non-negative integer"),
        (["--guidance", "/nonexistent/guidance.md"], "guidance.md"),
    ],
)
def test_bad_options_are_usage_errors(tmp_path, capsys, argv, message):
    with pytest.raises(SystemExit) as raised:
        cli.main(
            ["run", "--workload", FACTORY, "--session", str(tmp_path / "s"), *argv]
        )
    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "python -m torch_tensorrt.autotune run: error: " in error
    assert message in error
    assert not (tmp_path / "s").exists()


def test_run_needs_a_workload_and_a_session(capsys):
    for argv in (["--session", "s"], ["--workload", FACTORY]):
        with pytest.raises(SystemExit) as raised:
            cli.main(["run", *argv])
        assert raised.value.code == 2
        assert "the following arguments are required" in capsys.readouterr().err


# ------------------------------------------------------- factory paths


def test_a_module_factory_is_kept(tmp_path):
    factory = "my_package.workloads:create_workload"
    assert build(tmp_path, "--workload", factory).workload_factory.factory == factory


def test_run_then_status_and_report(tmp_path, workers, capsys):
    directory = str(tmp_path / "s")
    argv = ["run", "--workload", FACTORY, "--args", FASTER_FP16]
    argv += ["--session", directory, "--max-trials=1", *QUICK]
    assert cli.main(argv) == 0
    assert "# Auto-Tuner report" in capsys.readouterr().out
    assert "trials/0001" in workers and (tmp_path / "s/artifact").is_dir()
    session_json = json.loads((tmp_path / "s/session.json").read_text())
    assert session_json["workload_factory"] == {
        "factory": FACTORY,
        "kwargs": {"speedups": {"engine.precision=fp16": 0.5}},
    }
    assert session_json["planner"] == {"kind": "rule-based"}

    assert cli.main(["status", directory]) == 0
    status = capsys.readouterr().out
    assert "0001   ok" in status and "accepted" in status
    assert '{"engine.precision": "fp16"}' in status and "finished; report" in status
    assert cli.main(["report", directory]) == 0
    report = capsys.readouterr().out
    assert report.startswith("# Auto-Tuner report")
    assert report == (tmp_path / "s/report.md").read_text() + "\n"

    # Running the command again, with a larger budget, resumes the session:
    # the recorded trial is reused, not run again.
    workers.clear()
    argv[argv.index("--max-trials=1")] = "--max-trials=2"
    assert cli.main(argv) == 0
    assert "trials/0001" not in workers and "trials/0002" in workers
    assert cli.main(["status", directory]) == 0
    assert "0002 " in capsys.readouterr().out


def test_rerun_with_a_different_workload_is_refused(tmp_path, workers, capsys):
    argv = ["run", "--workload", FACTORY, "--session", str(tmp_path / "s")]
    assert cli.main([*argv, "--max-trials=0", *QUICK]) == 0
    capsys.readouterr()
    # A refusal is a message and exit status 1, not a traceback.
    assert cli.main([*argv, "--args", FASTER_FP16, "--max-trials=0", *QUICK]) == 1
    assert "holds a different session" in capsys.readouterr().err


def test_status_shows_where_the_planner_stopped(tmp_path, workers, capsys):
    directory = tmp_path / "s"
    argv = ["run", "--workload", FACTORY, "--session", str(directory)]
    assert cli.main([*argv, "--max-trials=0", *QUICK]) == 0
    # The trial at which the planner stopped has its context and proposal,
    # but no trial.json: nothing was built or measured.
    stop = directory / "trials/0001"
    stop.mkdir(parents=True)
    (stop / "proposal.json").write_text(
        json.dumps({"changes": {}, "rationale": "nothing left to try", "stop": True})
    )
    capsys.readouterr()
    assert cli.main(["status", str(directory)]) == 0
    status = capsys.readouterr().out
    assert "stopped" in status and "the planner stopped: nothing left to try" in status
    assert "pending" not in status


def test_python_dash_m_runs_the_command_line():
    """``python -m torch_tensorrt.autotune`` runs :func:`main` in a process of its own."""
    run = subprocess.run(
        [sys.executable, "-P", "-m", "torch_tensorrt.autotune", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=300,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    assert "{run,status,report,resume}" in run.stdout
