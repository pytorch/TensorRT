# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior checks for the CI test-suite runner, without installing dependencies."""

import subprocess

import pytest

from . import runner


@pytest.mark.unit
@pytest.mark.parametrize("step", ["executorch", "cuda-tile", "hub", "cuda-core", "mpi"])
@pytest.mark.parametrize("failure_index", [None, 0, 1])
def test_required_setup_failures_stop_the_suite(
    monkeypatch, tmp_path, capsys, step, failure_index
) -> None:
    """ExecuTorch and CUDA Tile setup failures stop the suite; others warn and continue.

    Check failures in either setup command, including a post-install validation, without
    installing dependencies or depending on the runner's source spelling.
    """
    monkeypatch.setenv("RUNNER_TEST_RESULTS_DIR", str(tmp_path))
    setup_commands = [["setup-install"], ["setup-validate"]]
    monkeypatch.setattr(
        runner,
        "_setup_commands",
        lambda name: [(argv, runner.REPO_ROOT) for argv in setup_commands],
    )
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        failed = failure_index is not None and argv == setup_commands[failure_index]
        return subprocess.CompletedProcess(argv, 7 if failed else 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    suite = runner.Suite(
        name="setup-policy",
        tier="l0",
        lanes=("fast",),
        setup=(step,),
        follow=(("follow-command",),),
    )
    fatal = failure_index is not None and step in {"executorch", "cuda-tile"}
    assert runner.run_suite(suite, "standard") == (7 if fatal else 0)
    output = capsys.readouterr().out
    if fatal:
        assert calls == setup_commands[: failure_index + 1]
        assert f"::error::setup step {step!r} exited 7" in output
        assert "::warning::setup step" not in output
    else:
        assert calls[:2] == setup_commands
        assert "pytest" in calls[2]
        assert calls[3][-1] == "follow-command"
        assert len(calls) == 4
        assert "::error::setup step" not in output
        if failure_index is not None:
            assert f"::warning::setup step {step!r} exited 7" in output
        else:
            assert "::warning::setup step" not in output
