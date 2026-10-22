# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive skills use the same saved tasks and evaluator as compilation."""

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from torch_tensorrt.dynamo.region_compilation._generation import __main__ as cli
from torch_tensorrt.dynamo.region_compilation._generation import _factory


@pytest.fixture
def run(tmp_path):
    run = tmp_path / "cache" / "task-hash" / "run-1"
    run.mkdir(parents=True)
    (run / "reference.pt").touch()
    task = {
        "backend": "triton",
        "backend_version": "1",
        "proposal_schema_sha256": "schema-digest",
        "inputs": [{"shape": [7], "dtype": "float32", "device": "cuda:0"}],
        "outputs": [{"shape": [7], "dtype": "float32", "device": "cuda:0"}],
        "objective": "candidate",
        "validation": {"num_tests": 3, "benchmark_iters": 20, "check_nonfinite": True},
    }
    (run / "task.json").write_text(json.dumps(task))
    return run


def test_inspect_resolves_attempt_and_accepted_manifest(run, monkeypatch, capsys):
    attempt = run / "attempt-1"
    attempt.mkdir()
    manifest = run.parent / "accepted.json"
    manifest.write_text(json.dumps({"run": str(run)}))
    assert cli._run_directory(manifest) == run
    monkeypatch.setattr(sys, "argv", ["generation", "inspect", str(attempt)])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["directory"] == str(run)
    assert result["task.json"]["backend"] == "triton"


def test_cli_evaluate_uses_fresh_worker_request_and_persists_feedback(run, monkeypatch):
    candidate = run / "candidate.json"
    candidate.write_text(json.dumps({"source": "kernel source"}))
    requests = []

    def evaluate(request, directory, timeout):
        requests.append(request)
        assert 0 < timeout <= 30
        return {"status": "passed", "latency_ms": 0.1}

    monkeypatch.setattr(_factory, "_evaluate", evaluate)
    result = cli._evaluate(Namespace(candidate=candidate, timeout=30), run)
    assert requests[0]["reference_path"] == str(run / "reference.pt")
    assert requests[0]["validation_phase"] == "holdout"
    assert "validation_seed" not in requests[0]
    assert requests[0]["candidate"]["source"] == "kernel source"
    assert requests[0]["backend_version"] == "1"
    assert requests[0]["proposal_schema_sha256"] == "schema-digest"
    assert (
        json.loads((Path(result["directory"]) / "feedback.json").read_text())
        == result["report"]
    )


def test_cli_evaluate_failure_has_nonzero_exit(run, monkeypatch, capsys):
    candidate = run / "candidate.json"
    candidate.write_text("{}")
    monkeypatch.setattr(
        _factory,
        "_evaluate",
        lambda *args: {"status": "error", "stage": "correctness", "error": "Mismatch"},
    )
    monkeypatch.setattr(
        sys, "argv", ["generation", "evaluate", str(run), "--candidate", str(candidate)]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["report"]["stage"] == "correctness"


def test_cli_evaluate_rejects_correct_candidate_that_misses_speedup(
    run, monkeypatch, capsys
):
    task = json.loads((run / "task.json").read_text())
    task.update(objective="eager", min_speedup=1.05)
    (run / "task.json").write_text(json.dumps(task))
    candidate = run / "candidate.json"
    candidate.write_text("{}")
    monkeypatch.setattr(
        _factory,
        "_evaluate",
        lambda *args: {
            "status": "passed",
            "latency_ms": 1.0,
            "baseline_latency_ms": 1.01,
        },
    )
    monkeypatch.setattr(
        sys, "argv", ["generation", "evaluate", str(run), "--candidate", str(candidate)]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert not json.loads(capsys.readouterr().out)["report"]["meets_objective"]


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_cli_rejects_unbounded_evaluation_timeouts(run, timeout):
    with pytest.raises(ValueError, match="timeout"):
        cli._evaluate(Namespace(candidate=run / "missing.json", timeout=timeout), run)


def test_instruction_command_selects_backend_without_author_or_gpu(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["generation", "instructions", "--backend", "cuda_cpp"]
    )
    cli.main()
    guidance = capsys.readouterr().out
    assert "__global__" in guidance
    assert "INSTRUCTION RESOURCE: references/cuda_cpp.md" in guidance
    assert "INSTRUCTION RESOURCE: references/triton.md" not in guidance
