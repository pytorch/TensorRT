# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive skills use the same saved tasks and evaluator as compilation."""

import json
import sys
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from torch_tensorrt.dynamo.region_compilation._generation import __main__ as cli
from torch_tensorrt.dynamo.region_compilation._generation import _factory
from torch_tensorrt.dynamo.region_compilation._generation._instructions import (
    load_instructions,
)


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
    task = json.loads((run / "task.json").read_text())
    task["diagnostics"] = True
    (run / "task.json").write_text(json.dumps(task))
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
    assert requests[0]["diagnostics"] is True
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


def _arguments(**overrides):
    values = dict(
        provider="claude",
        backend=None,
        model=None,
        instructions=None,
        instruction_file=[],
        attempts=4,
        timeout=600,
        cache_dir=None,
        objective=None,
        min_speedup=None,
    )
    values.update(overrides)
    return Namespace(**values)


def test_replay_preserves_saved_search_and_instruction_snapshots(run):
    task = json.loads((run / "task.json").read_text())
    task.update(
        search={
            "batch_size": 2,
            "max_candidates": 12,
            "max_no_improvement": 5,
            "max_variants": 4,
            "finalists": 2,
            "resume": True,
            "tuning": {"num_warps": [4, 8]},
        },
        diagnostics=True,
    )
    (run / "instructions.json").write_text(
        json.dumps(
            {"resources": [{"origin": "file", "text": "Prefer sequential loads."}]}
        )
    )
    config = cli._generation_config(_arguments(), task, run)
    assert config.search.batch_size == 2
    assert config.search.max_candidates == 12
    assert config.search.max_no_improvement == 5
    assert config.search.max_variants == 4
    assert config.search.finalists == 2
    assert config.search.resume is True
    assert config.search.tuning["num_warps"] == (4, 8)
    assert config.diagnostics is True
    assert config.instructions == "Prefer sequential loads."
    assert config.max_attempts == 4
    assert config.timeout_s == 600
    assert config.cache_dir == run.parent.parent


def test_replay_search_overrides_can_disable_saved_options(run):
    task = json.loads((run / "task.json").read_text())
    task.update(
        search={"batch_size": 2, "resume": True, "tuning": {"num_warps": [4, 8]}},
        diagnostics=True,
    )
    config = cli._generation_config(
        _arguments(resume=False, diagnostics=False, tuning={}, max_candidates=7),
        task,
        run,
    )
    assert config.search.batch_size == 2
    assert config.search.max_candidates == 7
    assert config.search.resume is False
    assert config.search.tuning == {}
    assert config.diagnostics is False
    assert config.cache_dir is None


def test_resume_respects_explicit_cache_directory(run, tmp_path):
    destination = tmp_path / "another-cache"
    config = cli._generation_config(
        _arguments(resume=True, cache_dir=destination),
        cli._read(run / "task.json"),
        run,
    )
    assert config.cache_dir == destination


def test_inspect_checkpoint_shows_search_progress(run, monkeypatch, capsys):
    checkpoint = run.parent / "checkpoint.json"
    state = {"run": str(run), "history": [], "finalists": [], "pending": []}
    checkpoint.write_text(json.dumps({"state": state, "sha256": "fixture"}))
    summary = {"evaluations": 5, "stop_reason": "max_candidates"}
    (run / "search.json").write_text(json.dumps(summary))
    monkeypatch.setattr(sys, "argv", ["generation", "inspect", str(checkpoint)])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["directory"] == str(run)
    assert result["checkpoint.json"]["state"] == state
    assert result["search.json"] == summary


def test_generate_cli_parses_search_overrides_without_execution(
    run, monkeypatch, capsys
):
    configs = []

    def generate(args, directory):
        configs.append(
            cli._generation_config(args, cli._read(directory / "task.json"), directory)
        )
        return {"status": "configured"}

    monkeypatch.setattr(cli, "_generate", generate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generation",
            "generate",
            str(run),
            "--batch-size",
            "3",
            "--max-candidates",
            "15",
            "--max-no-improvement",
            "6",
            "--max-variants",
            "4",
            "--finalists",
            "2",
            "--resume",
            "--diagnostics",
            "--tuning",
            '{"num_warps":[4,8]}',
        ],
    )
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "configured"
    config = configs[0]
    assert config.search.batch_size == 3
    assert config.search.max_candidates == 15
    assert config.search.max_no_improvement == 6
    assert config.search.max_variants == 4
    assert config.search.finalists == 2
    assert config.search.resume is True
    assert config.search.tuning["num_warps"] == (4, 8)
    assert config.diagnostics is True


@pytest.mark.parametrize("value", ["not-json", "[]", '{"num_warps":4}'])
def test_generate_cli_rejects_malformed_tuning_before_execution(
    run, monkeypatch, value
):
    monkeypatch.setattr(
        sys, "argv", ["generation", "generate", str(run), "--tuning", value]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_replay_old_tasks_use_default_search_policy(run):
    config = cli._generation_config(_arguments(), cli._read(run / "task.json"), run)
    assert config.search.batch_size == 1
    assert config.search.max_candidates is None
    assert config.search.resume is False
    assert config.search.tuning == {}
    assert config.diagnostics is False


def _saved_bundle(run):
    source = run / "custom-guide.md"
    source.write_text("Use coalesced loads.")
    bundle = load_instructions("triton", instruction_files=(source,))
    (run / "instructions.json").write_text(json.dumps(bundle))
    source.unlink()
    task = cli._read(run / "task.json")
    task["instructions_sha256"] = bundle["sha256"]
    (run / "task.json").write_text(json.dumps(task))
    return task, bundle


def test_replay_passes_exact_instruction_bundle_to_controller(run, monkeypatch):
    from torch_tensorrt.dynamo.region_compilation._generation import _worker

    _, bundle = _saved_bundle(run)
    reference = torch.fx.symbolic_trace(torch.nn.Identity())
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: reference)
    monkeypatch.setattr(
        torch,
        "randn",
        lambda shape, **kwargs: torch.zeros(shape, dtype=kwargs["dtype"]),
    )
    monkeypatch.setattr(_worker, "_restore_metadata", lambda *args: None)
    captured = []

    def generate(reference, config, *, region_id, instruction_bundle):
        captured.append(instruction_bundle)
        (run / "accepted.json").write_text(
            json.dumps({"validation": {"status": "passed"}})
        )
        return None, run

    monkeypatch.setattr(_factory, "_generate", generate)
    result = cli._generate(_arguments(), run)
    assert result["status"] == "passed"
    assert captured == [bundle]


@pytest.mark.parametrize(
    "overrides",
    [{"backend": "cutile"}, {"instructions": ""}, {"instruction_file": ["new.md"]}],
)
def test_explicit_backend_or_guidance_uses_new_instruction_bundle(run, overrides):
    task, _ = _saved_bundle(run)
    assert cli._saved_instruction_bundle(_arguments(**overrides), task, run) is None


def test_replay_rejects_instruction_bundle_from_a_different_task(run):
    task, _ = _saved_bundle(run)
    task["instructions_sha256"] = "different-task"
    with pytest.raises(ValueError, match="task instruction digest"):
        cli._saved_instruction_bundle(_arguments(), task, run)
