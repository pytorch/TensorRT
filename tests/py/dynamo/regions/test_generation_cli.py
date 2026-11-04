# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive skills use the same saved tasks and evaluator as compilation."""

import json
import os
import shutil
import sys
from argparse import Namespace
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from filelock import FileLock

from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo.region_compilation._generation import __main__ as cli
from torch_tensorrt.dynamo.region_compilation._generation import _factory
from torch_tensorrt.dynamo.region_compilation._generation._backends import get_backend
from torch_tensorrt.dynamo.region_compilation._generation._instructions import (
    load_instructions,
)
from torch_tensorrt.region import SearchPolicy

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Kernel generation requires a POSIX host"
)

BUDGET = {
    "max_attempts": 3,
    "timeout_s": 900,
    "candidate_timeout_s": 300,
    "author_timeout_s": 90,
    "max_context_requests": 2,
    "use_cache": True,
}
VALIDATION = {
    "num_tests": 3,
    "benchmark_iters": 20,
    "benchmark_repeats": 5,
    "check_nonfinite": True,
    "rtol": None,
    "atol": None,
}
COMPILE_OPTIONS = {
    "disable_tf32": True,
    "optimization_level": 5,
    "use_fp32_acc": False,
    "tiling_optimization_level": "none",
    "version_compatible": False,
}


def _write(path, value):
    path.write_text(json.dumps(value))


def _read(path):
    return json.loads(path.read_text())


@pytest.fixture
def run(tmp_path):
    """A complete run as the controller saves it, in a task cache directory."""
    run = tmp_path / "cache" / "task-hash" / "run-1"
    run.mkdir(parents=True)
    (run / "reference.pt").touch()
    specs = [{"shape": [7], "dtype": "float32", "device": "cuda:0"}]
    instructions = load_instructions(
        "triton", get_backend("triton").instruction_resources
    )
    task = {
        "version": _factory._VERSION,
        "reference": "def forward(self, arg0):\n    return (arg0,)",
        "constants": {},
        "inputs": specs,
        "outputs": specs,
        "backend": "triton",
        "backend_version": "1",
        "instructions_sha256": instructions["sha256"],
        "proposal_schema_sha256": "schema-digest",
        "objective": "candidate",
        "min_speedup": 1.0,
        "search": asdict(SearchPolicy()),
        "diagnostics": False,
        "analysis": {"operations": [], "boundary_bytes": 56},
        "target": {"name": "fixture GPU"},
        "validation": VALIDATION,
        "budget": BUDGET,
        "compile_options": COMPILE_OPTIONS,
    }
    template = {
        "reference_path": "/where/it/was/generated/reference.pt",
        "backend": "triton",
        "backend_version": "1",
        "proposal_schema_sha256": "schema-digest",
        "inputs": specs,
        "outputs": specs,
        "objective": "candidate",
        "diagnostics": False,
        **VALIDATION,
        "validation_seed": 7,
        "validation_phase": "search",
        "compile_options": COMPILE_OPTIONS,
    }
    _write(run / "task.json", task)
    _write(run / "evaluation-template.json", template)
    _write(run / "instructions.json", instructions)
    return run


def _update_task(run, **changes):
    task = _read(run / "task.json")
    task.update(changes)
    _write(run / "task.json", task)
    return task


def _another_run(run, name="run-2"):
    other = run.parent / name
    other.mkdir()
    for artifact in ("task.json", "reference.pt"):
        shutil.copy(run / artifact, other / artifact)
    return other


@pytest.fixture
def gpu(monkeypatch, tmp_path):
    """Record GPU lock requests without querying a device."""
    requests = []

    def lock(root, device):
        requests.append((root, device))
        return FileLock(str(tmp_path / "gpu.lock"))

    monkeypatch.setattr(cli, "gpu_lock", lock)
    return requests


@pytest.fixture
def candidate(run):
    path = run / "candidate.json"
    _write(path, {"source": "kernel source"})
    return path


def test_inspect_resolves_attempt_and_accepted_manifest(run, monkeypatch, capsys):
    attempt = run / "attempt-1"
    attempt.mkdir()
    manifest = run.parent / "accepted.json"
    _write(manifest, {"run": str(run)})
    assert cli._run_directory(manifest) == run
    monkeypatch.setattr(sys, "argv", ["generation", "inspect", str(attempt)])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["directory"] == str(run)
    assert result["task.json"]["backend"] == "triton"


@pytest.mark.parametrize("manifests", [("accepted", "checkpoint"), ("checkpoint",), ()])
def test_task_cache_directory_resolves_to_its_recorded_run(run, manifests):
    newest = _another_run(run, "run-newest")
    checkpointed = _another_run(run, "run-checkpointed")
    os.utime(newest, (2e9, 2e9))
    # Manifests may name the run where the cache was created, not where it is.
    if "accepted" in manifests:
        _write(run.parent / "accepted.json", {"run": f"/moved/task-hash/{run.name}"})
    if "checkpoint" in manifests:
        state = {"run": str(checkpointed)}
        _write(run.parent / "checkpoint.json", {"state": state, "sha256": "x"})
    expected = {
        ("accepted", "checkpoint"): run,
        ("checkpoint",): checkpointed,
        (): newest,
    }[manifests]
    assert cli._run_directory(run.parent) == expected


def test_task_cache_without_runs_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="task cache directory"):
        cli._run_directory(tmp_path)


def test_inspect_shows_the_checkpoint_only_for_its_run(run, monkeypatch, capsys):
    other = _another_run(run)
    state = {"run": str(other), "history": [], "finalists": [], "pending": []}
    _write(run.parent / "checkpoint.json", {"state": state, "sha256": "fixture"})
    shown = {}
    for directory in (run, other):
        monkeypatch.setattr(sys, "argv", ["generation", "inspect", str(directory)])
        cli.main()
        shown[directory] = json.loads(capsys.readouterr().out)
    assert "checkpoint.json" not in shown[run]
    assert shown[other]["checkpoint.json"]["state"] == state


def test_inspect_checkpoint_shows_search_progress(run, monkeypatch, capsys):
    checkpoint = run.parent / "checkpoint.json"
    state = {"run": str(run), "history": [], "finalists": [], "pending": []}
    _write(checkpoint, {"state": state, "sha256": "fixture"})
    summary = {"evaluations": 5, "stop_reason": "max_candidates"}
    _write(run / "search.json", summary)
    monkeypatch.setattr(sys, "argv", ["generation", "inspect", str(checkpoint)])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["directory"] == str(run)
    assert result["checkpoint.json"]["state"] == state
    assert result["search.json"] == summary


@pytest.mark.parametrize(("timeout", "bound"), [(30, 30), (None, 300)])
def test_cli_evaluate_builds_a_fresh_holdout_request_from_the_template(
    run, monkeypatch, gpu, candidate, timeout, bound
):
    requests = []

    def evaluate(request, directory, timeout):
        requests.append(request)
        assert 0 < timeout <= bound
        return {"status": "passed", "latency_ms": 0.1}

    monkeypatch.setattr(_factory, "_evaluate", evaluate)
    result = cli._evaluate(Namespace(candidate=candidate, timeout=timeout), run)
    (request,) = requests
    template = _read(run / "evaluation-template.json")
    assert request.pop("candidate") == {"source": "kernel source"}
    assert request.pop("reference_path") == str(run / "reference.pt")
    assert request.pop("validation_phase") == "holdout"
    seed = request.pop("validation_seed")
    assert type(seed) is int and seed != template["validation_seed"]
    for name in ("candidate", "reference_path", "validation_phase", "validation_seed"):
        template.pop(name, None)
    assert request == template
    assert gpu == [(run.parent.parent, "cuda:0")]
    assert result["report"]["meets_objective"]
    assert _read(Path(result["directory"]) / "feedback.json") == result["report"]


@pytest.mark.parametrize("wait", ["lock_timeout", "acquired_too_late"])
def test_cli_evaluate_reports_budget_when_the_gpu_wait_leaves_no_time(
    run, monkeypatch, gpu, candidate, tmp_path, wait
):
    def unexpected_evaluation(*args):
        pytest.fail("No evaluation fits in the remaining time")

    monkeypatch.setattr(_factory, "_evaluate", unexpected_evaluation)
    held = FileLock(str(tmp_path / "gpu.lock"))
    if wait == "lock_timeout":
        held.acquire()
    else:
        clock = iter((0.0, 31.0))
        monkeypatch.setattr(cli, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    try:
        result = cli._evaluate(Namespace(candidate=candidate, timeout=0.2), run)
    finally:
        held.release()
    assert result["report"]["status"] == "error"
    assert result["report"]["stage"] == "budget"
    assert not result["report"]["meets_objective"]


@pytest.mark.parametrize("waited,stage", [(0.0, "timeout"), (20.0, "budget")])
def test_cli_evaluate_blames_a_timeout_on_the_gpu_wait_that_shortened_it(
    run, monkeypatch, gpu, candidate, waited, stage
):
    slices = []

    def evaluation(request, directory, timeout):
        slices.append(timeout)
        return {"status": "error", "stage": "timeout", "error": "Evaluator exceeded"}

    monkeypatch.setattr(_factory, "_evaluate", evaluation)
    clock = iter((0.0, waited))
    monkeypatch.setattr(cli, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    result = cli._evaluate(Namespace(candidate=candidate, timeout=30.0), run)
    assert slices == [30.0 - waited]
    assert result["report"]["stage"] == stage


def test_cli_evaluate_failure_has_nonzero_exit(run, monkeypatch, capsys, gpu):
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
    run, monkeypatch, capsys, gpu
):
    _update_task(run, objective="eager", min_speedup=1.05)
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


@pytest.mark.parametrize("command", ["evaluate", "generate"])
def test_replay_rejects_another_task_format(run, monkeypatch, capsys, command):
    _update_task(run, version=_factory._VERSION - 1)
    arguments = (
        ["--candidate", str(run / "missing.json")] if command == "evaluate" else []
    )
    monkeypatch.setattr(sys, "argv", ["generation", command, str(run), *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert "compile the region again" in capsys.readouterr().err


def test_instruction_command_prints_guidance_and_schema_without_author_or_gpu(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        sys, "argv", ["generation", "instructions", "--backend", "cuda_cpp"]
    )
    cli.main()
    guidance, schema = capsys.readouterr().out.split("\nPROPOSAL SCHEMA:\n")
    assert "__global__" in guidance
    assert "INSTRUCTION RESOURCE: references/cuda_cpp.md" in guidance
    assert "INSTRUCTION RESOURCE: references/triton.md" not in guidance
    assert json.loads(schema) == get_backend("cuda_cpp").proposal_schema


def _arguments(**overrides):
    values = dict(
        provider="claude",
        backend=None,
        model=None,
        instructions=None,
        instruction_file=[],
        cache_dir=None,
        objective=None,
        min_speedup=None,
    )
    values.update(overrides)
    return Namespace(**values)


def test_replay_preserves_saved_search_budget_and_instruction_snapshots(run):
    task = _update_task(
        run,
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
    _write(
        run / "instructions.json",
        {"resources": [{"origin": "file", "text": "Prefer sequential loads."}]},
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
    assert {name: getattr(config, name) for name in BUDGET} == BUDGET
    assert {name: getattr(config, name) for name in VALIDATION} == VALIDATION
    assert config.cache_dir == run.parent.parent


def test_replay_overrides_can_replace_saved_options(run):
    task = _update_task(
        run,
        search={**asdict(SearchPolicy()), "resume": True, "tuning": {"num_warps": [4]}},
        diagnostics=True,
    )
    budget = {
        "max_attempts": 2,
        "timeout_s": 60.0,
        "candidate_timeout_s": 30.0,
        "author_timeout_s": 10.0,
        "max_context_requests": 0,
        "use_cache": False,
    }
    config = cli._generation_config(
        _arguments(
            resume=False, diagnostics=False, tuning={}, max_candidates=7, **budget
        ),
        task,
        run,
    )
    assert config.search.max_candidates == 7
    assert config.search.resume is False
    assert config.search.tuning == {}
    assert config.diagnostics is False
    assert {name: getattr(config, name) for name in budget} == budget
    # Replay reuses the run's own cache root, so its accepted kernel is found.
    assert config.cache_dir == run.parent.parent


def test_resume_respects_explicit_cache_directory(run, tmp_path):
    destination = tmp_path / "another-cache"
    config = cli._generation_config(
        _arguments(resume=True, cache_dir=destination),
        _read(run / "task.json"),
        run,
    )
    assert config.cache_dir == destination


def test_cli_output_option_writes_exactly_the_json_result(
    run, monkeypatch, capsys, tmp_path
):
    destination = tmp_path / "inspect.json"
    monkeypatch.setattr(
        sys, "argv", ["generation", "--output", str(destination), "inspect", str(run)]
    )
    cli.main()
    printed = json.loads(capsys.readouterr().out)
    assert json.loads(destination.read_text()) == printed
    assert printed["directory"] == str(run)


def test_generate_cli_parses_overrides_without_execution(run, monkeypatch, capsys):
    configs = []

    def generate(args, directory):
        configs.append(
            cli._generation_config(args, cli._read_task(directory), directory)
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
            "--attempts",
            "5",
            "--timeout",
            "1200",
            "--candidate-timeout",
            "240",
            "--author-timeout",
            "60",
            "--max-context-requests",
            "1",
            "--no-cache",
        ],
    )
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "configured"
    (config,) = configs
    assert config.search.batch_size == 3
    assert config.search.max_candidates == 15
    assert config.search.max_no_improvement == 6
    assert config.search.max_variants == 4
    assert config.search.finalists == 2
    assert config.search.resume is True
    assert config.search.tuning["num_warps"] == (4, 8)
    assert config.diagnostics is True
    assert config.max_attempts == 5
    assert config.timeout_s == 1200
    assert config.candidate_timeout_s == 240
    assert config.author_timeout_s == 60
    assert config.max_context_requests == 1
    assert config.use_cache is False


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


@pytest.mark.parametrize(
    "search", [{**asdict(SearchPolicy()), "unknown": 1}, ["not", "a", "policy"]]
)
def test_malformed_saved_search_policy_fails_before_execution(
    run, monkeypatch, capsys, search
):
    _update_task(run, search=search)
    monkeypatch.setattr(sys, "argv", ["generation", "generate", str(run)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert "Invalid generation settings" in capsys.readouterr().err


def _saved_bundle(run):
    source = run / "custom-guide.md"
    source.write_text("Use coalesced loads.")
    bundle = load_instructions("triton", instruction_files=(source,))
    _write(run / "instructions.json", bundle)
    source.unlink()
    return _update_task(run, instructions_sha256=bundle["sha256"]), bundle


@pytest.mark.parametrize("cache_hit", [False, True])
def test_replay_passes_exact_guidance_and_compile_options_to_the_controller(
    run, monkeypatch, cache_hit
):
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
    calls = []

    def generate(reference, config, *, region_id, settings, instruction_bundle):
        calls.append((settings, instruction_bundle))
        replay = run.parent / "run-replay"
        replay.mkdir()
        _write(replay / "cache_check.json", {"status": "passed", "check": "cache"})
        if not cache_hit:
            _write(replay / "search.json", {"stop_reason": "attempt_limit"})
            _write(replay / "accepted.json", {"validation": {"status": "passed"}})
        return None, replay

    monkeypatch.setattr(_factory, "_generate", generate)
    result = cli._generate(_arguments(), run)
    (settings, instruction_bundle), *_ = calls
    assert instruction_bundle == bundle
    assert isinstance(settings, CompilationSettings)
    assert {name: getattr(settings, name) for name in COMPILE_OPTIONS} == (
        COMPILE_OPTIONS
    )
    assert result["status"] == "passed"
    assert result["cache_hit"] is cache_hit
    assert result["report"] == (
        {"status": "passed", "check": "cache"} if cache_hit else {"status": "passed"}
    )


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


def test_replay_missing_instruction_snapshot_fails_before_execution(
    run, monkeypatch, capsys
):

    _saved_bundle(run)
    (run / "instructions.json").unlink()

    def unexpected_execution(*args, **kwargs):
        pytest.fail("Replay must validate saved instructions before execution")

    monkeypatch.setattr(torch.cuda, "device", unexpected_execution)
    monkeypatch.setattr(_factory, "_generate", unexpected_execution)
    monkeypatch.setattr(sys, "argv", ["generation", "generate", str(run)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert "instructions.json is missing" in capsys.readouterr().err
