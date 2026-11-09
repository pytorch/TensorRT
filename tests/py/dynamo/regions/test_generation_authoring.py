# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""The author loop: protocol repair, rejected proposals, turn budgets, retries,
shared-memory fitting and code-generation options, without agent or GPU access."""

import json
import logging
import os
import time
from types import SimpleNamespace

import pytest
from filelock import FileLock

from torch_tensorrt.dynamo.region_compilation._generation import (
    _backends,
    _factory,
    _providers,
    _worker,
)
from torch_tensorrt.dynamo.region_compilation._generation._backends import (
    SharedMemoryLimitError,
    get_backend,
    registration_for,
)
from torch_tensorrt.dynamo.region_compilation._generation._candidate import (
    _halvable_tiles,
    _halve_largest,
)
from torch_tensorrt.region import KernelGenerator, RegionError

from ._generation_utils import _candidate, _generate, _passed, _proposal

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Kernel generation requires a POSIX host"
)


def _run(factory):
    return next(factory.root.glob("*/run-*"))


def _history(factory):
    return json.loads((_run(factory) / "history.json").read_text())


def _null_reply():
    return {"action": "propose", "candidates": None, "resource": None, "offset": None}


def _clocked(monkeypatch):
    """Give the controller a clock that the test advances; sleeps return at once."""
    clock = [0.0]
    monkeypatch.setattr(
        _factory,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda seconds: None),
    )
    return clock


def _sleeps(monkeypatch, observe=lambda seconds: seconds):
    """Keep the controller's clock, but record ``observe`` instead of sleeping."""
    observed = []
    monkeypatch.setattr(
        _factory,
        "time",
        SimpleNamespace(
            monotonic=time.monotonic,
            sleep=lambda seconds: observed.append(observe(seconds)),
        ),
    )
    return observed


# ---- Protocol repair ----


def test_null_candidates_are_repaired_within_the_round(factory):
    factory.author.side_effect = [_null_reply(), _proposal(_candidate("fixed"))]
    factory.evaluate.return_value = _passed(0.2)
    assert _generate(factory) == _candidate("fixed")
    assert factory.author.call_count == 2
    first, retry = (call.kwargs for call in factory.author.call_args_list)
    assert "PROTOCOL ERROR" not in first["prompt"]
    assert (
        "PROTOCOL ERROR: Proposal candidates must be a bounded batch" in retry["prompt"]
    )
    assert "but candidates is null" in retry["prompt"]
    # The repair turn may not read and must carry candidates.
    properties = retry["schema"]["properties"]
    assert properties["action"]["enum"] == ["propose", "stop"]
    assert properties["candidates"]["type"] == "array"
    assert properties["candidates"]["minItems"] == 1
    assert retry["workdir"].name == "retry-1"
    assert first["schema"]["properties"]["candidates"]["anyOf"][1] == {"type": "null"}
    factory.evaluate.assert_called_once()


def test_a_reply_the_provider_could_not_fit_to_the_schema_is_repaired(factory):
    factory.author.side_effect = [
        _providers.ProtocolError("claude could not produce a schema-valid reply: x"),
        _proposal(_candidate("fixed")),
    ]
    factory.evaluate.return_value = _passed(0.2)
    assert _generate(factory) == _candidate("fixed")
    assert "schema-valid reply" in factory.author.call_args.kwargs["prompt"]


@pytest.mark.parametrize("retries, calls", [(0, 1), (1, 2), (2, 3)])
def test_repeated_protocol_failures_name_the_defect_and_keep_the_reply(
    factory, retries, calls
):
    factory.author.return_value = _proposal(_candidate("one"), _candidate("two"))
    with pytest.raises(RegionError, match="but candidates holds 2 proposals"):
        _generate(factory, max_protocol_retries=retries)
    assert factory.author.call_count == calls
    history = _history(factory)
    assert [entry["result"]["stage"] for entry in history] == ["author"]
    rejected = history[0]["rejected_reply"]
    assert [item["kernel_name"] for item in rejected["candidates"]] == ["one", "two"]
    factory.evaluate.assert_not_called()


@pytest.mark.parametrize(
    "reply, defect",
    [
        ({"action": "propose", "resource": None}, "candidates is missing"),
        ({**_null_reply(), "candidates": []}, "candidates is empty"),
        ({**_null_reply(), "candidates": "x"}, "candidates is a str, not an array"),
        (
            {**_null_reply(), "candidates": [[]]},
            "candidates[0] is a list, not an object",
        ),
        ({"action": "explain"}, "action must be propose, read, diagnose, or stop"),
    ],
)
def test_protocol_defects_are_named(factory, reply, defect):
    factory.author.return_value = reply
    with pytest.raises(RegionError) as error:
        _generate(factory)
    assert defect in str(error.value)


@pytest.mark.parametrize("action", ["read", "diagnose"])
def test_a_request_without_reads_left_is_repaired_as_a_protocol_error(factory, action):
    # The turn's schema allows only propose and stop; an adapter that ignores
    # it gets a repair turn instead of losing the round.
    request = {"action": action, "candidates": None, "resource": "x", "offset": 0}
    factory.author.side_effect = [request, _proposal(_candidate("fixed"))]
    factory.evaluate.return_value = _passed(0.2)
    fixed = _generate(factory, max_context_requests=0, diagnostics=True)
    assert fixed == _candidate("fixed")
    first, retry = (call.kwargs for call in factory.author.call_args_list)
    assert first["schema"]["properties"]["action"]["enum"] == ["propose", "stop"]
    assert retry["workdir"].name == "retry-1"
    assert (
        "PROTOCOL ERROR: Author response action must be propose or stop when no "
        f"evidence reads remain, but it is {action!r}. That reply was not used"
    ) in retry["prompt"]
    assert "EVIDENCE RESULT" not in retry["prompt"]
    factory.evaluate.assert_called_once()


def test_a_read_on_a_protocol_retry_breaks_the_protocol_again(factory):
    # A repair turn may not read even with reads left; the round fails once
    # the retries are spent, and the next round sees the reply.
    read = {"action": "read", "candidates": None, "resource": "task.json", "offset": 0}
    factory.author.side_effect = [_null_reply(), read]
    with pytest.raises(
        RegionError,
        match="must be propose or stop on a protocol retry, but it is 'read'",
    ):
        _generate(factory)
    assert factory.author.call_count == 2
    history = _history(factory)
    assert [entry["result"]["stage"] for entry in history] == ["author"]
    assert history[0]["rejected_reply"] == read
    factory.evaluate.assert_not_called()


def test_turns_without_reads_left_use_the_propose_only_schema(factory):
    factory.author.return_value = _candidate("only")
    factory.evaluate.return_value = _passed(0.2)
    _generate(factory, max_context_requests=0)
    schema = factory.author.call_args.kwargs["schema"]
    assert schema["properties"]["candidates"]["type"] == "array"
    assert schema["properties"]["resource"] == {"type": "null"}


def _types(schema):
    if isinstance(schema, dict):
        yield schema.get("type")
        for value in schema.values():
            yield from _types(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _types(value)


@pytest.mark.parametrize("backend", ["triton", "cutile"])
def test_reply_schemas_have_no_union_type_lists(backend):
    # Strict structured-output validators reject "type": [..., "null"].
    proposal_schema = get_backend(backend).proposal_schema
    for schema in (
        proposal_schema,
        _factory._response_schema(proposal_schema, 2),
        _factory._response_schema(proposal_schema, 2, propose_only=True),
    ):
        assert not any(isinstance(kind, list) for kind in _types(schema))


# ---- Rejected proposals stay in the history ----


def test_a_rejected_source_is_shown_to_the_next_round(factory):
    broken = _candidate("broken")
    broken["source"] = broken["source"].replace(
        "import triton\n", "import triton\nimport os  # marker-7f3a\n"
    )
    factory.author.side_effect = [_proposal(broken), _proposal(_candidate("fixed"))]
    factory.evaluate.return_value = _passed(0.2)
    assert _generate(factory, max_attempts=2) == _candidate("fixed")
    second = factory.author.call_args.kwargs["prompt"]
    assert "rejected_proposal" in second and "marker-7f3a" in second
    assert "line 3: 'import os' is not allowed" in second
    history = _history(factory)
    assert history[0]["candidate"] is None
    assert history[0]["rejected_proposal"]["kernel_name"] == "broken"


def test_rejected_output_is_kept_as_bounded_json():
    huge = {
        "source": "x" * (2 * _factory._REJECTED_CHARS),
        "kernel_name": "short",
        "value": float("nan"),
        "grid": (1, 2),
        "other": object,
    }
    bounded = _factory._bounded(huge)
    assert bounded["source"] == "x" * _factory._REJECTED_CHARS + "\n[truncated]"
    assert bounded["kernel_name"] == "short" and bounded["grid"] == [1, 2]
    assert bounded["value"] is None and bounded["other"] == repr(object)
    json.dumps(bounded, allow_nan=False)


# ---- Turn budgets ----


def test_each_author_turn_gets_its_own_time_slice(factory, monkeypatch):
    clock = _clocked(monkeypatch)
    replies = iter(
        [
            {"action": "read", "candidates": None, "resource": "catalog", "offset": 0},
            {"action": "read", "candidates": None, "resource": "catalog", "offset": 0},
            _proposal(_candidate("late")),
        ]
    )

    def slow_author(**request):
        clock[0] += 40  # each turn takes 40 of its 50 seconds
        return next(replies)

    factory.author.side_effect = slow_author
    factory.evaluate.return_value = _passed(0.2)
    _generate(factory, timeout_s=3600, author_timeout_s=50, max_context_requests=2)
    budgets = [call.kwargs["timeout_s"] for call in factory.author.call_args_list]
    assert budgets == [50, 50, 50]


def test_the_default_budget_gives_round_one_a_full_author_turn(factory):
    config = KernelGenerator()
    assert (config.timeout_s, config.candidate_timeout_s) == (1800, 300)
    factory.author.return_value = _candidate("first")
    factory.evaluate.return_value = _passed(0.2)
    _generate(
        factory,
        timeout_s=config.timeout_s,
        candidate_timeout_s=config.candidate_timeout_s,
        author_timeout_s=config.author_timeout_s,
    )
    assert factory.author.call_args.kwargs["timeout_s"] == pytest.approx(
        config.author_timeout_s, abs=1
    )


def test_a_shortened_turn_that_times_out_continues_while_a_full_turn_fits(
    factory, monkeypatch
):
    clock = _clocked(monkeypatch)
    # timeout_s 1800, candidate_timeout_s 120: the search deadline is 1680, and
    # a round starting at 0 must stop authoring at 1440.
    turns = []

    def author(**kwargs):
        turns.append(kwargs["timeout_s"])
        if len(turns) == 1:
            clock[0] = 1400.0  # a slow evidence read
            return {
                "action": "read",
                "resource": "catalog",
                "candidates": None,
                "offset": 0,
            }
        if len(turns) == 2:
            clock[0] += kwargs["timeout_s"]  # the shortened turn runs out
            raise TimeoutError("author CLI timed out")
        return _candidate("second_round")

    factory.author.side_effect = author
    factory.evaluate.return_value = _passed(0.2)
    assert _generate(
        factory,
        timeout_s=1800,
        candidate_timeout_s=120,
        author_timeout_s=100,
        max_attempts=2,
    ) == _candidate("second_round")
    assert turns[1] == pytest.approx(40)
    # At 1440 the next round may author until 1560: a full 100 s turn.
    assert turns[2] == pytest.approx(100)
    assert _history(factory)[0]["result"]["stage"] == "author"


# ---- Provider and setup retries ----


def test_an_unavailable_model_api_is_asked_again_after_a_wait(factory, monkeypatch):
    waits = _sleeps(monkeypatch)
    factory.author.side_effect = [
        _providers.ProviderUnavailableError("The model API returned status 529", 529),
        _proposal(_candidate("after_wait")),
    ]
    factory.evaluate.return_value = _passed(0.2)
    assert _generate(factory) == _candidate("after_wait")
    assert waits == [_factory._UNAVAILABLE_BACKOFF_S[0]]
    assert factory.author.call_args.kwargs["workdir"].name == "unavailable-1"


def test_an_api_that_stays_unavailable_stops_the_search(factory, monkeypatch):
    monkeypatch.setattr(_factory, "_UNAVAILABLE_BACKOFF_S", (0.0,))
    _sleeps(monkeypatch)
    unavailable = _providers.ProviderUnavailableError("status 529", 529)
    factory.author.side_effect = [
        unavailable,
        unavailable,
        _proposal(_candidate("late")),
    ]
    with pytest.raises(RegionError, match="the provider failed"):
        _generate(factory, max_attempts=3)
    # One wait within the round, then the search stops.
    assert factory.author.call_count == 2
    assert [entry["result"]["stage"] for entry in _history(factory)] == ["provider"]


def test_a_setup_failure_is_evaluated_once_more_without_holding_the_gpu(
    factory, monkeypatch
):
    lock = FileLock(str(factory.root.parent / "gpu.lock"))
    monkeypatch.setattr(_factory, "gpu_lock", lambda root, device: lock)
    held = _sleeps(monkeypatch, lambda seconds: lock.is_locked)
    factory.author.return_value = _candidate("flaky_gpu")
    factory.evaluate.side_effect = [
        {"status": "error", "stage": "setup", "error": "CUDA out of memory"},
        _passed(0.2),
    ]
    assert _generate(factory) == _candidate("flaky_gpu")
    assert factory.evaluate.call_count == 2
    # The wait releases the GPU to other processes' evaluations.
    assert held == [False]
    run = _run(factory)
    failure = json.loads((run / "attempt-1" / "setup_failure.json").read_text())
    assert failure["stage"] == "setup"
    # The repeat does not use up a candidate evaluation.
    assert json.loads((run / "search.json").read_text())["evaluations"] == 1


def test_a_setup_failure_that_cannot_change_is_not_evaluated_again(factory):
    factory.author.return_value = _candidate("unpromotable")
    factory.evaluate.return_value = {
        "status": "error",
        "stage": "setup",
        "error": "The region's PyTorch reference could not be evaluated",
        "retryable": False,
    }
    with pytest.raises(RegionError, match="could not be evaluated"):
        _generate(factory)
    factory.evaluate.assert_called_once()
    assert not (_run(factory) / "attempt-1" / "setup_failure.json").exists()


# ---- Provider spend ----


def test_the_spend_that_author_turns_report_is_totaled_and_logged(factory, caplog):
    turns = iter(
        [
            (
                {"action": "read", "candidates": None, "resource": "x", "offset": 0},
                0.25,
            ),
            (_proposal(_candidate("priced")), 0.5),
        ]
    )

    def author(**kwargs):
        reply, cost = next(turns)
        # As the claude provider records what its CLI reports for a turn.
        spend = kwargs["workdir"] / _providers.SPEND_FILENAME
        spend.write_text(json.dumps({"total_cost_usd": cost}))
        return reply

    factory.author.side_effect = author
    factory.evaluate.return_value = _passed(0.2)
    with caplog.at_level(logging.INFO, logger=_factory.__name__):
        _generate(factory)
    search = json.loads((_run(factory) / "search.json").read_text())
    assert search["provider_cost_usd"] == 0.75
    assert search["priced_author_turns"] == 2
    assert (
        "Agent region activation: the provider reported a cost of $0.7500 for "
        "this search (author turns priced: 2)"
    ) in caplog.text


def test_no_spend_is_recorded_when_the_provider_reports_none(factory, caplog):
    factory.author.return_value = _candidate("unpriced")
    factory.evaluate.return_value = _passed(0.2)
    with caplog.at_level(logging.INFO, logger=_factory.__name__):
        _generate(factory)
    search = json.loads((_run(factory) / "search.json").read_text())
    assert not {"provider_cost_usd", "priced_author_turns"} & search.keys()
    assert "provider reported" not in caplog.text


# ---- Shared-memory fitting ----


def _fitted(original):
    variant = json.loads(json.dumps(original))
    variant["num_stages"] = 1
    variant["constants"][1]["value"] = 16
    return variant


def test_a_fitted_variant_is_recorded_validated_and_accepted(factory):
    original = _candidate("too_big")
    original["num_stages"] = 3
    fitted = _fitted(original)
    factory.author.return_value = original
    factory.evaluate.return_value = {
        **_passed(0.2),
        "fitted_candidate": fitted,
        "fit_log": ["num_stages 3 -> 1: 65536 -> 53248 bytes", "BLOCK 32 -> 16: ..."],
    }
    assert _generate(factory) == fitted
    holdout = factory.finalize.call_args.args[0]
    assert holdout["candidate"] == fitted and holdout["auto_fit"] is False
    run = _run(factory)
    history = _history(factory)
    assert history[0]["candidate"] == fitted
    result = history[0]["result"]
    assert "fitted_candidate" not in result and result["fit_log"]
    assert result["fitted_from"] == {
        "num_stages": 3,
        "constants": original["constants"],
    }
    accepted = json.loads((run / "accepted.json").read_text())
    assert accepted["candidate"] == fitted


def test_a_fitted_variant_with_other_source_is_refused(factory):
    original = _candidate("too_big")
    other = dict(_fitted(original), source=_candidate("other")["source"])
    factory.author.return_value = original
    factory.evaluate.return_value = {**_passed(0.2), "fitted_candidate": other}
    with pytest.raises(RegionError, match="invalid fitted candidate"):
        _generate(factory)


class _FitBackend:
    def __init__(self, fitted):
        self.fitted, self.calls = fitted, []

    def fit_shared_memory(self, candidate, inputs, outputs, directory, options, error):
        self.calls.append((candidate, options, error.required))
        return self.fitted, ["BLOCK 64 -> 32: 65536 -> 32768 bytes"]

    def validate_candidate(self, proposal, num_inputs, num_outputs):
        return dict(proposal)


def _overflow():
    error = _worker._EvaluationError("compile", "too much shared memory")
    error.__cause__ = SharedMemoryLimitError(
        "too much", required=65536, limit=49152, allocations=()
    )
    return error


@pytest.mark.parametrize(
    "request_changes, fits",
    [
        ({}, True),
        ({"auto_fit": False}, False),
        ({"validation_phase": "holdout"}, False),
    ],
)
def test_the_worker_fits_only_search_evaluations_that_ask_for_it(request_changes, fits):
    backend = _FitBackend({"kernel_name": "fitted"})
    request = {
        "auto_fit": True,
        "validation_phase": "search",
        "codegen": {"aligned_io": True},
        **request_changes,
    }
    record = {}
    fitted = _worker._fit_shared_memory(
        backend, request, {"kernel_name": "k"}, ([], []), None, _overflow(), record
    )
    assert (fitted is not None) == fits
    if fits:
        assert record == {
            "fit_log": ["BLOCK 64 -> 32: 65536 -> 32768 bytes"],
            "shared_memory": {
                "required_bytes": 65536,
                "limit_bytes": 49152,
                "largest_allocations": [],
            },
            "fitted_candidate": {"kernel_name": "fitted"},
        }
        assert backend.calls == [({"kernel_name": "k"}, {"aligned_io": True}, 65536)]


def test_the_worker_does_not_fit_other_compile_errors():
    backend = _FitBackend({"kernel_name": "fitted"})
    error = _worker._EvaluationError("compile", "syntax")
    error.__cause__ = ValueError("syntax")
    request = {"auto_fit": True, "validation_phase": "search"}
    assert (
        _worker._fit_shared_memory(backend, request, {}, ([], []), None, error, {})
        is None
    )
    assert backend.calls == []


def test_error_reports_name_the_fitted_variant(tmp_path):
    error = _worker._EvaluationError("correctness", "wrong")
    error.fit = {"fit_log": ["x"], "fitted_candidate": {"kernel_name": "f"}}
    report = _worker._error_report(error, tmp_path, "search")
    assert report["fitted_candidate"] == {"kernel_name": "f"}
    assert report["fit_log"] == ["x"]


SLIDING_WINDOW = """import triton
import triton.language as tl

@triton.jit
def swa(arg0, out0, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
        H: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % H
    mb = pid // H
    m0 = mb * BLOCK_M
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    lo = (m0 - 512) // BLOCK_N * BLOCK_N
    for start_n in range(lo, m0 + BLOCK_M, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
"""

CHUNKED = """import triton
import triton.language as tl

@triton.jit
def gdn(arg0, out0, BT: tl.constexpr, NC: tl.constexpr, BV: tl.constexpr, NV: tl.constexpr):
    pid = tl.program_id(0)
    vb = pid % NV
    r = tl.arange(0, BT)
    dv = vb * BV + tl.arange(0, BV)
    for c in range(NC):
        rows = c * BT + r
"""


def _tiled(source, **constants):
    return {
        "source": source,
        "kernel_name": "k",
        "grid": [8],
        "num_warps": 4,
        "num_stages": 2,
        "constants": [{"name": k, "value": v} for k, v in constants.items()],
    }


def test_fitting_keeps_grid_tiles_and_boundary_dimensions():
    tiles, log = _halvable_tiles(
        _tiled(SLIDING_WINDOW, BLOCK_M=64, BLOCK_N=64, D=256, H=4), {1, 4, 256, 2048}
    )
    # H * BLOCK_N == 256 is a dimension, but H is no loop bound.
    assert tiles == {"BLOCK_N": None}
    assert any("BLOCK_M kept: it scales tl.program_id" in line for line in log)
    assert any("D kept: it equals a boundary dimension" in line for line in log)


def test_fitting_doubles_a_tiles_trip_count_with_it():
    candidate = _tiled(CHUNKED, BT=32, NC=32, BV=32, NV=4)
    tiles, _ = _halvable_tiles(candidate, {1, 16, 128, 1024})
    assert tiles == {"BT": "NC"}
    variant, change, name = _halve_largest(candidate, tiles)
    assert name == "BT" and "NC 32 -> 64" in change
    assert {c["name"]: c["value"] for c in variant["constants"]} == {
        "BT": 16,
        "NC": 64,
        "BV": 32,
        "NV": 4,
    }
    assert variant["num_stages"] == 1


def test_fitting_never_halves_a_tile_below_16():
    candidate = _tiled(CHUNKED, BT=16, NC=64, BV=32, NV=4)
    assert _halve_largest(candidate, {"BT": "NC"}) is None


# ---- Code-generation and validation options ----


def test_aligned_io_is_hashed_and_reaches_both_registrations(factory):
    factory.author.return_value = _candidate("aligned")
    factory.evaluate.return_value = _passed(0.2)
    _generate(factory, aligned_io=True)
    _generate(factory, aligned_io=False)
    runs = sorted(factory.root.glob("*/run-*"))
    assert len({run.parent for run in runs}) == 2  # two cache keys
    for run in runs:
        task = json.loads((run / "task.json").read_text())
        template = json.loads((run / "evaluation-template.json").read_text())
        assert template["codegen"] == task["codegen"]
        assert template["auto_fit"] is True
    options = sorted(
        call.args[1]._options["aligned_io"] for call in factory.register.call_args_list
    )
    assert options == [False, True]
    request = factory.finalize.call_args.args[0]
    assert request["auto_fit"] is False and "codegen" in request


def test_fp32_matmul_precision_is_hashed_and_reaches_every_evaluation(factory):
    assert KernelGenerator().fp32_matmul_precision == "highest"
    assert KernelGenerator(fp32_matmul_precision="high") != KernelGenerator()
    factory.author.return_value = _candidate("doubled")
    factory.evaluate.return_value = _passed(0.2)
    tasks = {}
    for precision in ("highest", "high"):
        _generate(factory, fp32_matmul_precision=precision)
        (run,) = (
            run
            for run in factory.root.glob("*/run-*")
            if run.parent.name not in tasks.values()
        )
        task = json.loads((run / "task.json").read_text())
        assert task["validation"]["fp32_matmul_precision"] == precision
        tasks[precision] = run.parent.name
    # A different precision is a different cache key, and every worker request
    # carries it.
    assert tasks["highest"] != tasks["high"]
    requests = [
        call.args[0]
        for mock in (factory.evaluate, factory.finalize)
        for call in mock.call_args_list
    ]
    assert sorted(request["fp32_matmul_precision"] for request in requests) == [
        "high",
        "high",
        "highest",
        "highest",
    ]


def test_registration_passes_options_only_to_backends_that_name_them():
    calls = []

    class Old:
        def make_registration(self, candidate, inputs, outputs, directory):
            calls.append("old")

    class New:
        def make_registration(
            self, candidate, inputs, outputs, directory, options=None
        ):
            calls.append(options)

    class Catchall:
        def make_registration(self, *args, **kwargs):
            calls.append(kwargs)

    for backend in (Old(), New(), Catchall()):
        registration_for(backend, {}, [], [], None, {"aligned_io": True})
    assert calls == ["old", {"aligned_io": True}, {}]
    bound = _backends.with_options(New(), {"aligned_io": True})
    bound.make_registration({}, [], [], None)
    assert calls[-1] == {"aligned_io": True}
