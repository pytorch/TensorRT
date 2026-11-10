# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""The rule-based and LLM planners, the LLM's prompt and reply, and proposal checks."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from torch_tensorrt.autotune import (
    Knob,
    LLMPlanner,
    Proposal,
    RuleBasedPlanner,
    TuningSession,
    WorkloadFactory,
)
from torch_tensorrt.autotune._knobs import apply_changes, knobs_by_name
from torch_tensorrt.autotune._planner import (
    _proposal_from_reply,
    check_proposal,
    planner_from_json,
    proposal_schema,
)
from torch_tensorrt.dynamo.region_compilation._generation import _providers

from ._workloads import FAKE_KNOBS, KERNEL_KNOB

KNOBS = knobs_by_name(FAKE_KNOBS)
FAKE_FACTORY = f"{Path(__file__).with_name('_workloads.py')}:create_fake"


def planner_context(
    config=None, hints=(), history=(), baseline=None, tried=(), knobs=KNOBS
):
    """A planner context for the fake knobs, with ``config`` as the best config."""
    config = config or {}
    return {
        "goal": {"metric": "latency_ms", "min_improvement": 0.02, "target": None},
        "baseline": {"config": baseline or {}},
        "best": {"config": config, "bottleneck_hints": list(hints)},
        "history": list(history),
        "knobs": [
            {**knob.to_json(), "current": config.get(name, knob.default)}
            for name, knob in knobs.items()
        ],
        "tried": [baseline or {}, config, *tried],
    }


def hint(name, detail="measured"):
    """A bottleneck hint as the profiler reports it."""
    return {"name": name, "strength": 0.9, "detail": detail}


def test_rule_based_planner_tries_what_relieves_the_strongest_hint(tmp_path):
    proposal = RuleBasedPlanner().propose(
        planner_context(hints=[hint("precision"), hint("launch_overhead")]), tmp_path
    )
    assert proposal.changes == {"engine.precision": "fp16"}
    assert proposal.rationale.startswith("precision: measured")
    # Without a hint, runtime options come first.
    plain = RuleBasedPlanner().propose(planner_context(), tmp_path)
    assert plain.changes == {"runtime.cudagraphs": True}


def test_rule_based_planner_skips_tried_options_and_never_reverts(tmp_path):
    history = [
        {
            "trial": "0001",
            "proposed": {"runtime.cudagraphs": True},
            "decision": "accepted",
        },
        {
            "trial": "0002",
            "proposed": {"engine.precision": "fp16"},
            "decision": "rejected",
        },
    ]
    proposal = RuleBasedPlanner().propose(
        planner_context(config={"runtime.cudagraphs": True}, history=history),
        tmp_path,
    )
    # cudagraphs=False is the baseline's value and fp16 was tried already.
    assert proposal.changes == {"engine.optimization_level": 5}


def test_rule_based_planner_stops_when_every_option_was_tried(tmp_path):
    history = [
        {"proposed": {"runtime.cudagraphs": True}},
        {"proposed": {"engine.precision": "fp16"}},
        {"proposed": {"engine.optimization_level": 5}},
        {"proposed": {"workload.variant": "b"}},
        {"proposed": {"workload.variant": "c"}},
    ]
    proposal = RuleBasedPlanner().propose(planner_context(history=history), tmp_path)
    assert proposal.stop and not proposal.changes


def test_rule_based_planner_revisits_options_under_a_new_best_config(tmp_path):
    history = [
        {"trial": "0001", "parent": "0000", "proposed": {"runtime.cudagraphs": True}},
        {"trial": "0002", "parent": "0000", "proposed": {"engine.precision": "fp16"}},
        {
            "trial": "0003",
            "parent": "0002",
            "proposed": {"engine.optimization_level": 5},
        },
        {"trial": "0004", "parent": "0002", "proposed": {"workload.variant": "b"}},
        {"trial": "0005", "parent": "0002", "proposed": {"workload.variant": "c"}},
    ]
    context = planner_context(config={"engine.precision": "fp16"}, history=history)
    context["best"]["trial"] = "0002"
    proposal = RuleBasedPlanner().propose(context, tmp_path)
    # Every option was tried once; CUDA graphs only under the earlier best config.
    assert proposal.changes == {"runtime.cudagraphs": True}
    assert proposal.rationale.startswith("revisit under the new best config")
    context["history"].append(
        {"trial": "0006", "parent": "0002", "proposed": {"runtime.cudagraphs": True}}
    )
    assert RuleBasedPlanner().propose(context, tmp_path).stop


# A provider adapter that answers without an LLM: ``reply`` records what it was
# sent and proposes CUDA graphs, ``broken`` fails, ``stop`` stops and ``many``
# proposes more changes than allowed. The provider layer, shared with kernel
# generation, passes the planner context as ``evidence``.
ADAPTER = """
import json, os

def reply(*, model, prompt, workdir, timeout_s, schema, evidence):
    with open(os.path.join(workdir, "seen.json"), "w") as stream:
        json.dump({"model": model, "prompt": prompt, "schema": schema, "evidence": evidence}, stream)
    return {"action": "experiment", "changes": [{"knob": "runtime.cudagraphs", "value": True}],
            "rationale": "launch overhead dominates"}

def broken(**kwargs):
    raise RuntimeError("provider is down")

def stop(**kwargs):
    return {"action": "stop", "changes": [], "rationale": "nothing left"}

def many(**kwargs):
    return {"action": "experiment", "rationale": "everything at once", "changes": [
        {"knob": "runtime.cudagraphs", "value": True},
        {"knob": "engine.precision", "value": "fp16"},
        {"knob": "engine.optimization_level", "value": 5},
    ]}
"""


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    """Put the adapter module on PYTHONPATH; return its name."""
    (tmp_path / "planner_fixture.py").write_text(ADAPTER)
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join((str(tmp_path), os.environ.get("PYTHONPATH", "")))
    )
    return "planner_fixture"


def test_llm_planner_asks_the_provider_with_the_planner_context(adapter, tmp_path):
    planner = LLMPlanner(
        f"{adapter}:reply", "model-x", instructions="Prefer runtime changes."
    )
    request = planner_context(hints=[hint("launch_overhead")])
    proposal = planner.propose(request, tmp_path / "planner")
    assert proposal == Proposal(
        changes={"runtime.cudagraphs": True},
        rationale="launch overhead dominates",
        planner="llm",
    )
    seen = json.loads((tmp_path / "planner" / "seen.json").read_text())
    assert seen["model"] == "model-x"
    # The provider layer, shared with kernel generation, calls it "evidence".
    assert seen["evidence"] == request
    assert seen["schema"] == proposal_schema(2)
    assert "PLANNER CONTEXT:" in seen["prompt"]
    assert "Prefer runtime changes." in seen["prompt"]
    assert '"launch_overhead"' in seen["prompt"]


def test_llm_planner_can_stop(adapter, tmp_path):
    proposal = LLMPlanner(f"{adapter}:stop").propose(planner_context(), tmp_path)
    assert proposal.stop and proposal.rationale == "nothing left"


def test_llm_planner_falls_back_when_the_provider_fails(adapter, tmp_path):
    proposal = LLMPlanner(f"{adapter}:broken").propose(planner_context(), tmp_path)
    assert proposal.changes == {"runtime.cudagraphs": True}
    assert proposal.planner == "rule-based"
    assert "LLM planner unavailable" in proposal.note
    strict = LLMPlanner(f"{adapter}:broken", fallback=None)
    assert strict.to_json()["fallback"] is None
    with pytest.raises(_providers.ProviderError):
        strict.propose(planner_context(), tmp_path / "strict")


@pytest.mark.parametrize(
    "reply,match",
    [
        ({"action": "maybe", "changes": [], "rationale": ""}, "no valid action"),
        ({"action": "experiment", "changes": [], "rationale": ""}, "without changes"),
        ({"action": "experiment", "changes": ["x"], "rationale": ""}, "malformed"),
    ],
)
def test_malformed_replies_are_rejected(reply, match):
    with pytest.raises(ValueError, match=match):
        _proposal_from_reply(reply, "llm")


def test_proposals_are_checked_against_the_knobs():
    assert check_proposal(
        Proposal(changes={"engine.optimization_level": 5}), {}, KNOBS
    ) == {"engine.optimization_level": 5}
    with pytest.raises(ValueError, match="changes nothing"):
        check_proposal(Proposal(), {}, KNOBS)
    with pytest.raises(ValueError, match="Unknown knob"):
        check_proposal(Proposal(changes={"engine.fast": True}), {}, KNOBS)
    with pytest.raises(ValueError, match="does not allow"):
        check_proposal(Proposal(changes={"engine.optimization_level": 4}), {}, KNOBS)
    assert apply_changes({}, {"workload.variant": "a"}, KNOBS) == {}


def test_planners_round_trip_through_json():
    assert isinstance(planner_from_json(RuleBasedPlanner().to_json()), RuleBasedPlanner)
    llm = LLMPlanner("codex", "gpt", timeout_s=30, max_changes=3, instructions="hi")
    rebuilt = planner_from_json(json.loads(json.dumps(llm.to_json())))
    assert rebuilt.to_json() == llm.to_json()
    with pytest.raises(ValueError, match="Cannot rebuild"):
        planner_from_json({"kind": "custom"})
    with pytest.raises(ValueError, match="max_changes"):
        LLMPlanner(max_changes=0)


def test_llm_replies_may_carry_file_edits():
    schema = proposal_schema(2, ["kernels"])
    assert schema["properties"]["edits"]["items"]["properties"]["file"]["enum"] == [
        "kernels"
    ]
    assert "edits" not in proposal_schema(2)["properties"]
    proposal = _proposal_from_reply(
        {
            "action": "experiment",
            "changes": [],
            "edits": [{"file": "kernels", "search": "A", "replace": "B"}],
            "rationale": "fuse the norm",
        },
        "llm",
    )
    assert proposal.edits == {"kernels": [{"search": "A", "replace": "B"}]}
    assert not proposal.changes
    # Each entry is one edit; the edits of a file keep the reply's order.
    grouped = _proposal_from_reply(
        {
            "action": "experiment",
            "changes": [],
            "edits": [
                {"file": "kernels", "search": "A", "replace": "B"},
                {"file": "norms", "search": "N", "replace": "M"},
                {"file": "kernels", "search": "C", "replace": "D"},
            ],
            "rationale": "",
        },
        "llm",
    )
    assert grouped.edits == {
        "kernels": [{"search": "A", "replace": "B"}, {"search": "C", "replace": "D"}],
        "norms": [{"search": "N", "replace": "M"}],
    }
    with pytest.raises(ValueError, match="malformed edit"):
        _proposal_from_reply(
            {"action": "experiment", "changes": [], "edits": [{"file": 1}]}, "llm"
        )
    prompt = LLMPlanner().prompt(
        {
            **planner_context(),
            "editable_files": [{"name": "kernels"}],
            "guidance": "Go for fusion.",
            "noise": {"noise_threshold": 0.05},
        }
    )
    assert "SEARCH/REPLACE" in prompt and "Go for fusion." in prompt
    assert "by more than 5.0% beyond noise" in prompt


def test_the_reply_schema_requires_every_property():
    # Strict structured output (codex) needs every property listed as required.
    for editable_files in ((), ("speed",)):
        schema = proposal_schema(2, editable_files)
        assert set(schema["required"]) == set(schema["properties"])
    assert "edits" in proposal_schema(2, ("speed",))["required"]


def test_rule_based_planner_proposes_an_option_with_what_it_requires(tmp_path):
    knobs = knobs_by_name(
        [
            Knob("runtime.cudagraphs", (False, True), False, "graphs"),
            Knob("workload.layout", ("dense", "packed"), "dense", "layout"),
            Knob(
                "kernels.blocks",
                ("", "fused"),
                "",
                "fused kernels",
                ("roofline_gap",),
                {"runtime.cudagraphs": True, "workload.layout": "packed"},
            ),
        ]
    )
    context = {
        "goal": {"metric": "latency_ms", "min_improvement": 0.02, "target": None},
        "baseline": {"config": {}},
        "best": {
            "config": {"workload.layout": "packed"},
            "bottleneck_hints": [hint("roofline_gap")],
        },
        "history": [],
        "knobs": [knob.to_json() for knob in knobs.values()],
        "tried": [{}, {"workload.layout": "packed"}],
    }
    proposal = RuleBasedPlanner().propose(context, tmp_path)
    # The requirement the best config lacks comes along; the one it has does not.
    assert proposal.changes == {"kernels.blocks": "fused", "runtime.cudagraphs": True}
    assert "with runtime.cudagraphs=True" in proposal.rationale


def context_with_knobs(knobs, config=None, hints=(), history=(), tried=()):
    """A planner context for ``knobs``, with ``config`` as the best config."""
    config = config or {}
    return {
        "goal": {"metric": "latency_ms", "min_improvement": 0.02, "target": None},
        "baseline": {"config": {}},
        "best": {"config": config, "bottleneck_hints": list(hints), "trial": "0000"},
        "history": list(history),
        "knobs": [knob.to_json() for knob in knobs.values()],
        "tried": [{}, config, *tried],
    }


GRAPHS = Knob("runtime.cudagraphs", (False, True), False, "graphs")


def test_rule_based_planner_follows_requirements_transitively(tmp_path):
    knobs = knobs_by_name(
        [
            GRAPHS,
            Knob(
                "workload.layout",
                ("dense", "packed"),
                "dense",
                "",
                (),
                {"runtime.cudagraphs": True},
            ),
            Knob(
                "kernels.blocks",
                ("", "fused"),
                "",
                "",
                ("roofline_gap",),
                {"workload.layout": "packed"},
            ),
        ]
    )
    proposal = RuleBasedPlanner().propose(
        context_with_knobs(knobs, hints=[hint("roofline_gap")]), tmp_path
    )
    # The layout the blocks need needs CUDA graphs in turn.
    assert proposal.changes == {
        "kernels.blocks": "fused",
        "workload.layout": "packed",
        "runtime.cudagraphs": True,
    }
    assert proposal.rationale.endswith(
        "with workload.layout='packed', with runtime.cudagraphs=True"
    )


def test_rule_based_planner_skips_options_whose_requirements_cannot_hold(tmp_path):
    knobs = knobs_by_name(
        [
            GRAPHS,
            # Debug mode forbids CUDA graphs: it requires their default, off.
            Knob(
                "runtime.debug",
                (False, True),
                False,
                "",
                (),
                {"runtime.cudagraphs": False},
            ),
            Knob("engine.level", (3, 5), 3, ""),
            Knob(
                "kernels.blocks",
                ("", "fused"),
                "",
                "",
                ("roofline_gap",),
                {"runtime.cudagraphs": True},
            ),
        ]
    )
    config = {"runtime.debug": True}
    proposal = RuleBasedPlanner().propose(
        context_with_knobs(knobs, config, hints=[hint("roofline_gap")]), tmp_path
    )
    # Neither the fused blocks nor CUDA graphs can join this best config.
    assert proposal.changes == {"engine.level": 5}


def test_rule_based_planner_counts_only_the_chosen_changes_as_tried(tmp_path):
    knobs = knobs_by_name(
        [
            GRAPHS,
            Knob("engine.level", (3, 5), 3, ""),
            Knob(
                "kernels.blocks",
                ("", "fused"),
                "",
                "",
                (),
                {"runtime.cudagraphs": True},
            ),
        ]
    )
    rejected = {
        "trial": "0001",
        "parent": "0000",
        "changes": {"kernels.blocks": "fused", "runtime.cudagraphs": True},
        "proposed": {"kernels.blocks": "fused"},
        "decision": "rejected",
    }
    context = context_with_knobs(
        knobs,
        hints=[hint("launch_overhead")],
        history=[rejected],
        tried=[rejected["changes"]],
    )
    assert RuleBasedPlanner().propose(context, tmp_path).changes == {
        "runtime.cudagraphs": True
    }
    # An invalid proposal has no "proposed" changes: every change it asked
    # for counts as tried.
    context["history"].append(
        {
            "trial": "0002",
            "parent": "0000",
            "changes": {"runtime.cudagraphs": True},
            "decision": "invalid",
        }
    )
    assert RuleBasedPlanner().propose(context, tmp_path).changes == {"engine.level": 5}


def test_rule_based_planner_bundles_what_each_value_requires(tmp_path):
    knobs = knobs_by_name(
        [
            GRAPHS,
            Knob(
                "runtime.graph_pool",
                (False, True),
                False,
                "",
                (),
                {"runtime.cudagraphs": True},
            ),
            Knob(
                "kernels.mixed",
                ("none", "fused", "tiled"),
                "none",
                "",
                ("dominant_kernel",),
                {
                    "fused": {"runtime.cudagraphs": True},
                    "tiled": {"runtime.graph_pool": True},
                },
            ),
        ]
    )
    context = context_with_knobs(knobs, hints=[hint("dominant_kernel")])
    assert RuleBasedPlanner().propose(context, tmp_path).changes == {
        "kernels.mixed": "fused",
        "runtime.cudagraphs": True,
    }
    fused = {"kernels.mixed": "fused", "runtime.cudagraphs": True}
    context["history"].append(
        {"trial": "0001", "parent": "0000", "changes": fused, "proposed": fused}
    )
    context["tried"].append(fused)
    assert RuleBasedPlanner().propose(context, tmp_path).changes == {
        "kernels.mixed": "tiled",
        "runtime.graph_pool": True,
        "runtime.cudagraphs": True,
    }


def test_replies_with_more_changes_than_allowed_are_refused(adapter, tmp_path):
    reply = {
        "action": "experiment",
        "rationale": "",
        "changes": [{"knob": "a.b", "value": 1}, {"knob": "a.c", "value": 2}],
    }
    with pytest.raises(ValueError, match="2 knob changes; at most 1"):
        _proposal_from_reply(reply, "llm", 1)
    assert len(_proposal_from_reply(reply, "llm", 2).changes) == 2
    # The adapter ignores the schema's limit; the rule-based planner proposes instead.
    proposal = LLMPlanner(f"{adapter}:many", max_changes=2).propose(
        planner_context(), tmp_path
    )
    assert proposal.planner == "rule-based"
    assert "3 knob changes; at most 2 are allowed" in proposal.note


def test_the_prompt_carries_the_guidance_once():
    guidance = "Fuse the norms before anything else."
    prompt = LLMPlanner().prompt({**planner_context(), "guidance": guidance})
    assert prompt.count(guidance) == 1
    assert '"guidance"' not in prompt
    assert "Additional guidance from the user:" in prompt


def test_the_prompt_explains_requirements_per_value():
    prompt = LLMPlanner(max_changes=3).prompt(planner_context())
    assert 'or is a list of {"value", "requires"} entries' in prompt
    assert "they count toward the 3 changes" in prompt


def test_a_rate_limited_or_overloaded_api_falls_back_for_that_trial(
    tmp_path, monkeypatch
):
    def unavailable(**kwargs):
        raise _providers.ProviderUnavailableError(
            "The model API returned status 529", 529
        )

    monkeypatch.setattr(_providers, "propose", unavailable)
    custom = RuleBasedPlanner()
    proposal = LLMPlanner("claude", fallback=custom).propose(
        planner_context(), tmp_path
    )
    assert proposal.planner == custom.name
    assert "ProviderUnavailableError: The model API returned status 529" in (
        proposal.note
    )
    with pytest.raises(_providers.ProviderUnavailableError):
        LLMPlanner("claude", fallback=None).propose(
            planner_context(), tmp_path / "strict"
        )


def test_a_codex_turn_without_a_reply_fails_the_planner_and_a_rule_based_planner_stands_in(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(_providers.shutil, "which", lambda _: "/bin/codex")
    monkeypatch.setattr(_providers, "_CODEX_MCP_SERVERS", {"/bin/codex": ()})
    monkeypatch.setattr(
        _providers,
        "run_process",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )
    # Not a ProviderError, so the LLM planner's own fallback lets it through.
    planner = LLMPlanner("codex")
    with pytest.raises(RuntimeError, match="without writing") as raised:
        planner.propose(planner_context(), tmp_path / "planner")
    assert not isinstance(raised.value, _providers.ProviderError)
    # The session stands in a rule-based planner on any planner exception.
    session = TuningSession(
        WorkloadFactory(FAKE_FACTORY), directory=tmp_path / "s", planner=planner
    )
    proposal, failure = session._ask_planner(planner_context(), tmp_path / "session")
    assert failure == "RuntimeError: codex finished without writing a kernel proposal"
    assert proposal.planner == "rule-based"
    assert proposal.changes == {"runtime.cudagraphs": True}
    assert proposal.note == (
        f"the llm planner failed ({failure}); the rule-based planner proposed "
        "instead"
    )
    assert "without writing" in (tmp_path / "session/error.txt").read_text()


# ------------------------------------------------------- prepared kernels

KERNEL_KNOBS = knobs_by_name(FAKE_KNOBS + (KERNEL_KNOB,))


def with_prepared(status="ok", speedup=2.0, **kwargs):
    """A planner context with the prepared value kernels.fused=generated."""
    context = planner_context(knobs=KERNEL_KNOBS, **kwargs)
    context["kernels"] = [
        {
            "knob": "kernels.fused",
            "value": "generated",
            "status": status,
            "reason": None if status == "ok" else "build failed: RuntimeError: no",
            "prepared_config": {"kernels.fused": "generated"},
            "build_s": 1.0,
            "region_speedup": speedup if status == "ok" else None,
            "used_by_best": False,
            "regions": [],
        }
    ]
    return context


def test_rule_based_planner_tries_fast_prepared_kernels_before_engine_options(
    tmp_path,
):
    hints = [hint("precision", "engine.precision is fp32")]
    # Without prepared kernels, the knob the strongest hint names comes first.
    plain = RuleBasedPlanner().propose(
        planner_context(knobs=KERNEL_KNOBS, hints=hints), tmp_path
    )
    assert plain.changes == {"engine.precision": "fp16"}
    # A prepared kernel that beats its region's baseline comes before any
    # engine option, hint or not: an engine option can change its region's
    # generation task and leave it unusable.
    proposal = RuleBasedPlanner().propose(with_prepared(hints=hints), tmp_path)
    assert proposal.changes == {"kernels.fused": "generated"}
    assert proposal.rationale == (
        "its prepared kernels run their regions at least 2.00x faster than their "
        "baselines, and options that could change their regions' generation "
        "tasks come later -> try kernels.fused='generated'"
    )
    # One slower than its region's baseline is left out of the hints and
    # waits for the layer order, where kernels come last.
    dominant = [hint("dominant_kernel", "one kernel takes 80%")]
    slow = RuleBasedPlanner().propose(
        with_prepared(speedup=0.8, hints=dominant), tmp_path
    )
    assert slow.changes == {"engine.optimization_level": 5}
    tried = [
        {"engine.optimization_level": 5},
        {"runtime.cudagraphs": True},
        {"engine.precision": "fp16"},
        {"workload.variant": "b"},
        {"workload.variant": "c"},
    ]
    history = [
        {"trial": f"000{n}", "parent": "0000", "proposed": changes}
        for n, changes in enumerate(tried, start=1)
    ]
    last = RuleBasedPlanner().propose(
        with_prepared(speedup=0.8, hints=dominant, history=history), tmp_path
    )
    assert last.changes == {"kernels.fused": "generated"}


def test_rule_based_planner_never_proposes_a_value_that_could_not_be_prepared(
    tmp_path,
):
    hints = [hint("dominant_kernel")]
    history = [
        {"trial": f"000{n}", "parent": "0000", "proposed": changes}
        for n, changes in enumerate(
            [
                {"engine.optimization_level": 5},
                {"runtime.cudagraphs": True},
                {"engine.precision": "fp16"},
                {"workload.variant": "b"},
                {"workload.variant": "c"},
            ],
            start=1,
        )
    ]
    proposal = RuleBasedPlanner().propose(
        with_prepared(status="failed", hints=hints, history=history), tmp_path
    )
    assert proposal.stop


def test_the_prompt_explains_prepared_kernels_only_when_there_are_some():
    prompt = " ".join(LLMPlanner().prompt(with_prepared()).split())
    assert "Prepared kernels are already measured per region" in prompt
    assert "must still win the whole-model head-to-head" in prompt
    assert (
        'such an experiment is refused as an invalid proposal (stage "kernels"), '
        "which costs no trial"
    ) in prompt
    assert '"region_speedup": 2.0' in prompt
    assert "Prepared kernels" not in LLMPlanner().prompt(planner_context())
