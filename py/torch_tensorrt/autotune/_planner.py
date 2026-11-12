# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Planners: they read the planner context and propose the next experiment.

Before each trial the session gives the planner a JSON planner context: the
goal, the noise, the limits and the budget, the baseline and the best config
so far with its profile and bottleneck hints, every earlier trial, the knobs,
the editable files and the prepared knob values with their region kernels.
The planner answers with a :class:`Proposal`: changes to the best config, or
stop.

A planner only proposes. It cannot build, measure or accept a candidate: the
session checks every proposal against the workload's knobs, and a separate
worker process measures what the candidate achieves.

Two planners are built in. :class:`RuleBasedPlanner` tries one option at a
time, starting with what the bottleneck hints point at. :class:`LLMPlanner`
asks an LLM coding agent, and falls back to the rule-based planner when the
agent fails.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from torch_tensorrt.autotune._knobs import (
    Knob,
    apply_changes,
    is_member,
    knobs_by_name,
    with_requirements,
)
from torch_tensorrt.autotune._settings import digest

# The order in which the rule-based planner goes through the layers' knobs,
# after the knobs the bottleneck hints point at.
_LAYER_ORDER = ("runtime", "engine", "graph", "workload", "kernels")
# LLMPlanner's default fallback: a new RuleBasedPlanner (None means no fallback).
_RULE_BASED_FALLBACK: Any = object()


@dataclass(frozen=True)
class Proposal:
    """The next experiment: knob changes and file edits to the best config, or stop.

    ``changes`` maps knob names to their new values. ``edits`` maps an
    editable file's name to SEARCH/REPLACE edits (``{"search": ...,
    "replace": ...}``) of that file's version in the best config; see
    :class:`~torch_tensorrt.autotune.EditableFile`. ``rationale`` says why.
    ``stop=True`` ends the search instead. ``planner`` names the planner
    that proposed it, and ``note`` records anything unusual, such as a
    fallback planner standing in.
    """

    changes: Mapping[str, Any] = field(default_factory=dict)
    rationale: str = ""
    stop: bool = False
    planner: str = ""
    note: str = ""
    edits: Mapping[str, Sequence[Mapping[str, str]]] = field(default_factory=dict)


class Planner(ABC):
    """Base class of planners: map the planner context to a :class:`Proposal`.

    ``workdir`` (``trials/NNNN/planner``) is where the planner may keep its
    files, such as an LLM provider's request and reply. If ``propose``
    raises, the session records the error and asks a
    :class:`RuleBasedPlanner` instead.
    """

    name: str = "planner"

    @abstractmethod
    def propose(self, context: Mapping[str, Any], workdir: Path) -> Proposal: ...

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.name}


def knobs_from_context(context: Mapping[str, Any]) -> dict[str, Knob]:
    """The workload's knobs, rebuilt from the planner context's ``knobs`` entries."""
    return knobs_by_name([Knob.from_json(knob) for knob in context["knobs"]])


def _tried_options(
    context: Mapping[str, Any],
    knobs: Mapping[str, Knob],
    parent: str | None = None,
) -> set[tuple[str, str]]:
    """Options already tried: the baseline's values and every value a trial chose.

    A measured trial counts its ``proposed`` changes, not the requirements
    that came with them. An invalid proposal was never measured and has no
    ``proposed`` changes; every change it asked for counts as tried. With
    ``parent``, only trials proposed from that best config count, so an
    option judged under an earlier best config can be revisited.
    """
    baseline = context["baseline"]["config"]
    tried = {
        (name, json.dumps(baseline.get(name, knob.default)))
        for name, knob in knobs.items()
    }
    for trial in context.get("history", ()):
        if parent is not None and trial.get("parent") != parent:
            continue
        if "proposed" in trial:
            chosen = trial["proposed"]
        elif trial.get("decision") == "invalid":
            chosen = trial.get("changes", {})
        else:
            chosen = {}
        for name, value in chosen.items():
            tried.add((name, json.dumps(value)))
    return tried


def _prepared_kernels(
    context: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    """The planner context's prepared knob values, by knob and JSON value."""
    return {
        (entry["knob"], json.dumps(entry["value"])): entry
        for entry in context.get("kernels", ())
    }


def _region_speedup(entry: Mapping[str, Any] | None) -> float | None:
    """A prepared value's smallest measured region speedup, if it was prepared."""
    if entry is None or entry.get("status") != "ok":
        return None
    speedup = entry.get("region_speedup")
    if isinstance(speedup, bool) or not isinstance(speedup, (int, float)):
        return None
    return float(speedup)


def _options_in_order(
    best: Mapping[str, Any],
    knobs: Mapping[str, Knob],
    prepared: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[tuple[Knob, Any, str]]:
    """The options in the order the rule-based planner tries them, each with its reason.

    First the prepared values whose region kernels all run faster than their
    baselines, fastest first: they were prepared in the baseline config, and
    an accepted engine option can change their regions' generation tasks and
    leave them unusable. Then, for each bottleneck hint of the best config
    (strongest first), the values of the knobs that declare they relieve it,
    without the prepared values whose kernels are slower than their
    baselines. Then every knob's values, layer by layer (see
    ``_LAYER_ORDER``). An option may appear more than once.
    """
    fast = []
    for knob in knobs.values():
        for value in knob.values:
            speedup = _region_speedup(prepared.get((knob.name, json.dumps(value))))
            if speedup is not None and speedup > 1:
                fast.append((speedup, knob, value))
    fast.sort(key=lambda item: -item[0])
    ordered: list[tuple[Knob, Any, str]] = [
        (
            knob,
            value,
            f"its prepared kernels run their regions at least {speedup:.2f}x faster "
            "than their baselines, and options that could change their regions' "
            "generation tasks come later",
        )
        for speedup, knob, value in fast
    ]
    for hint in best.get("bottleneck_hints", ()):
        reason = f"{hint['name']}: {hint['detail']}"
        for knob in knobs.values():
            if hint["name"] not in knob.bottleneck_hints:
                continue
            for value in knob.values:
                speedup = _region_speedup(prepared.get((knob.name, json.dumps(value))))
                if speedup is None or speedup > 1:
                    ordered.append((knob, value, reason))
    for layer in _LAYER_ORDER:
        for knob in knobs.values():
            if knob.layer == layer:
                ordered += [
                    (knob, value, f"untried {layer}-layer option")
                    for value in knob.values
                ]
    return ordered


class RuleBasedPlanner(Planner):
    """Try one option at a time, starting with what the profile points at.

    An option is one value of one knob. The planner goes through the knobs
    in this order: for each bottleneck hint of the best config, strongest
    first, the knobs that declare they relieve it; then every knob, runtime
    first, then engine, graph, workload and kernels. Each knob's values are
    taken in the order the knob lists them. Prepared values (the planner
    context's ``kernels``) whose region kernels all run faster than their
    baselines come before everything else, fastest first, since an accepted
    engine option can change their regions' generation tasks and leave them
    unusable; a prepared value whose kernels are slower waits for the layer
    order.

    It proposes the first option not tried yet, together with the settings
    the option requires (and what those require in turn). It skips the best
    config's current values, the baseline's values (so an accepted change is
    never undone), an option whose requirements conflict with each other or
    with the best config, an option that needs a value whose preparation
    failed, and a config that was already measured. A setting that only came
    along as another option's requirement does not count as tried. When
    every option was tried once, it revisits an option rejected under an
    earlier best config, since options interact: what did not help then may
    help now. It stops when every option was tried under the current best
    config.
    """

    name = "rule-based"

    def propose(self, context: Mapping[str, Any], workdir: Path) -> Proposal:
        knobs = knobs_from_context(context)
        best = context["best"]
        config = best["config"]
        tried_configs = {digest(dict(c)) for c in context.get("tried", ())}
        tried_anywhere = _tried_options(context, knobs)
        tried_here = _tried_options(context, knobs, parent=best.get("trial"))
        current = {name: config.get(name, knob.default) for name, knob in knobs.items()}
        prepared = _prepared_kernels(context)
        unusable = {
            option for option, entry in prepared.items() if entry.get("status") != "ok"
        }
        ordered = _options_in_order(best, knobs, prepared)

        # First an option no trial chose; then one that no trial chose under
        # the current best config.
        for tried, prefix in (
            (tried_anywhere, ""),
            (tried_here, "revisit under the new best config: "),
        ):
            for knob, value, reason in ordered:
                if is_member(value, (current[knob.name],)):
                    continue
                if (knob.name, json.dumps(value)) in tried:
                    continue
                # An option comes with the settings it requires; one whose
                # requirements cannot hold in this config is skipped, and so
                # is one that needs a value that could not be prepared.
                try:
                    changes = with_requirements(config, {knob.name: value}, knobs)
                    candidate = apply_changes(config, changes, knobs)
                except ValueError:
                    continue
                if any(
                    (name, json.dumps(setting)) in unusable
                    for name, setting in changes.items()
                ):
                    continue
                if digest(candidate) in tried_configs:
                    continue
                needs = "".join(
                    f", with {name}={changes[name]!r}"
                    for name in changes
                    if name != knob.name
                )
                return Proposal(
                    changes=changes,
                    rationale=f"{prefix}{reason} -> try {knob.name}={value!r}"
                    f"{needs}",
                    planner=self.name,
                )
        return Proposal(
            stop=True,
            rationale="Every exposed option has been tried",
            planner=self.name,
        )


_PROMPT = """\
You are the planner of a Torch-TensorRT Auto-Tuner session. The user's goal,
limits and budget, the validated baseline, the best config so far ("best"),
its profile and bottleneck hints, the history of every trial and the knobs you
may turn are in the PLANNER CONTEXT below.

Choose the next experiment: at most {max_changes} knob changes applied to the
best config, each a knob name and one of that knob's allowed values. Or stop,
when no remaining experiment is likely to help. A separate worker process
builds and measures your experiment on the same workload; the session keeps it
only if it improves {metric} by more than {min_improvement:.1%} beyond noise and
passes the accuracy and memory limits. You cannot measure or accept results
yourself.

Prefer the change that addresses the strongest measured bottleneck. Learn from
the history: do not repeat a config that was already tried, and do not retry an
option that failed for a reason the new experiment does not address. A knob's
"requires" maps other knobs to the values every non-default value of it needs,
or is a list of {{"value", "requires"}} entries, each naming what that one value
needs. The best config must already have them or the experiment must set them
too, and they count toward the {max_changes} changes. Explain in "rationale"
which part of the planner context motivates the experiment and what you expect.
{kernels}{edits}{instructions}
PLANNER CONTEXT:
{context}
"""

_KERNELS_PROMPT = """
Some knob values were prepared before the search ("kernels" in the PLANNER
CONTEXT): a worker built the baseline config with the value ("prepared_config")
and generated the region kernels it needs. For each prepared value you see
whether its preparation succeeded, and for each of its regions the kernel's
digest, how it was found ("new", "cached" or "shared"), how long generation
took, and the kernel's median time against the region's eager PyTorch or
TensorRT baseline ("speedup" above 1 is faster; "region_speedup" is the
smallest of them). Use them to choose between kernel sources. Prepared kernels
are already measured per region, but a region speedup is not a whole-model
gain: an experiment that uses one must still win the whole-model head-to-head
like any other. A run that uses a prepared value only loads kernels from the
cache, and for every prepared generation task it must run that task's prepared
kernel. A setting that changes what a region's kernel is generated for (for
example an engine build option that kernel generation records, or a precision
that changes the region's data types) needs a kernel that was never prepared:
such an experiment is refused as an invalid proposal (stage "kernels"), which
costs no trial but counts toward the invalid proposals that stop the search.
So try a prepared value before such settings, and combine it with settings
like those of its prepared_config. A value whose preparation failed cannot be
used.
"""

_EDIT_PROMPT = """
The workload also exposes source files you may edit ("editable_files" in the
PLANNER CONTEXT: each file's description and its EDITABLE regions as they are
in the best config). An experiment may change knobs, edit files, or both.
Each entry of "edits" is one SEARCH/REPLACE edit of one file: "file" names the
file, "search" must match its current text exactly once inside its EDITABLE
regions, and "replace" is the new text. Edits of one file apply in order, each
to the text the previous edit left. The session refuses edits outside those
regions, edits that change a fence line or empty a region, and files that do
not compile; the worker checks accuracy against the reference and only measured
gains count. Send an empty "edits" list when you change only knobs. Keep edits
small, focused and correct; equal speed with simpler code is better.
"""


def proposal_schema(
    max_changes: int, editable_files: Sequence[str] = ()
) -> dict[str, Any]:
    """The JSON schema of the LLM's reply.

    The reply has an ``action`` (``experiment`` or ``stop``), at most
    ``max_changes`` knob changes (``{"knob", "value"}``) and a
    ``rationale``. When the workload has editable files, it also has a list
    of ``edits``, each one SEARCH/REPLACE edit of one of those files.
    """
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["experiment", "stop"]},
            "changes": {
                "type": "array",
                "maxItems": max_changes,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "knob": {"type": "string"},
                        "value": {"type": ["string", "number", "boolean", "null"]},
                    },
                    "required": ["knob", "value"],
                },
            },
            "rationale": {"type": "string"},
        },
        "required": ["action", "changes", "rationale"],
    }
    if editable_files:
        schema["properties"]["edits"] = {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "file": {"type": "string", "enum": list(editable_files)},
                    "search": {"type": "string"},
                    "replace": {"type": "string"},
                },
                "required": ["file", "search", "replace"],
            },
        }
        # Strict structured output needs every property required; an empty
        # list means no edit.
        schema["required"].append("edits")
    return schema


class LLMPlanner(Planner):
    """Ask an LLM coding agent to choose the next experiment from the planner context.

    ``provider`` is ``"claude"``, ``"codex"`` (logged-in CLIs) or a trusted
    ``"module:attribute"`` adapter, as for
    :class:`~torch_tensorrt.region.KernelGenerator`. The agent receives the
    rules, the user's guidance (the session's ``guidance.md`` and
    ``instructions``) and the planner context as JSON, and answers with at
    most ``max_changes`` knob changes (and edits, if the workload has
    editable files), which the session then checks.

    If the provider reports an error (a ``ProviderError``, or a
    ``ProviderUnavailableError`` for a rate-limited or overloaded model
    API), times out or sends a malformed reply (one with more than
    ``max_changes`` changes, say), ``fallback`` (by default a
    :class:`RuleBasedPlanner`) proposes instead, and the proposal's note
    says why. Any other exception, and any of these with ``fallback=None``,
    propagates; the session then records the failure and asks a
    :class:`RuleBasedPlanner` for that trial.
    """

    name = "llm"

    def __init__(
        self,
        provider: str = "claude",
        model: str | None = None,
        *,
        timeout_s: float = 600.0,
        max_changes: int = 2,
        instructions: str | None = None,
        fallback: Planner | None = _RULE_BASED_FALLBACK,
    ) -> None:
        if not isinstance(max_changes, int) or not 1 <= max_changes <= 5:
            raise ValueError("LLMPlanner max_changes must be between 1 and 5")
        self.provider = provider
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_changes = max_changes
        self.instructions = instructions or ""
        self.fallback = (
            RuleBasedPlanner() if fallback is _RULE_BASED_FALLBACK else fallback
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.name,
            "provider": self.provider,
            "model": self.model,
            "timeout_s": self.timeout_s,
            "max_changes": self.max_changes,
            "instructions": self.instructions,
            "fallback": self.fallback.to_json() if self.fallback else None,
        }

    def prompt(self, context: Mapping[str, Any]) -> str:
        """The prompt: the rules, the user's guidance and ``context`` as JSON."""
        goal = context["goal"]
        noise_threshold = (context.get("noise") or {}).get("noise_threshold")
        guidance = "\n\n".join(
            part for part in (context.get("guidance"), self.instructions) if part
        )
        # The guidance has its own section; the PLANNER CONTEXT does not repeat it.
        shown = {key: value for key, value in context.items() if key != "guidance"}
        return _PROMPT.format(
            max_changes=self.max_changes,
            metric=goal["metric"],
            min_improvement=(
                noise_threshold
                if noise_threshold is not None
                else goal["min_improvement"]
            ),
            kernels=_KERNELS_PROMPT if context.get("kernels") else "",
            edits=_EDIT_PROMPT if context.get("editable_files") else "",
            instructions=(
                f"\nAdditional guidance from the user:\n{guidance}\n"
                if guidance
                else ""
            ),
            context=json.dumps(shown, indent=1, sort_keys=True, default=str),
        )

    def propose(self, context: Mapping[str, Any], workdir: Path) -> Proposal:
        from torch_tensorrt.dynamo.region_compilation._generation import _providers

        try:
            reply = _providers.propose(
                provider=self.provider,
                model=self.model,
                prompt=self.prompt(context),
                workdir=workdir,
                timeout_s=self.timeout_s,
                schema=proposal_schema(
                    self.max_changes,
                    [
                        editable["name"]
                        for editable in context.get("editable_files", ())
                    ],
                ),
                # The provider layer (shared with kernel generation) calls the
                # planner context "evidence".
                evidence=dict(context),
            )
            return _proposal_from_reply(reply, self.name, self.max_changes)
        except (
            _providers.ProviderError,
            # A rate-limited or overloaded model API (429, 5xx): the next trial
            # asks the LLM again.
            _providers.ProviderUnavailableError,
            TimeoutError,
            ValueError,
            TypeError,
        ) as error:
            if self.fallback is None:
                raise
            proposal = self.fallback.propose(context, workdir)
            return replace(
                proposal,
                note=f"LLM planner unavailable ({type(error).__name__}: {error}); "
                f"{self.fallback.name} planner proposed instead",
            )


def _proposal_from_reply(
    reply: Mapping[str, Any], planner: str, max_changes: int | None = None
) -> Proposal:
    """Turn the LLM's reply into a :class:`Proposal`; ValueError if it is malformed.

    With ``max_changes``, a reply listing more knob changes is refused even
    if its provider did not enforce the schema's limit. The reply's edits,
    one entry per SEARCH/REPLACE edit, are grouped by file in reply order.
    """
    action = reply.get("action")
    if action not in ("experiment", "stop"):
        raise ValueError(f"Planner reply has no valid action: {action!r}")
    rationale = str(reply.get("rationale", ""))
    if action == "stop":
        return Proposal(stop=True, rationale=rationale, planner=planner)
    listed = reply.get("changes") or ()
    if not isinstance(listed, (list, tuple)):
        raise ValueError(f"Planner reply has malformed changes: {listed!r}")
    if max_changes is not None and len(listed) > max_changes:
        raise ValueError(
            f"Planner reply proposes {len(listed)} knob changes; at most "
            f"{max_changes} are allowed"
        )
    changes: dict[str, Any] = {}
    for change in listed:
        if not isinstance(change, Mapping) or "knob" not in change:
            raise ValueError(f"Planner reply has a malformed change: {change!r}")
        changes[str(change["knob"])] = change.get("value")
    edits: dict[str, list[dict[str, str]]] = {}
    for edit in reply.get("edits") or ():
        if not (
            isinstance(edit, Mapping)
            and isinstance(edit.get("file"), str)
            and isinstance(edit.get("search"), str)
            and isinstance(edit.get("replace"), str)
        ):
            raise ValueError(f"Planner reply has a malformed edit: {edit!r}")
        edits.setdefault(edit["file"], []).append(
            {"search": edit["search"], "replace": edit["replace"]}
        )
    if not changes and not edits:
        raise ValueError("Planner reply proposes an experiment without changes")
    return Proposal(changes=changes, rationale=rationale, planner=planner, edits=edits)


def planner_from_json(data: Mapping[str, Any]) -> Planner:
    """Rebuild a built-in planner from :meth:`Planner.to_json` (for resuming)."""
    kind = data.get("kind")
    if kind == RuleBasedPlanner.name:
        return RuleBasedPlanner()
    if kind == LLMPlanner.name:
        fallback = data.get("fallback")
        return LLMPlanner(
            data["provider"],
            data.get("model"),
            timeout_s=data["timeout_s"],
            max_changes=data["max_changes"],
            instructions=data.get("instructions") or None,
            fallback=planner_from_json(fallback) if fallback else None,
        )
    raise ValueError(f"Cannot rebuild planner {kind!r}; pass the planner explicitly")


def check_proposal(
    proposal: Proposal, config: Mapping[str, Any], knobs: Mapping[str, Knob]
) -> dict[str, Any]:
    """Check a proposal before anything is built; return the config it leads to.

    ``config`` is the best config and ``proposal.changes`` already includes
    the ``edit.<name>`` digests of its edited files. Raises ValueError,
    saying why, for a proposal that changes nothing, names an unknown knob
    or a value the knob does not allow, or breaks a requirement.
    """
    if not proposal.changes:
        raise ValueError("The proposal changes nothing")
    return apply_changes(config, proposal.changes, knobs)
