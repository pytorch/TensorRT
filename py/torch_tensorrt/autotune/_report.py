# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""What a session writes for people to read.

* ``report.md``: the decision, a before/after comparison of the baseline and
  the winner (the config the session ends with), the reproducible config,
  the prepared kernels, every trial and the profiles; ``report.json`` holds
  the same as data.
* ``trials.tsv``: one line per decided trial, rewritten as the search goes.

It also words the reason of a failed result (:func:`failure_reason`), the
same way for decisions, reports and the planner.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from torch_tensorrt.autotune._knobs import Knob, difference, is_member

if TYPE_CHECKING:
    from torch_tensorrt.autotune._settings import Goal


# A failure reason keeps this much of the first error line.
_REASON_CHARS = 500


def first_error_line(error: Any) -> str:
    """The line of an error message that says what failed.

    That is the first line that starts in the first column: an exception's
    ``Type: message`` even after the source excerpt a ``SyntaxError`` begins
    with, or the first line of a message the session wrote.
    """
    lines = [line for line in str(error or "").splitlines() if line.strip()]
    if not lines:
        return "?"
    line = next((line for line in lines if not line[0].isspace()), lines[0]).strip()
    if len(line) > _REASON_CHARS:
        line = line[: _REASON_CHARS - 3] + "..."
    return line


def failure_detail(result: Mapping[str, Any]) -> str:
    """The first error line of a failed result, with the signal that ended its worker.

    A worker that left no result also gets the first error line of its log.
    """
    detail = first_error_line(result.get("error"))
    signal = result.get("signal")
    if signal and signal not in detail:
        detail += f" (signal {signal})"
    log_error = result.get("log_error")
    if log_error and log_error not in detail:
        detail += f"; log: {first_error_line(log_error)}"
    return detail


def failure_reason(result: Mapping[str, Any]) -> str:
    """``"<stage> failed: <first error line>"``, naming the signal that ended the worker."""
    return f"{result.get('stage')} failed: {failure_detail(result)}"


# Formatting helpers: each returns the text of one table cell.


def _status_text(result: Mapping[str, Any]) -> str:
    """``ok``, or ``failed (<stage>, <signal>)``."""
    status = str(result.get("status", "-"))
    if status != "failed":
        return status
    cause = ", ".join(
        str(part) for part in (result.get("stage"), result.get("signal")) if part
    )
    return f"failed ({cause})"


def _number_text(value: Any, digits: int = 4) -> str:
    """A number to ``digits`` significant digits; ``-`` for None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-" if value is None else str(value)
    return f"{value:.{digits}g}"


def _latency_text(result: Mapping[str, Any]) -> str:
    """``median [p25, p75]`` of a result's latency in ms per unit."""
    latency = result.get("latency_ms")
    if not latency:
        return "-"
    return (
        f"{_number_text(latency['median'])} [{_number_text(latency['p25'])}, "
        f"{_number_text(latency['p75'])}]"
    )


def _change_text(before: Any, after: Any, lower_is_better: bool) -> str:
    """The relative change from ``before`` to ``after``, marked when it is better."""
    if not all(isinstance(v, (int, float)) for v in (before, after)) or not before:
        return ""
    relative = (after - before) / before
    better = relative < 0 if lower_is_better else relative > 0
    return f"{relative:+.1%}" + (" (better)" if better and abs(relative) > 0 else "")


def _profile_lines(title: str, result: Mapping[str, Any]) -> list[str]:
    """One result's profile: overheads, bottleneck hints and top kernels."""
    profile = result.get("profile")
    lines = [f"### {title}", ""]
    if not profile:
        return lines + ["No profile was recorded.", ""]
    lines += [
        f"- {_number_text(profile['wall_ms'])} ms per unit wall (untraced), "
        f"{_number_text(profile['device_ms'])} ms GPU work: the GPU waits for the host "
        f"{profile['host_overhead_fraction']:.0%} of the time",
        f"- {profile['kernels']:.0f} kernel launches and "
        f"{profile['host_syncs']:.0f} device-to-host copies per unit",
    ]
    for hint in result.get("bottleneck_hints", ()):
        lines.append(f"- bottleneck hint `{hint['name']}`: {hint['detail']}")
    lines += [
        "",
        "| GPU time (ms/unit) | Share | Calls/unit | Kernel |",
        "|---|---|---|---|",
    ]
    for kernel in profile.get("top_kernels", ()):
        lines.append(
            f"| {_number_text(kernel['ms'])} | {kernel['share']:.0%} | "
            f"{_number_text(kernel['calls'], 3)} | `{kernel['name']}` |"
        )
    return lines + [""]


def _before_after_lines(
    baseline: Mapping[str, Any],
    winner: Mapping[str, Any],
    bounds: Mapping[str, tuple[str, float]],
) -> list[str]:
    """The before/after table: latency, throughput, accuracy, memory and structure."""
    before, after = baseline["result"], winner["result"]
    latency_change = _change_text(
        before["latency_ms"]["median"], after["latency_ms"]["median"], True
    )
    lines = [
        "## Before / after",
        "",
        f"| | Baseline ({baseline['id']}) | Winner ({winner['id']}) | Change |",
        "|---|---|---|---|",
        f"| latency ms/unit, median [p25, p75] | {_latency_text(before)} | "
        f"{_latency_text(after)} | {latency_change} |",
        f"| throughput items/s | {_number_text(before.get('throughput'))} | "
        f"{_number_text(after.get('throughput'))} | "
        f"{_change_text(before.get('throughput'), after.get('throughput'), False)} |",
    ]
    for name in bounds:
        lines.append(
            f"| accuracy `{name}` | {_number_text(before['accuracy'].get(name), 6)} | "
            f"{_number_text(after['accuracy'].get(name), 6)} | |"
        )
    for key, label in (
        ("process_gpu_mb", "GPU memory MB (process)"),
        ("torch_peak_mb", "PyTorch peak MB"),
    ):
        b = (before.get("memory") or {}).get(key)
        a = (after.get("memory") or {}).get(key)
        lines.append(
            f"| {label} | {_number_text(b)} | {_number_text(a)} | "
            f"{_change_text(b, a, True)} |"
        )
    structure_b, structure_a = before.get("structure", {}), after.get("structure", {})
    lines.append(
        "| TRT engines / PyTorch partitions / plugins | "
        + " / ".join(
            str(structure_b.get(k, "-"))
            for k in ("trt_engines", "torch_partitions", "plugin_regions")
        )
        + " | "
        + " / ".join(
            str(structure_a.get(k, "-"))
            for k in ("trt_engines", "torch_partitions", "plugin_regions")
        )
        + " | |"
    )
    lines.append(
        f"| build s | {_number_text(before.get('build_s'))} | "
        f"{_number_text(after.get('build_s'))} | |"
    )
    line_b, line_a = before.get("roofline") or {}, after.get("roofline") or {}
    if line_b or line_a:
        floor = line_a.get("floor_ms") or line_b.get("floor_ms")
        lines.append(
            f"| speed vs. the memory-bandwidth floor ({_number_text(floor)} ms/unit) | "
            + " | ".join(
                f"{line['fraction']:.0%}" if line.get("fraction") else "-"
                for line in (line_b, line_a)
            )
            + " | |"
        )
    return lines + [""]


def _final_check_lines(final_check: Mapping[str, Any]) -> list[str]:
    """The final check's verdict, its gain in each pair and any problems."""
    gains = ", ".join(f"{gain:+.1%}" for gain in final_check["gains"])
    median = final_check["median_gain"]
    lines = [
        "Final check in fresh baseline/best worker pairs: median gain "
        + (f"{median:+.1%}" if median is not None else "unavailable")
        + f" (per pair: {gains or 'none'}); "
        + ("passed." if final_check["passed"] else "did NOT pass.")
    ]
    for problem in final_check["problems"]:
        lines.append(f"- {problem}")
    return lines + [""]


def _uses(config: Mapping[str, Any], preparation: Mapping[str, Any]) -> bool:
    """Whether ``config`` sets the prepared knob value of ``preparation``."""
    knob = preparation["knob"]
    return knob in config and is_member(config[knob], (preparation["value"],))


def _kernel_lines(
    kernels: Sequence[Mapping[str, Any]], winner: Mapping[str, Any]
) -> list[str]:
    """The prepared knob values: their status, the region kernels each adds or
    changes, and the warnings about kernels that changed while they were
    prepared."""
    lines = [
        "## Kernels",
        "",
        "Knob values prepared before the search (`kernels/`), with the region "
        "kernels each adds or changes. A run that uses a prepared value must run, "
        "for every prepared generation task, that task's kernel; the time and "
        "speedup are each kernel's own, on its region, against the region's eager "
        "PyTorch or TensorRT baseline.",
        "",
        "| Value | Status | Winner uses | Region | Kernel | Origin | Kernel ms | "
        "Baseline ms | Region speedup | Generation s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for preparation in kernels:
        value = preparation["value"]
        text = value if isinstance(value, str) else json.dumps(value)
        status = str(preparation["status"])
        if preparation.get("reason"):
            status += f": {preparation['reason']}"
        first = [
            f"`{preparation['knob']}={text}`",
            status.replace("|", "\\|").replace("\n", " "),
            "yes" if _uses(winner["config"], preparation) else "no",
        ]
        for kernel in preparation.get("regions") or [None]:
            cells = ["-"] * 7 if kernel is None else _kernel_cells(kernel)
            lines.append("| " + " | ".join(first + cells) + " |")
    warnings = dict.fromkeys(
        warning
        for preparation in kernels
        for warning in preparation.get("warnings") or ()
    )
    if warnings:
        lines.append("")
        lines += [f"- warning: {warning}" for warning in warnings]
    return lines + [""]


def _kernel_cells(kernel: Mapping[str, Any]) -> list[str]:
    """A region kernel's cells: region, digest, origin, times, speedup, generation."""
    speedup = kernel.get("speedup")
    baseline = kernel.get("baseline_latency_ms")
    return [
        f"`{kernel.get('region_id')}`",
        f"`{str(kernel.get('kernel_sha256'))[:12]}`",
        str(kernel.get("origin", "-")),
        _number_text(kernel.get("latency_ms")),
        (
            "-"
            if baseline is None
            else f"{_number_text(baseline)} ({kernel.get('baseline')})"
        ),
        f"{speedup:.2f}x" if isinstance(speedup, (int, float)) else "-",
        _number_text(kernel.get("generation_s")),
    ]


def _trial_rows(trials: list[dict[str, Any]]) -> list[str]:
    """One row per trial: its changes, result, decision and the planner's rationale."""
    rows = []
    for trial in trials:
        result = trial.get("result") or {}
        rationale = (trial.get("proposal") or {}).get("rationale", "")
        note = (trial.get("proposal") or {}).get("note")
        if note:
            rationale = f"{rationale} [{note}]"
        rows.append(
            "| {id} | {parent} | `{changes}` | {status} | {latency} | {decision} | {reason} | {rationale} |".format(
                id=trial["id"],
                parent=trial.get("parent") or "-",
                changes=json.dumps(trial.get("changes") or {}, sort_keys=True),
                status=_status_text(result),
                latency=_latency_text(result),
                decision=trial["decision"]["decision"],
                reason=str(trial["decision"]["reason"]).replace("|", "\\|"),
                rationale=rationale.replace("|", "\\|").replace("\n", " "),
            )
        )
    return rows


def write_report(
    directory: Path,
    *,
    goal: Goal,
    bounds: Mapping[str, tuple[str, float]],
    knobs: Mapping[str, Knob],
    trials: list[dict[str, Any]],
    baseline: dict[str, Any],
    winner: dict[str, Any],
    decision: str,
    stop_reason: str,
    final_check: dict[str, Any] | None,
    artifact: Path | None,
    environment: Mapping[str, Any] | None,
    planner: Mapping[str, Any],
    elapsed_s: float,
    noise: Mapping[str, Any] | None = None,
    export: Mapping[str, Any] | None = None,
    workload: Mapping[str, Any] | None = None,
    kernels: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Write ``report.md`` and ``report.json`` in ``directory``; return the Markdown path.

    ``winner`` is the trial the session ends with: the best config if it
    passed the final check, otherwise the baseline. ``export`` is the
    export's summary (its ``status`` and ``reason``), None when the session
    does not export; ``workload`` the pinned workload description, whose
    source files the report lists. ``kernels`` are the prepared knob values
    (the entries of ``kernels/summary.json``); the report lists each one's
    status and region kernels, and whether the winner uses it.
    """
    before, after = baseline["result"], winner["result"]
    lower = goal.lower_is_better
    measured = [t for t in trials[1:] if t["decision"]["decision"] != "invalid"]
    invalid = len(trials) - 1 - len(measured)
    headline = (
        f"accepted trial {winner['id']}"
        if decision == "accepted"
        else (
            "baseline kept (no candidate beat it in the final check)"
            if final_check
            else "baseline kept (no candidate was accepted)"
        )
    )
    noise_threshold = (noise or {}).get("noise_threshold", goal.min_improvement)
    spread = (noise or {}).get("spread")
    noise_note = (
        f" (goal {goal.min_improvement:.1%}; the baseline moved {spread:.2%} across "
        f"{len((noise or {}).get('metrics', []))} fresh-process runs)"
        if spread is not None
        else (
            " (no noise run qualified, so the noise is unmeasured)"
            if (noise or {}).get("requested")
            else ""
        )
    )
    # The overview: decision, goal, bounds, planner, trial counts, environment.
    lines = [
        f"# Auto-Tuner report: {directory.name}",
        "",
        f"**Decision: {headline}.** Search stopped: {stop_reason.rstrip('.')}.",
        "",
        f"- Goal: {'minimize' if lower else 'maximize'} `{goal.metric}`"
        + (f", target {goal.target:g}" if goal.target else "")
        + f"; a candidate must improve it by more than the noise threshold, "
        f"{noise_threshold:.1%}{noise_note}",
        "- Accuracy bounds: "
        + ", ".join(f"`{name} {op} {value:g}`" for name, (op, value) in bounds.items()),
        f"- Planner: `{json.dumps(dict(planner), sort_keys=True)}`",
        f"- Trials: {len(measured)} measured, {invalid} invalid proposals; "
        f"elapsed {elapsed_s:.0f} s",
    ]
    if environment:
        lines.append(
            "- Environment: "
            + ", ".join(
                f"{key} {_environment_value(value)}"
                for key, value in environment.items()
            )
        )
    files = (workload or {}).get("workload_files")
    if files:
        lines.append(
            f"- Workload source: {len(files)} files pinned by SHA-256 "
            f"(beside `{Path(sorted(files)[0]).parent}`)"
        )
    lines.append("")
    lines += _before_after_lines(baseline, winner, bounds)
    if final_check:
        lines += _final_check_lines(final_check)
    # The winner's config, where it is saved, and every trial.
    changes = difference(baseline["config"], winner["config"], knobs)
    config_file = (
        artifact / "config.json"
        if artifact
        else directory / "export" / "config.json" if export else None
    )
    lines += [
        "## Config",
        "",
        "Changes from the baseline: "
        + (f"`{json.dumps(changes, sort_keys=True)}`" if changes else "none"),
        "",
        f"Full config: `{json.dumps(winner['config'], sort_keys=True)}`"
        + (f" (also in `{config_file}`)" if config_file else ""),
        "",
        "Artifact: " + _artifact_line(artifact, export),
        "",
    ]
    if kernels:
        lines += _kernel_lines(kernels, winner)
    lines += [
        "## Trials",
        "",
        "| Trial | Parent | Changes | Status | latency ms/unit | Decision | Reason | Planner rationale |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += _trial_rows(trials)
    lines += ["", "## Profiles", ""]
    lines += _profile_lines(f"Baseline ({baseline['id']})", before)
    if winner is not baseline:
        lines += _profile_lines(f"Winner ({winner['id']})", after)
    report_path = directory / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    summary = {
        "decision": decision,
        "stop_reason": stop_reason,
        "goal": asdict(goal),
        "accuracy_bounds": {name: list(bound) for name, bound in bounds.items()},
        "baseline": {
            "trial": baseline["id"],
            "config": baseline["config"],
            "result": before,
        },
        "winner": {
            "trial": winner["id"],
            "config": winner["config"],
            "result": after,
        },
        "changes": changes,
        "final_check": final_check,
        "noise": dict(noise) if noise else None,
        "artifact": str(artifact) if artifact else None,
        "export": dict(export) if export else None,
        "environment": dict(environment or {}),
        "workload_files": dict((workload or {}).get("workload_files") or {}),
        "planner": dict(planner),
        "elapsed_s": elapsed_s,
        "kernels": [
            {**preparation, "winner_uses": _uses(winner["config"], preparation)}
            for preparation in kernels or ()
        ],
        "trials": [_trial_summary(trial) for trial in trials],
    }
    (directory / "report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    return report_path


def _trial_summary(trial: Mapping[str, Any]) -> dict[str, Any]:
    """A trial as ``report.json`` lists it."""
    result = trial.get("result") or {}
    summary = {
        "id": trial["id"],
        "parent": trial.get("parent"),
        "changes": trial.get("changes"),
        "config": trial.get("config"),
        "status": result.get("status"),
        "decision": trial["decision"],
        "proposal": trial.get("proposal"),
        "latency_ms": result.get("latency_ms"),
    }
    if "proposed" in trial:
        summary["proposed"] = trial["proposed"]
    if result.get("status") == "failed":
        summary.update(
            stage=result.get("stage"),
            error=failure_detail(result),
            exit_status=result.get("exit_status"),
            signal=result.get("signal"),
        )
    return summary


def _environment_value(value: Any) -> str:
    """An environment entry as text; a nested mapping as ``(key value, ...)``."""
    if isinstance(value, Mapping):
        return "(" + ", ".join(f"{k} {v}" for k, v in value.items()) + ")"
    return str(value)


def _artifact_line(artifact: Path | None, export: Mapping[str, Any] | None) -> str:
    """Where the artifact is, or why there is none (and what an earlier export left)."""
    if export is None:
        return f"`{artifact}`" if artifact else "not exported"
    status, reason = export.get("status"), export.get("reason")
    if artifact:
        return f"`{artifact}` ({status}: {reason})"
    line = f"not exported ({status}: {reason})"
    kept = export.get("kept")
    if kept:
        line += f"; `{kept.get('artifact')}` keeps an earlier export"
        if kept.get("config") == export.get("config"):
            line += " of this config"
        elif kept.get("config") is not None:
            line += " of another config (see its config.json)"
    return line


# The columns of trials.tsv.
_TRIALS_TABLE_COLUMNS = (
    "trial",
    "parent",
    "decision",
    "metric",
    "paired_gain",
    "roofline",
    "accuracy",
    "build_s",
    "changes",
    "reason",
    "stage",
    "signal",
)


def write_trials_table(
    directory: Path, trials: list[dict[str, Any]], goal: Goal
) -> Path:
    """Write ``trials.tsv``: one line per decided trial, for people to read.

    ``metric`` is the trial's first measurement and ``paired_gain`` the
    median gain of its head-to-head runs. ``reason`` is the decision's
    reason (cut to 200 characters; report.json keeps it whole). ``stage``
    and ``signal`` say where a failed trial failed and which signal, if any,
    ended its worker.
    """
    rows = ["\t".join(_TRIALS_TABLE_COLUMNS)]
    for trial in trials:
        decision = trial.get("decision")
        if not decision:
            continue
        result = trial.get("result") or {}
        ok = result.get("status") == "ok"
        head_to_head = decision.get("head_to_head") or {}
        roofline = result.get("roofline") or {}
        metric = (
            result["throughput"]
            if ok and goal.metric == "throughput"
            else (result["latency_ms"]["median"] if ok else None)
        )
        fields = {
            "trial": trial["id"],
            "parent": trial.get("parent") or "",
            "decision": decision["decision"],
            "metric": _number_text(metric) if metric is not None else "",
            "paired_gain": (
                f"{head_to_head['median_gain']:+.2%}"
                if head_to_head.get("median_gain") is not None
                else ""
            ),
            "roofline": (
                f"{roofline['fraction']:.0%}" if roofline.get("fraction") else ""
            ),
            "accuracy": ";".join(
                f"{name}={_number_text(value, 4)}"
                for name, value in sorted((result.get("accuracy") or {}).items())
            ),
            "build_s": _number_text(result.get("build_s"), 3) if ok else "",
            "changes": json.dumps(trial.get("changes") or {}, sort_keys=True),
            "reason": " ".join(str(decision.get("reason", "")).split())[:200],
            "stage": str(result.get("stage") or "") if not ok else "",
            "signal": str(result.get("signal") or ""),
        }
        rows.append(
            "\t".join(
                fields[column].replace("\t", " ") for column in _TRIALS_TABLE_COLUMNS
            )
        )
    path = directory / "trials.tsv"
    path.write_text("\n".join(rows) + "\n")
    return path
