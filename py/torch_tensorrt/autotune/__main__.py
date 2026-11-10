# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""The command line: run, inspect, report on and resume Auto-Tuner sessions.

    python -m torch_tensorrt.autotune run --workload FILE.py:FACTORY --session DIR [options]
    python -m torch_tensorrt.autotune status SESSION_DIR
    python -m torch_tensorrt.autotune report SESSION_DIR
    python -m torch_tensorrt.autotune resume SESSION_DIR [--max-trials N] ...

``run`` tunes the :class:`~torch_tensorrt.autotune.Workload` that a workload
factory (``path/to/file.py:function`` or ``package.module:function``) returns
when called with the ``--args`` JSON keyword arguments, for example::

    python -m torch_tensorrt.autotune run --session sessions/my-model \\
        --workload my_model.py:create_workload --args '{"batch": 8}' \\
        --accuracy max_abs_error '<=' 0.01 --max-trials 10

Running the same command again resumes the session. Accuracy limits not given
with ``--accuracy`` use the workload's defaults. ``--planner llm`` asks the
logged-in ``claude`` CLI (or the ``--provider``) to choose the experiments,
with ``--guidance FILE`` as the user's guidance.

``status`` and ``report`` only read the session's records. ``resume``
continues the recorded session, optionally with a larger budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, NoReturn

from torch_tensorrt.autotune import (
    Budget,
    Goal,
    Limits,
    LLMPlanner,
    NoiseControl,
    Planner,
    RuleBasedPlanner,
    TuningError,
    TuningSession,
    WorkloadFactory,
)
from torch_tensorrt.autotune._session import read_session_json


def _print_status(directory: Path, session_json: dict[str, Any]) -> int:
    """Print the session's settings, its prepared knob values and one line per
    trial (two with a reason)."""
    factory = session_json["workload_factory"]
    print(f"Session {directory}")
    print(f"  workload: {factory['factory']} {json.dumps(factory['kwargs'])}")
    print(f"  goal: {json.dumps(session_json['goal'])}")
    print(
        f"  planner: {json.dumps(session_json['planner'])}; "
        f"elapsed {session_json['elapsed_s']:.0f} s"
    )
    kernels = _read_json(directory / "kernels" / "summary.json") or {}
    for preparation in kernels.get("preparations", ()):
        speedup = preparation.get("region_speedup")
        detail = preparation.get("reason") or (
            f"{len(preparation.get('regions') or ())} region kernels"
            + (f", region speedup {speedup:.2f}x or more" if speedup else "")
        )
        print(f"  prepared {preparation['name']}: {preparation['status']}; {detail}")
    for warning in kernels.get("warnings", ()):
        print(f"  warning: {warning.get('message')}")
    rows = []
    for trial in sorted((directory / "trials").glob("[0-9][0-9][0-9][0-9]")):
        record = _read_json(trial / "trial.json") or {}
        result = _read_json(trial / "result.json") or {}
        decision = _read_json(trial / "decision.json") or record.get("decision") or {}
        latency = (result.get("latency_ms") or {}).get("median")
        if result:
            status = result.get("status", "-")
        elif decision:
            # An invalid proposal is never measured. A failed baseline that a
            # resume measures again keeps its old decision, but not its
            # result, until the new measurement is decided.
            status = "invalid" if decision.get("decision") == "invalid" else "failed"
        else:
            status = "running" if record else "-"
        label = decision.get("decision", "pending")
        reason = decision.get("reason", "")
        proposal = _read_json(trial / "proposal.json") or {}
        if not record and proposal.get("stop"):
            # The planner chose to stop here: nothing was built or measured.
            label, reason = (
                "stopped",
                f"the planner stopped: {proposal.get('rationale', '')}",
            )
        rows.append(
            (
                trial.name,
                json.dumps(record.get("changes") or {}, sort_keys=True),
                status,
                "-" if latency is None else f"{latency:.4g}",
                label,
                reason,
            )
        )
    print(
        f"  {'trial':5}  {'status':8}  {'ms/unit':>8}  {'decision':9}  changes / reason"
    )
    for name, changes, status, latency, decision, reason in rows:
        print(f"  {name:5}  {status:8}  {latency:>8}  {decision:9}  {changes}")
        if reason:
            print(f"  {'':35}{reason}")
    report = directory / "report.md"
    if report.is_file():
        print(f"  finished; report: {report}")
    return 0


def _read_json(path: Path) -> Any:
    """The JSON in ``path``, or None if there is no such file."""
    return json.loads(path.read_text()) if path.is_file() else None


def _json_object(text: str) -> dict[str, Any]:
    """Parse an option's JSON object, for argparse."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON ({error}): {text!r}") from None
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(f"expected a JSON object; got {text!r}")
    return value


def _add_run_arguments(run: argparse.ArgumentParser) -> None:
    """The options of ``run``; each maps to one setting of the Python API."""
    run.add_argument(
        "--workload",
        required=True,
        metavar="FACTORY",
        help="workload factory: path/to/file.py:function or package.module:function",
    )
    run.add_argument(
        "--args",
        type=_json_object,
        default="{}",
        metavar="JSON",
        help="JSON keyword arguments of the factory",
    )
    run.add_argument(
        "--session",
        required=True,
        metavar="DIR",
        help="session directory; running again in it resumes the session",
    )
    run.add_argument(
        "--planner",
        choices=("rule-based", "llm"),
        default="rule-based",
        help="who proposes the experiments",
    )
    run.add_argument(
        "--provider", default="claude", help="LLM agent CLI or module:attribute"
    )
    run.add_argument("--llm-model", help="model name for the LLM agent CLI")
    run.add_argument(
        "--metric",
        choices=("latency_ms", "throughput"),
        default="latency_ms",
        help="what to optimize",
    )
    run.add_argument(
        "--max-trials", type=int, default=8, help="most candidate trials to run"
    )
    run.add_argument("--max-time-s", type=float, help="wall-clock limit of the search")
    run.add_argument(
        "--patience",
        type=int,
        help="stop after this many trials in a row without an accepted candidate",
    )
    run.add_argument(
        "--trial-timeout-s",
        type=float,
        default=1800.0,
        help="kill a worker that builds or measures for longer",
    )
    run.add_argument(
        "--prepare-timeout-s",
        type=float,
        default=7200.0,
        help="kill a worker that prepares a knob value (generates its kernels) "
        "for longer",
    )
    run.add_argument(
        "--min-improvement",
        type=float,
        default=0.02,
        help="smallest relative gain that gets a candidate accepted",
    )
    run.add_argument(
        "--target", type=float, help="stop once the metric reaches this value"
    )
    run.add_argument(
        "--accuracy",
        nargs=3,
        action="append",
        default=[],
        metavar=("METRIC", "OP", "VALUE"),
        help="accuracy limit, e.g. --accuracy max_abs_error '<=' 0.01 (repeatable)",
    )
    run.add_argument("--max-memory-mb", type=float)
    run.add_argument(
        "--guidance",
        metavar="FILE",
        help="Markdown guidance for the LLM planner (copied to guidance.md)",
    )
    run.add_argument(
        "--baseline",
        type=_json_object,
        metavar="JSON",
        help="JSON knob changes to the workload's baseline config",
    )
    run.add_argument(
        "--noise-runs",
        type=int,
        default=2,
        help="extra baseline runs that measure the run-to-run noise",
    )
    run.add_argument(
        "--head-to-head-pairs",
        type=int,
        default=2,
        help="best/candidate run pairs a candidate must win to be accepted",
    )
    run.add_argument(
        "--final-check-pairs",
        type=int,
        default=2,
        help="baseline/best run pairs of the final check",
    )
    run.add_argument(
        "--quiet-gpu-s",
        type=float,
        default=600.0,
        help="seconds each worker may wait for an idle GPU; 0 turns the wait off",
    )
    run.add_argument(
        "--max-spread",
        type=float,
        default=0.03,
        help="largest interquartile range, relative to the median, of a stable run",
    )
    run.add_argument(
        "--retries",
        type=int,
        default=2,
        help="times a worker times an unstable measurement again",
    )


def _parser() -> argparse.ArgumentParser:
    """The parser of the four commands: run, status, report and resume."""
    parser = argparse.ArgumentParser(
        prog="python -m torch_tensorrt.autotune",
        description="Run, inspect, report on and resume Auto-Tuner sessions.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="tune a workload; running again resumes the session",
        description=(
            "Tune the workload that a workload factory returns, recording "
            "the session in --session. Running the same command again "
            "resumes the session."
        ),
    )
    _add_run_arguments(run)
    # Option values that only the session's own checks reject are reported
    # like those argparse rejects: as usage errors of `run`.
    run.set_defaults(usage_error=run.error)
    commands.add_parser(
        "status", help="list a session's trials and decisions"
    ).add_argument("directory", type=Path)
    commands.add_parser(
        "report", help="print a finished session's report"
    ).add_argument("directory", type=Path)
    resume = commands.add_parser(
        "resume", help="continue a recorded session, optionally with a new budget"
    )
    resume.add_argument("directory", type=Path)
    resume.add_argument("--max-trials", type=int)
    resume.add_argument("--max-time-s", type=float)
    resume.add_argument("--patience", type=int)
    return parser


def _accuracy_limits(limits: list[list[str]]) -> dict[str, tuple[str, float]]:
    """The ``--accuracy METRIC OP VALUE`` options as bounds; a later one wins."""
    bounds = {}
    for name, comparator, value in limits:
        try:
            bounds[name] = (comparator, float(value))
        except ValueError:
            raise ValueError(
                f"--accuracy {name} {comparator} needs a number; got {value!r}"
            ) from None
    return bounds


def _session_from_args(args: argparse.Namespace) -> TuningSession:
    """The session that ``run`` tunes; it resumes if ``--session`` already holds it."""
    planner: Planner = (
        LLMPlanner(args.provider, args.llm_model)
        if args.planner == "llm"
        else RuleBasedPlanner()
    )
    return TuningSession(
        # WorkloadFactory makes a path/to/file.py factory absolute, since
        # workers and later resumes may start in another directory.
        WorkloadFactory(args.workload, args.args),
        directory=args.session,
        goal=Goal(
            metric=args.metric,
            target=args.target,
            min_improvement=args.min_improvement,
        ),
        limits=Limits(
            accuracy=_accuracy_limits(args.accuracy), max_memory_mb=args.max_memory_mb
        ),
        budget=Budget(
            max_trials=args.max_trials,
            max_time_s=args.max_time_s,
            patience=args.patience,
            trial_timeout_s=args.trial_timeout_s,
            prepare_timeout_s=args.prepare_timeout_s,
        ),
        planner=planner,
        baseline=args.baseline,
        final_check_pairs=args.final_check_pairs,
        noise_control=NoiseControl(
            noise_runs=args.noise_runs,
            head_to_head_pairs=args.head_to_head_pairs,
            quiet_gpu_s=args.quiet_gpu_s,
            retries=args.retries,
            max_spread=args.max_spread,
        ),
        guidance=Path(args.guidance).read_text() if args.guidance else None,
    )


def main(argv: list[str] | None = None) -> int:
    """Run one command; return the exit status (usage errors exit with 2)."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        try:
            session = _session_from_args(args)
        except (OSError, ValueError) as error:
            usage_error: Callable[[str], NoReturn] = args.usage_error
            usage_error(str(error))
        return _run_and_report(session)
    directory = args.directory.expanduser().resolve()
    try:
        session_json = read_session_json(directory)
    except TuningError as error:
        print(error, file=sys.stderr)
        return 1
    if args.command == "status":
        return _print_status(directory, session_json)
    if args.command == "report":
        report = directory / "report.md"
        if not report.is_file():
            print(
                "The session has not finished; run `status` to see its progress",
                file=sys.stderr,
            )
            return 1
        print(report.read_text())
        return 0

    try:
        session = TuningSession.resume(directory)
    except TuningError as error:
        print(error, file=sys.stderr)
        return 1
    changes = {
        key: value
        for key, value in (
            ("max_trials", args.max_trials),
            ("max_time_s", args.max_time_s),
            ("patience", args.patience),
        )
        if value is not None
    }
    if changes:
        session.budget = replace(session.budget, **changes)
    return _run_and_report(session)


def _run_and_report(session: TuningSession) -> int:
    """Run the session and print its report; a TuningError is printed, exit 1.

    A TuningError is a refusal the user can act on (a directory of another
    session or version, changed code, a baseline that fails), so it is shown
    as a message, not a traceback.
    """
    try:
        result = session.run()
    except TuningError as error:
        print(error, file=sys.stderr)
        return 1
    print(result.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
