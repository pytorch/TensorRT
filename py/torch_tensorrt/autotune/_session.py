# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""The tuning session: the steps of a run, and what each step records.

:meth:`TuningSession.run` takes these steps. Each writes its records under
the session directory, so a stopped session resumes where it stopped:

1. **Start or resume** (``session.json``): record the session's settings or,
   on a resume, check that the directory holds this same session.
2. **Record the reference** (``reference/``, ``reference.pt``): a worker
   describes the workload and, on the first run, saves the reference
   outputs that every candidate is checked against. A resumed session
   refuses to go on if the workload or the environment changed.
3. **Measure the baseline** (``trials/0000``), then **measure the noise**
   (``noise/``): more fresh runs of the baseline set the noise threshold.
4. **Prepare kernels** (``kernels/``): build each knob value that declares
   ``prepare`` once, such as a value that generates region kernels, while
   the search would still run a trial, and record the kernels it generated;
   later runs that use a prepared value are pinned to the kernel each
   prepared generation task got.
5. **Search**, one trial (``trials/NNNN``) at a time: ask the planner
   (``context.json``, ``proposal.json``), check the proposal, take the
   candidate's first measurement (``trial.json``, ``result.json``) and, if
   it beats the best config, run it head-to-head against the best config
   (``head_to_head/``); then accept or reject it (``decision.json``).
6. **Final check** (``final_check/``): the best config against the baseline.
7. **Export** the winner (``export/``, ``artifact/``) and **report**
   (``report.md``, ``report.json``, ``trials.tsv``).

``events.jsonl`` logs the steps as they happen. Every build and measurement
runs in a fresh worker process (see ``_worker``), so a crash, a hang or
leaked GPU memory cannot affect the session or the other candidates.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from torch_tensorrt.autotune._code_edits import (
    EDIT_PREFIX,
    EditableFile,
    EditedFileStore,
    apply_edits,
    check_compiles,
    editable_blocks,
    is_edit_key,
)
from torch_tensorrt.autotune._knobs import (
    Knob,
    apply_changes,
    chosen_changes,
    difference,
    is_member,
    knobs_by_name,
    with_requirements,
)
from torch_tensorrt.autotune._planner import (
    Planner,
    Proposal,
    RuleBasedPlanner,
    check_proposal,
    planner_from_json,
)
from torch_tensorrt.autotune._report import (
    failure_detail,
    failure_reason,
    write_report,
    write_trials_table,
)
from torch_tensorrt.autotune._settings import (
    Budget,
    Config,
    Goal,
    Limits,
    NoiseControl,
    Timing,
    WorkloadFactory,
    digest,
    violations,
)
from torch_tensorrt.autotune._worker import write_json

logger = logging.getLogger(__name__)

# session.json names its format and version, so a directory that another
# program or another version of the Auto-Tuner wrote is refused, not misread.
_FORMAT = "torch_tensorrt.autotune session"
_VERSION = 1
# The search stops after this many invalid proposals in a row (unknown knobs,
# disallowed values, configs already measured).
_MAX_INVALID_PROPOSALS = 3
# A worker past trial_timeout_s first gets SIGUSR1, on which it writes its
# Python stacks to its log, and this long to do so before it is killed.
_DUMP_GRACE_S = 1.0
# For a worker that left no result: how much of the end of its log and of
# its stack dump the record keeps, and how much of the log is read.
_LOG_TAIL_CHARS = 1500
_LOG_STACK_CHARS = 2000
_LOG_SCAN_BYTES = 1 << 20
# A log line that reports an error, unless it is a warning (_WARNING_LINE)
# that only mentions one.
_ERROR_LINE = re.compile(
    r"error|exception|fatal|segmentation fault|terminate called|abort|killed|"
    r"out of memory|^E\d{4} |\[E\]",
    re.IGNORECASE,
)
_WARNING_LINE = re.compile(
    r"^\s*(?:W\d{4}\s|WARNING\b|\[W\b)|\[TRT\] \[W\]|Warning\b|warnings\.warn"
)
# The characters a prepared value's directory name (kernels/<knob>=<value>)
# keeps; others become "_".
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._=+-]")
# Digest characters shown in messages about pinned kernels.
_SHORT_DIGEST = 12


class TuningError(RuntimeError):
    """The session cannot continue, for example because the baseline failed."""


@dataclass(frozen=True)
class TuningResult:
    """The outcome of a session.

    * ``decision`` is ``"accepted"`` when a candidate beat the baseline in
      the final check, otherwise ``"baseline kept"``; ``stop_reason`` says
      why the search stopped.
    * The winner is the config the session ends with: the best config, if
      it passed the final check, otherwise the baseline. ``config`` is the
      winner's config; ``baseline`` and ``winner`` summarize the baseline
      and the winner.
    * ``final_check`` summarizes the final check (``None`` when no
      candidate was accepted) and ``noise`` the noise runs.
    * ``artifact`` is the directory the winner was exported to (``None``
      unless the export was made and passed its accuracy check).
    * ``export_summary`` says how the export went (``None`` when the
      session does not export). Its ``status`` is ``"exported"``,
      ``"export_rejected"`` (the rebuilt winner missed the accuracy bounds),
      ``"export_unsupported"`` (the workload cannot save it) or
      ``"export_failed"``, with the ``reason``. An export that did not pass
      leaves ``directory/artifact`` as the last export that passed left it,
      and ``export_summary["kept"]`` then names it and the config it holds.
    * ``report_path`` is ``report.md``; :meth:`report` returns its text.

    Memory is the process's GPU memory from NVML where available, otherwise
    PyTorch's peak allocation, which does not include TensorRT's own
    allocations.
    """

    directory: Path
    decision: str
    stop_reason: str
    config: Config
    baseline: dict[str, Any]
    winner: dict[str, Any]
    final_check: dict[str, Any] | None
    artifact: Path | None
    report_path: Path
    noise: dict[str, Any] | None = None
    export_summary: dict[str, Any] | None = None

    def report(self) -> str:
        return self.report_path.read_text()


def metric_value(result: Mapping[str, Any], goal: Goal) -> float:
    """A measured result's goal metric: median latency (ms per unit) or throughput."""
    if goal.metric == "throughput":
        return float(result["throughput"])
    return float(result["latency_ms"]["median"])


def _quartiles_separated(candidate: Mapping[str, Any], best: Mapping[str, Any]) -> bool:
    """Whether the candidate's latency quartiles lie entirely below the best config's.

    With equal items per unit this also decides throughput; otherwise compare
    the throughput implied by the quartiles.
    """
    c, b = candidate["latency_ms"], best["latency_ms"]
    c_items, b_items = candidate.get("items", 1), best.get("items", 1)
    if c_items == b_items:
        return bool(c["p75"] < b["p25"])
    return bool(c_items / c["p75"] > b_items / b["p25"])


def memory_mb(result: Mapping[str, Any]) -> float | None:
    """GPU memory in MB: the process's (from NVML) if known, else PyTorch's peak."""
    memory = result.get("memory") or {}
    value = memory.get("process_gpu_mb")
    return value if value is not None else memory.get("torch_peak_mb")


def _accuracy_misses(
    result: Mapping[str, Any], bounds: Mapping[str, tuple[str, float]]
) -> list[str]:
    """Every accuracy bound a measured result misses, before or after timing."""
    return violations(result["accuracy"], bounds) + [
        f"after timing: {miss}"
        for miss in violations(result["accuracy_after_timing"], bounds)
    ]


def read_session_json(directory: str | os.PathLike[str]) -> dict[str, Any]:
    """The ``session.json`` of an Auto-Tuner session directory of this version.

    Raises :class:`TuningError` when ``directory`` has no ``session.json``,
    or one that another program or another version of the Auto-Tuner wrote.
    """
    path = Path(directory) / "session.json"
    if not path.is_file():
        raise TuningError(
            f"{directory} is not an Auto-Tuner session directory: it has no "
            "session.json"
        )
    try:
        recorded = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise TuningError(f"Cannot read {path}: {error}") from None
    if (
        not isinstance(recorded, dict)
        or recorded.get("format") != _FORMAT
        or recorded.get("version") != _VERSION
    ):
        raise TuningError(
            f"{directory} is not an Auto-Tuner session of this version: its "
            f"session.json does not record format {_FORMAT!r}, version "
            f"{_VERSION}; start a new session directory"
        )
    return recorded


class _Progress:
    """``session.json`` during a run: its contents, and the time spent so far.

    The time budget spans resumes, so the time spent is this run's time plus
    what earlier runs recorded. :meth:`checkpoint` saves it.
    """

    def __init__(self, path: Path, session_json: dict[str, Any]) -> None:
        self.path = path
        self.session_json = session_json
        self._started = time.monotonic()
        self._elapsed_before = float(session_json["elapsed_s"])

    def elapsed(self) -> float:
        """Seconds the session has run: this run and the earlier ones."""
        return self._elapsed_before + time.monotonic() - self._started

    def checkpoint(self) -> None:
        """Save ``session.json`` with the time spent so far."""
        self.session_json["elapsed_s"] = self.elapsed()
        write_json(self.path, self.session_json)


@dataclass
class _SearchState:
    """Where the search stands: the trials so far, the best config, and the counts.

    The counts decide when the search stops (see
    :meth:`TuningSession._stop_reason`).
    """

    # Every trial so far, the baseline (trial 0000) first.
    trials: list[dict[str, Any]]
    # The record of the best config so far: the baseline or the trial
    # accepted last.
    best: dict[str, Any]
    # Trials that were measured (not invalid); Budget.max_trials counts these.
    measured: int = 0
    # Invalid proposals in a row; the search stops at _MAX_INVALID_PROPOSALS.
    invalid_in_a_row: int = 0
    # Measured trials since the last accepted one; Budget.patience counts these.
    since_accepted: int = 0

    def count(self, record: dict[str, Any]) -> None:
        """Count a decided trial; an accepted one becomes the best config."""
        decision = record["decision"]["decision"]
        if decision == "invalid":
            self.invalid_in_a_row += 1
            return
        self.invalid_in_a_row = 0
        self.measured += 1
        if decision == "accepted":
            self.best, self.since_accepted = record, 0
        else:
            self.since_accepted += 1


class TuningSession:
    """Tune one workload toward a goal, within limits and a budget.

    The session first measures the baseline: the workload's baseline config,
    changed by ``baseline``. It prepares each knob value that declares
    ``prepare`` (see :class:`~torch_tensorrt.autotune.Knob`): one worker
    builds the baseline config with that value, within
    ``budget.prepare_timeout_s``, and records the region kernels the build
    generated; every later run that uses a prepared value must run each
    prepared generation task's kernel, and a value whose preparation failed
    cannot be proposed. Then it searches, one trial at a time:

    1. the planner reads the planner context and proposes changes to the
       best config so far;
    2. the session checks the proposal against the workload's knobs;
    3. a fresh worker process builds the candidate, checks its outputs
       against the reference outputs, times it and profiles it (the first
       measurement);
    4. the candidate is accepted, and becomes the best config, only if it
       meets the accuracy and memory limits, was measured stably, and
       improves the goal metric by more than the noise threshold: first in
       its first measurement, then head-to-head against the best config, in
       interleaved pairs of fresh runs (see
       :class:`~torch_tensorrt.autotune.NoiseControl`).

    The noise threshold is ``goal.min_improvement`` or, if larger, a
    multiple of how much the baseline's metric varies between fresh runs
    (the noise runs, measured when the session starts). The search stops
    when the best config reaches the goal's target, the budget is used up,
    ``budget.patience`` trials in a row were not accepted, the planner made
    three invalid proposals in a row, or the planner stops. If the planner
    raises, the error is recorded and a :class:`RuleBasedPlanner` proposes
    that trial instead; the search stops only if it has nothing left either.

    Finally the best config faces the baseline in the final check:
    ``final_check_pairs`` pairs of fresh baseline/best runs. It wins only if
    every run qualifies, it gains in every pair and the median gain exceeds
    the noise threshold; otherwise the baseline is kept. A worker then
    rebuilds the winner and exports it; the export replaces
    ``directory/artifact``, with the winner's ``config.json``, only if the
    rebuilt candidate still meets the accuracy bounds. ``trials.tsv`` lists
    every trial as it is decided.

    Every step is recorded under ``directory``. Running a session again in
    the same directory resumes it: decided trials are reused, an interrupted
    trial is measured again, a baseline that failed is measured again, and
    the final check always measures anew. A session refuses to resume once
    its workload, environment or code changed; see :meth:`run`.
    """

    def __init__(
        self,
        workload_factory: WorkloadFactory,
        *,
        directory: str | os.PathLike[str],
        goal: Goal | None = None,
        limits: Limits | None = None,
        budget: Budget | None = None,
        planner: Planner | None = None,
        baseline: Config | None = None,
        timing: Timing | None = None,
        final_check_pairs: int = 2,
        export: bool = True,
        python: str | None = None,
        env: Mapping[str, str] | None = None,
        noise_control: NoiseControl | None = None,
        guidance: str | None = None,
    ) -> None:
        if not isinstance(workload_factory, WorkloadFactory):
            raise TypeError(
                "TuningSession needs a torch_tensorrt.autotune.WorkloadFactory"
            )
        if not isinstance(final_check_pairs, int) or final_check_pairs < 1:
            raise ValueError("final_check_pairs must be a positive integer")
        self.workload_factory = workload_factory
        self.directory = Path(directory).expanduser().resolve()
        self.goal = goal or Goal()
        self.limits = limits or Limits()
        self.budget = budget or Budget()
        self.planner = planner or RuleBasedPlanner()
        # Knob changes to the workload's own baseline config.
        self.baseline_changes = dict(baseline or {})
        # None times every candidate with the workload's own Timing.
        self.timing = timing
        self.final_check_pairs = final_check_pairs
        self.export = export
        # The Python that runs the workers, with ``env`` added to this
        # process's environment.
        self.python = python or sys.executable
        self.env = dict(env or {})
        self.cwd = os.getcwd()
        self.noise_control = noise_control or NoiseControl()
        # Markdown guidance for an LLM planner, kept in guidance.md.
        self.guidance = guidance
        # The gain a candidate must exceed: goal.min_improvement until the
        # noise runs measure the spread between fresh runs (_measure_noise).
        self.noise_threshold = self.goal.min_improvement
        self.noise_spread: float | None = None
        # The prepared knob values, and the kernel that each generation task
        # they prepared must run (by task digest), from _prepare_kernels
        # (kernels/summary.json).
        self._preparations: list[dict[str, Any]] = []
        self._pins: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------- creating and resuming

    @classmethod
    def resume(
        cls,
        directory: str | os.PathLike[str],
        *,
        planner: Planner | None = None,
        budget: Budget | None = None,
    ) -> "TuningSession":
        """Rebuild the session recorded in ``directory``, optionally with a new budget.

        Raises :class:`TuningError` when ``directory`` holds no Auto-Tuner
        session of this version.
        """
        directory = Path(directory).expanduser().resolve()
        recorded = read_session_json(directory)
        limits = recorded["limits"]
        session = cls(
            WorkloadFactory(**recorded["workload_factory"]),
            directory=directory,
            goal=Goal(**recorded["goal"]),
            limits=Limits(
                accuracy={
                    name: tuple(bound) for name, bound in limits["accuracy"].items()
                },
                max_memory_mb=limits["max_memory_mb"],
            ),
            budget=budget or Budget(**recorded["budget"]),
            planner=planner or planner_from_json(recorded["planner"]),
            baseline=recorded["baseline_changes"],
            timing=Timing(**recorded["timing"]) if recorded["timing"] else None,
            final_check_pairs=recorded["final_check_pairs"],
            export=recorded["export"],
            python=recorded["python"],
            noise_control=NoiseControl(**recorded["noise_control"]),
        )
        session.cwd = recorded["cwd"]
        return session

    def _identity(self) -> dict[str, Any]:
        """The settings that make two runs the same session.

        The budget, the planner, the noise control and the Python executable
        may change between resumes; these may not.
        """
        return {
            "format": _FORMAT,
            "version": _VERSION,
            "workload_factory": self.workload_factory.to_json(),
            "goal": asdict(self.goal),
            "limits": self.limits.to_json(),
            "baseline_changes": self.baseline_changes,
            "timing": asdict(self.timing) if self.timing else None,
        }

    def _start_or_resume(self) -> dict[str, Any]:
        """Create ``session.json``, or check that the recorded one is this session.

        Returns its contents, updated with this run's budget, planner and
        settings. Copies the guidance to ``guidance.md``.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "session.json"
        identity = self._identity()
        session_json: dict[str, Any]
        if path.is_file():
            recorded = read_session_json(self.directory)
            mismatched = [
                name for name in identity if recorded.get(name) != identity[name]
            ]
            if mismatched:
                raise TuningError(
                    f"{self.directory} holds a different session (it differs in "
                    f"{', '.join(mismatched)}); use another directory"
                )
            session_json = recorded
        else:
            session_json = {**identity, "elapsed_s": 0.0, "created": time.time()}
        session_json.update(
            budget=asdict(self.budget),
            planner=self.planner.to_json(),
            final_check_pairs=self.final_check_pairs,
            export=self.export,
            python=self.python,
            cwd=self.cwd,
            noise_control=asdict(self.noise_control),
        )
        if self.guidance is not None:
            # guidance.md starts as the guidance and may be edited between
            # resumes; it is replaced only when different guidance is passed.
            guidance_file = self.directory / "guidance.md"
            sha = hashlib.sha256(self.guidance.encode()).hexdigest()
            if (
                not guidance_file.is_file()
                or session_json.get("guidance_sha256") != sha
            ):
                guidance_file.write_text(self.guidance)
            elif guidance_file.read_text() != self.guidance:
                logger.info("Keeping the edited %s", guidance_file)
            session_json["guidance_sha256"] = sha
        write_json(path, session_json)
        return session_json

    # ------------------------------------------------------ the steps of a run

    def run(self) -> TuningResult:
        """Run (or resume) the session and return its result.

        The steps: start or resume, record the reference (and check that the
        workload and the environment did not change), measure the baseline
        and the noise, prepare kernels, search, run the final check, export
        the winner and write the report. Each step records its results under
        ``directory``, so a stopped run resumes where it stopped.

        The first run records (pins) in ``session.json`` the workload (its
        knobs, baseline, accuracy defaults, editable files and the SHA-256 of
        the workload factory's file and the ``.py`` files beside it) and the
        environment (software and GPU, and the git commit and uncommitted
        changes of the Torch-TensorRT source). A later run refuses to
        continue, and so to reuse the reference outputs, if any of them
        changed. A directory that holds no Auto-Tuner session of this
        version is refused with a :class:`TuningError`.
        """
        # 1. Start or resume.
        progress = _Progress(self.directory / "session.json", self._start_or_resume())
        self._preparations, self._pins = [], {}

        # 2. Record the reference, and check that the workload and the
        # environment are the ones the first run recorded.
        described = self._record_reference()
        workload = described["workload"]
        knobs = knobs_by_name([Knob.from_json(knob) for knob in workload["knobs"]])
        self._pin_workload_and_environment(progress.session_json, described)
        editable_files = {
            editable["name"]: EditableFile.from_json(editable)
            for editable in workload["editable_files"]
        }
        timing = self.timing or Timing(**workload["timing"])
        bounds = self.limits.bounds(
            {
                name: tuple(bound)
                for name, bound in workload["accuracy_defaults"].items()
            }
        )
        progress.checkpoint()

        # 3. Measure the baseline (unless an earlier run validated it) and the noise.
        trials = self._load_trials()
        baseline_config = apply_changes(
            workload["baseline"], self.baseline_changes, knobs
        )
        recorded_baseline = trials[0] if trials else {}
        if not _is_validated_baseline(recorded_baseline):
            measured = self._measure_baseline(
                recorded_baseline, baseline_config, timing, bounds
            )
            trials = [measured] + trials[1:]
            progress.checkpoint()
        baseline = trials[0]
        if baseline["decision"]["decision"] != "baseline":
            raise TuningError(
                f"The baseline is not valid, so nothing can be compared with it: "
                f"{baseline['decision']['reason']}. See {self._trial_dir(0)}/worker.log"
            )
        noise = self._measure_noise(baseline, timing, bounds)
        progress.checkpoint()

        # 4. Prepare the knob values that need it, such as values that
        # generate region kernels, before any run uses them.
        kernels = self._prepare_kernels(knobs, trials, progress)

        # 5. Search: take up what earlier runs decided, then run new trials.
        search = self._replay_recorded_trials(trials, baseline, timing, bounds)
        stop_reason = self._search(
            search, knobs, editable_files, timing, bounds, progress
        )

        # 6. The final check; 7. the export and the report.
        final_check = self._final_check(baseline, search.best, timing, bounds)
        winner = search.best if final_check and final_check["passed"] else baseline
        decision = "accepted" if winner is not baseline else "baseline kept"
        export = self._export(winner, bounds) if self.export else None
        artifact = (
            self.directory / "artifact"
            if export and export["status"] == "exported"
            else None
        )
        write_trials_table(self.directory, search.trials, self.goal)
        progress.checkpoint()
        report_path = write_report(
            self.directory,
            goal=self.goal,
            bounds=bounds,
            knobs=knobs,
            trials=search.trials,
            baseline=baseline,
            winner=winner,
            decision=decision,
            stop_reason=stop_reason,
            final_check=final_check,
            artifact=artifact,
            environment=progress.session_json["environment"],
            planner=self.planner.to_json(),
            elapsed_s=progress.elapsed(),
            noise=noise,
            export=export,
            workload=workload,
            kernels=kernels,
        )
        self._log_event("finished", decision=decision, winner=winner["id"])
        return TuningResult(
            directory=self.directory,
            decision=decision,
            stop_reason=stop_reason,
            config=winner["config"],
            baseline=self.summary(baseline, self.goal),
            winner=self.summary(winner, self.goal),
            final_check=final_check,
            artifact=artifact,
            report_path=report_path,
            noise=noise,
            export_summary=export,
        )

    def _record_reference(self) -> dict[str, Any]:
        """Have a worker describe the workload and, once, save its reference outputs.

        The reference outputs are computed once per session, from the
        original code, and saved in ``reference.pt``. A resumed session
        describes the workload again (a ``describe`` job), so that a changed
        workload or environment is detected instead of mixed with old trials.
        """
        path = self.directory / "reference" / "result.json"
        have_reference = (
            path.is_file()
            and (self.directory / "reference.pt").is_file()
            and json.loads(path.read_text()).get("status") == "ok"
        )
        kind = "describe" if have_reference else "reference"
        result = self._run_worker({"kind": kind}, self.directory / kind)
        if result.get("status") != "ok":
            raise TuningError(
                f"Preparing the workload ({kind}) failed: {failure_reason(result)}. "
                f"See {self.directory / kind}/worker.log"
            )
        self._log_event(kind, workload=result["workload"]["name"])
        return result

    def _pin_workload_and_environment(
        self, session_json: dict[str, Any], described: Mapping[str, Any]
    ) -> None:
        """Record the workload and environment, or refuse them if they changed.

        The first run records (pins) them. A later run raises
        :class:`TuningError` if they differ from what was recorded, since the
        recorded measurements and reference outputs would no longer apply.
        Warns when the Torch-TensorRT source cannot be pinned.
        """
        # A run stopped before its first checkpoint left the reference's
        # record as the only pin.
        reference_path = self.directory / "reference" / "result.json"
        reference = (
            json.loads(reference_path.read_text()) if reference_path.is_file() else {}
        )
        for key, what in (
            (
                "workload",
                "workload's knobs, baseline, accuracy defaults or source files",
            ),
            ("environment", "software, Torch-TensorRT source or GPU"),
        ):
            recorded = session_json.get(key) or reference.get(key)
            if recorded is not None:
                self._check_unchanged(key, what, recorded, described[key])
            session_json[key] = described[key]
        source = described["environment"].get("torch_tensorrt_source")
        if isinstance(source, Mapping) and source.get("unavailable"):
            logger.warning(
                "The Torch-TensorRT source cannot be pinned (%s), so a resume of "
                "%s will not notice edits to it",
                source["unavailable"],
                self.directory,
            )

    def _check_unchanged(
        self,
        key: str,
        what: str,
        recorded: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> None:
        """Refuse a workload or environment that differs from the one recorded."""
        if dict(recorded) != dict(current):
            changes = _differences(recorded, current)
            shown = "; ".join(changes[:8]) + (
                f"; and {len(changes) - 8} more" if len(changes) > 8 else ""
            )
            raise TuningError(
                f"The {what} changed since {self.directory} was created, so its "
                f"recorded measurements and reference outputs no longer apply; "
                f"start a new session directory ({key}: {shown or 'it differs'})"
            )

    def _measure_baseline(
        self,
        previous: Mapping[str, Any],
        config: Config,
        timing: Timing,
        bounds: Mapping[str, tuple[str, float]],
    ) -> dict[str, Any]:
        """Measure the baseline config as trial 0000 and decide whether it is valid.

        Every candidate is compared with the baseline, so a baseline that
        does not qualify (see :meth:`check_result`) ends the session. This
        runs on the first run, and again on a resume after the baseline
        failed (``previous`` is its old record), since the cause may be gone
        now: a busy GPU, a ``trial_timeout_s`` since raised, a crash.
        """
        previous_decision = previous.get("decision") or {}
        if previous_decision:
            logger.warning(
                "The baseline failed when this session last ran (%s); "
                "measuring it again",
                previous_decision.get("reason"),
            )
            self._log_event("baseline_retry", previous=previous_decision.get("reason"))
        record: dict[str, Any] = {
            "id": "0000",
            "parent": None,
            "config": config,
            "changes": {},
            "digest": digest(config),
        }
        record["result"] = self._measure_trial(0, record, timing)
        qualifies, reason = self.check_result(record["result"], None, bounds)
        record["decision"] = {
            "decision": "baseline" if qualifies else "failed",
            "reason": reason,
        }
        write_json(self._trial_dir(0) / "decision.json", record["decision"])
        return record

    def _measure_noise(
        self,
        baseline: dict[str, Any],
        timing: Timing,
        bounds: Mapping[str, tuple[str, float]],
    ) -> dict[str, Any]:
        """Measure the noise: how far the baseline's metric moves between fresh runs.

        ``noise_control.noise_runs`` more fresh-process runs of the baseline
        config, with its first measurement, give the spread: (max - min) /
        median of the metric over the runs that qualify. The noise threshold
        is ``goal.min_improvement`` or ``noise_control.noise_factor`` times
        the spread, whichever is larger. A resumed session reuses its noise
        runs if they were made with the same number of runs and measured a
        spread. Returns the summary also written to ``noise/summary.json``.
        """
        path = self.directory / "noise" / "summary.json"
        requested = self.noise_control.noise_runs
        summary: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {}
        if summary.get("requested") != requested or (
            requested and summary.get("spread") is None
        ):
            values = [metric_value(baseline["result"], self.goal)]
            runs = []
            for index in range(self.noise_control.noise_runs):
                result = self._measure_config(
                    baseline["config"],
                    timing,
                    self.directory / "noise" / str(index),
                )
                qualifies, reason = self.check_result(result, None, bounds)
                runs.append({"qualified": qualifies, "reason": reason})
                if qualifies:
                    values.append(metric_value(result, self.goal))
            spread = (
                (max(values) - min(values)) / statistics.median(values)
                if len(values) > 1
                else None
            )
            summary = {
                "requested": requested,
                "metrics": values,
                "spread": spread,
                "runs": runs,
            }
            if requested and spread is None:
                logger.warning(
                    "No noise run qualified, so the noise threshold is the goal's "
                    "min_improvement; see %s",
                    path.parent,
                )
        self.noise_spread = summary["spread"]
        self.noise_threshold = max(
            self.goal.min_improvement,
            self.noise_control.noise_factor * (self.noise_spread or 0.0),
        )
        summary["noise_threshold"] = self.noise_threshold
        write_json(path, summary)
        self._log_event(
            "noise", spread=self.noise_spread, noise_threshold=self.noise_threshold
        )
        return summary

    def _prepare_kernels(
        self,
        knobs: Mapping[str, Knob],
        trials: list[dict[str, Any]],
        progress: _Progress,
    ) -> list[dict[str, Any]]:
        """Prepare each knob value that declares ``prepare``, once, before the search.

        A value the baseline config already sets needs no preparation: the
        baseline's own build made it. For every other one, a ``prepare``
        worker (``kernels/<knob>=<value>``) builds the baseline config with
        the value and the settings it requires, within
        ``budget.prepare_timeout_s``, without timing it, and lists the region
        kernels the build generated or loaded. A value whose preparation
        failed cannot be proposed (:meth:`_config_from_proposal`).

        Before each preparation, the search's stop conditions are checked as
        they will stand before the next trial (the goal target, ``max_trials``,
        ``max_time_s`` and ``patience``, see :meth:`_stop_before_trials`); once
        one holds, only values that recorded trials use are prepared, for the
        final check and the export, and the others are ``skipped``. A resume
        reuses a preparation that succeeded with the same config and prepares
        a failed or interrupted one again.

        After the preparations, the kernels their builds registered, in the
        order the preparations finished, give each generation task its final
        kernel (:func:`_kernel_pins`): the one every run that uses a prepared
        value must run for that task (:meth:`_run_worker`). A task whose
        kernel a later preparation changed gets a warning. Each value lists,
        as ``regions``, the kernels of the tasks the baseline's build did not
        register, with their final kernel. Returns the entries written to
        ``kernels/summary.json`` (``name``, ``knob``, ``value``, ``status``,
        ``reason``, ``config``, ``build_s``, ``regions``, ``region_speedup``
        and ``warnings``), which also holds the ``pins`` and the ``warnings``.
        """
        baseline = trials[0]
        wanted = [
            (name, value)
            for name, knob in knobs.items()
            for value in knob.prepare
            if not is_member(baseline["config"].get(name, knob.default), (value,))
        ]
        if not wanted:
            return []
        root = self.directory / "kernels"
        # Configs that later runs may build without a new trial: the final
        # check's, the export's and those of trials replayed on resume.
        recorded = [
            record["config"]
            for record in trials[1:]
            if (record.get("decision") or {}).get("decision") != "invalid"
        ]
        # The tasks the baseline's build registered are the baseline's regions.
        baseline_tasks = {
            kernel.get("task_sha256")
            for kernel in (baseline.get("result") or {}).get("kernels") or ()
        }
        entries: list[dict[str, Any]] = []
        finished: list[tuple[int, int, dict[str, Any], Mapping[str, Any]]] = []
        for index, ((name, value), label) in enumerate(
            zip(wanted, _preparation_names(wanted))
        ):
            entry: dict[str, Any] = {"name": label, "knob": name, "value": value}
            try:
                config = apply_changes(
                    baseline["config"],
                    with_requirements(baseline["config"], {name: value}, knobs),
                    knobs,
                )
            except ValueError as error:
                entries.append(
                    _unprepared(entry, "failed", f"its config is invalid: {error}")
                )
                continue
            directory = root / label
            recorded_result = self._recorded_preparation(directory, config)
            succeeded = (recorded_result or {}).get("status") == "ok"
            stop = ""
            if not succeeded and not any(
                _sets(other, name, value) for other in recorded
            ):
                stop = self._stop_before_trials(trials, progress.elapsed())
            if recorded_result is not None and (succeeded or stop):
                # Reuse a success; keep a failure that no trial could use.
                result, reused = recorded_result, True
            elif stop:
                entries.append(
                    _unprepared(
                        entry,
                        "skipped",
                        f"the search runs no further trial ({stop})",
                        config,
                    )
                )
                continue
            else:
                self._log_event("prepare", name=label)
                result = self._run_worker(
                    {
                        "kind": "prepare",
                        "config": config,
                        "timeout_s": self.budget.prepare_timeout_s,
                    },
                    directory,
                )
                reused = False
                progress.checkpoint()
            entries.append(_preparation_entry(entry, config, result, baseline_tasks))
            finished.append((_finished_at(directory), index, entries[-1], result))
            self._log_event(
                "prepared",
                name=label,
                status=entries[-1]["status"],
                reason=entries[-1]["reason"],
                reused=reused,
            )
        self._pins, warnings = _kernel_pins(finished)
        for entry in entries:
            _use_final_kernels(entry, self._pins, warnings)
        for warning in warnings:
            logger.warning("%s", warning["message"])
            self._log_event("kernel_changed", **warning)
        write_json(
            root / "summary.json",
            {
                "preparations": entries,
                "pins": [
                    {
                        key: pin[key]
                        for key in ("task_sha256", "region", "kernel_sha256", "values")
                    }
                    for pin in self._pins.values()
                ],
                "warnings": warnings,
            },
        )
        self._preparations = entries
        return entries

    def _recorded_preparation(
        self, directory: Path, config: Config
    ) -> dict[str, Any] | None:
        """The result of an earlier preparation of ``config``, if one finished."""
        request_path, result_path = (
            directory / "request.json",
            directory / "result.json",
        )
        if not (request_path.is_file() and result_path.is_file()):
            return None
        if json.loads(request_path.read_text()).get("config") != config:
            return None
        result: dict[str, Any] = json.loads(result_path.read_text())
        return result

    def _stop_before_trials(
        self, trials: list[dict[str, Any]], elapsed_s: float
    ) -> str:
        """Why the search would stop before its next trial, or ``""`` if it would not.

        The recorded trials are counted as the replay will count them (see
        :meth:`_stop_reason`): a trial without a decision counts as measured,
        since the replay measures it again, and the count of invalid
        proposals in a row starts anew.
        """
        state = _SearchState(trials=list(trials), best=trials[0])
        for record in trials[1:]:
            if record.get("decision"):
                state.count(record)
            else:
                state.measured += 1
        state.invalid_in_a_row = 0
        return self._stop_reason(state, elapsed_s)

    def _prepared_values(self, config: Mapping[str, Any]) -> list[dict[str, Any]]:
        """The preparations of the knob values that ``config`` sets."""
        return [
            entry
            for entry in self._preparations
            if _sets(config, entry["knob"], entry["value"])
        ]

    def _replay_recorded_trials(
        self,
        trials: list[dict[str, Any]],
        baseline: dict[str, Any],
        timing: Timing,
        bounds: Mapping[str, tuple[str, float]],
    ) -> _SearchState:
        """Take up the trials earlier runs recorded: the search state to go on from.

        A decided trial is reused as it is. A trial interrupted before its
        decision is measured and decided again. An invalid proposal stopped
        between its two writes gets its ``decision.json`` back from its
        ``trial.json``.
        """
        search = _SearchState(trials=trials, best=baseline)
        for number, record in enumerate(trials[1:], start=1):
            if "decision" not in record:
                record["result"] = self._measure_trial(number, record, timing)
                record["decision"] = self._decide(
                    number, record, search.best, timing, bounds
                )
                write_json(
                    self._trial_dir(number) / "decision.json", record["decision"]
                )
            elif not (self._trial_dir(number) / "decision.json").is_file():
                write_json(
                    self._trial_dir(number) / "decision.json", record["decision"]
                )
            search.count(record)
        # A resumed run (perhaps with an edited guidance.md) gets a fresh count
        # of consecutive invalid proposals.
        search.invalid_in_a_row = 0
        return search

    def _search(
        self,
        search: _SearchState,
        knobs: Mapping[str, Knob],
        editable_files: Mapping[str, EditableFile],
        timing: Timing,
        bounds: Mapping[str, tuple[str, float]],
        progress: _Progress,
    ) -> str:
        """Run trials until a stop condition holds; return why the search stopped.

        Each trial: ask the planner, check its proposal, then measure and
        decide the candidate (:meth:`_run_trial`).
        """
        while True:
            stop_reason = self._stop_reason(search, progress.elapsed())
            if stop_reason:
                return stop_reason
            number = len(search.trials)
            directory = self._trial_dir(number)
            context = self._planner_context(
                knobs,
                search.trials,
                search.best,
                bounds,
                search.measured,
                progress.elapsed(),
                editable_files,
            )
            write_json(directory / "context.json", context)
            proposal, failure = self._ask_planner(context, directory / "planner")
            proposal_json = self._record_proposal(number, proposal, failure)
            if proposal.stop:
                stop_reason = (
                    f"the planner stopped: {proposal.rationale}"
                    if failure is None
                    else f"the {self.planner.name} planner failed ({failure}) and "
                    f"the rule-based planner that stood in stopped: "
                    f"{proposal.rationale}"
                )
                self._log_event("stop", reason=stop_reason)
                return stop_reason
            record = self._run_trial(
                number,
                proposal,
                proposal_json,
                search,
                knobs,
                editable_files,
                timing,
                bounds,
            )
            search.trials.append(record)
            search.count(record)
            write_trials_table(self.directory, search.trials, self.goal)
            progress.checkpoint()

    def _stop_reason(self, search: _SearchState, elapsed_s: float) -> str:
        """Why the search stops before the next trial, or ``""`` to go on."""
        if self.goal.reached(metric_value(search.best["result"], self.goal)):
            return "goal target reached"
        if search.measured >= self.budget.max_trials:
            return f"trial budget of {self.budget.max_trials} used"
        if self.budget.max_time_s is not None and elapsed_s >= self.budget.max_time_s:
            return f"time budget of {self.budget.max_time_s:g} s used"
        if (
            self.budget.patience is not None
            and search.since_accepted >= self.budget.patience
        ):
            return f"no accepted candidate in the last {search.since_accepted} trials"
        if search.invalid_in_a_row >= _MAX_INVALID_PROPOSALS:
            return (
                f"the planner made {search.invalid_in_a_row} invalid proposals in a row"
            )
        return ""

    # --------------------------------------------------- asking the planner

    @staticmethod
    def summary(record: Mapping[str, Any], goal: Goal) -> dict[str, Any]:
        """A trial as the planner context and the result show it.

        Its config and status and, if it was measured, its metric, latency,
        throughput, accuracy, memory and build time; else where and why it
        failed. Its structure, profile, bottleneck hints, rooflines and
        timing stability come along when it has them.
        """
        result = record.get("result") or {}
        summary: dict[str, Any] = {
            "trial": record["id"],
            "config": record["config"],
            "status": result.get("status"),
        }
        if result.get("status") == "ok":
            summary.update(
                metric=metric_value(result, goal),
                latency_ms={
                    key: result["latency_ms"][key] for key in ("median", "p25", "p75")
                },
                throughput=result["throughput"],
                accuracy=result["accuracy"],
                memory=result.get("memory"),
                build_s=result.get("build_s"),
            )
        else:
            summary.update(stage=result.get("stage"), error=result.get("error"))
            for key in ("exit_status", "signal", "log_error"):
                if result.get(key) is not None:
                    summary[key] = result[key]
        for key in (
            "structure",
            "profile",
            "bottleneck_hints",
            "roofline",
            "compute_roofline",
            "timing_stability",
        ):
            if key in result:
                summary[key] = result[key]
        return summary

    def _planner_context(
        self,
        knobs: Mapping[str, Knob],
        trials: list[dict[str, Any]],
        best: dict[str, Any],
        bounds: Mapping[str, tuple[str, float]],
        used: int,
        elapsed_s: float,
        editable_files: Mapping[str, EditableFile] | None = None,
    ) -> dict[str, Any]:
        """What the planner is given (``context.json`` of each trial).

        The goal, the noise, the limits and the budget left; the baseline and
        the best config with its bottleneck hints; a history entry per trial
        (:meth:`_history_entry`); the knobs with their current values; the
        configs already tried; the editable files as the best config has
        them; the prepared knob values with the region kernels each pins
        (``kernels``, see :func:`_planner_kernels`); and the user's guidance.
        """
        history = [self._history_entry(record) for record in trials[1:]]
        current = best["config"]
        context: dict[str, Any] = {
            "goal": asdict(self.goal),
            "noise": {
                "between_process_spread": self.noise_spread,
                "noise_threshold": self.noise_threshold,
                "head_to_head_pairs": self.noise_control.head_to_head_pairs,
            },
            "limits": {
                "accuracy": {name: list(bound) for name, bound in bounds.items()},
                "max_memory_mb": self.limits.max_memory_mb,
            },
            "budget": {
                "trials_used": used,
                "trials_left": self.budget.max_trials - used,
                "time_left_s": (
                    None
                    if self.budget.max_time_s is None
                    else max(0.0, self.budget.max_time_s - elapsed_s)
                ),
            },
            "baseline": self.summary(trials[0], self.goal),
            "best": {
                **self.summary(best, self.goal),
                "bottleneck_hints": (best.get("result") or {}).get(
                    "bottleneck_hints", []
                ),
            },
            "history": history,
            "knobs": [
                {**knob.to_json(), "current": current.get(name, knob.default)}
                for name, knob in knobs.items()
            ],
            # Readable for the planner; the session itself de-duplicates by digest.
            "tried": [
                record["config"]
                for record in trials
                if record.get("decision", {}).get("decision") != "invalid"
            ],
        }
        store = EditedFileStore(self.directory)
        if editable_files:
            context["editable_files"] = [
                {
                    **editable.to_json(),
                    "edited": current.get(editable.key),
                    "editable_blocks": editable_blocks(store.source(editable, current)),
                }
                for editable in editable_files.values()
            ]
        if self._preparations:
            context["kernels"] = [
                _planner_kernels(entry, current) for entry in self._preparations
            ]
        guidance = self.directory / "guidance.md"
        if guidance.is_file():
            context["guidance"] = guidance.read_text()[:20000]
        return context

    def _history_entry(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """What the planner sees of one earlier trial.

        What it changed and why, its status and decision, and either its
        metric and bottleneck hints or where and why it failed (with the
        start of its worker's stack dump or the end of its log).
        """
        result = record.get("result") or {}
        entry = {
            "trial": record["id"],
            "parent": record.get("parent"),
            "changes": record.get("changes", {}),
            "rationale": record.get("proposal", {}).get("rationale", ""),
            # Only an invalid proposal has no result: it was never measured.
            "status": result.get("status", "invalid"),
            "decision": record.get("decision", {}).get("decision"),
            "reason": record.get("decision", {}).get("reason"),
        }
        if "proposed" in record:
            entry["proposed"] = record["proposed"]
        if result.get("status") == "ok":
            entry["metric"] = metric_value(result, self.goal)
            entry["bottleneck_hints"] = [
                hint["name"] for hint in result.get("bottleneck_hints", ())
            ]
            head_to_head = record.get("decision", {}).get("head_to_head")
            if head_to_head:
                entry["paired_gains"] = head_to_head.get("gains")
        elif result:
            entry["stage"] = result.get("stage")
            entry["error"] = failure_detail(result)
            for key in ("exit_status", "signal"):
                if result.get(key) is not None:
                    entry[key] = result[key]
            excerpt = _error_excerpt(result)
            if excerpt and excerpt != entry["error"]:
                entry["error_tail"] = excerpt
        return entry

    def _ask_planner(
        self, context: Mapping[str, Any], workdir: Path
    ) -> tuple[Proposal, str | None]:
        """The planner's proposal, or, if it raises, a :class:`RuleBasedPlanner`'s.

        A planner that returns anything but a well-formed :class:`Proposal`
        (see :func:`_malformed`) has failed too. Returns the proposal and the
        planner's error (None if it worked); the traceback goes to
        ``workdir/error.txt``.
        """
        try:
            proposal = self.planner.propose(context, workdir)
            if not isinstance(proposal, Proposal):
                raise TypeError(
                    f"{type(self.planner).__name__}.propose returned "
                    f"{type(proposal).__name__}, not a Proposal"
                )
            problem = _malformed(proposal)
            if problem:
                raise TypeError(
                    f"{type(self.planner).__name__}.propose returned a malformed "
                    f"Proposal: {problem}"
                )
            return proposal, None
        except Exception as error:  # noqa: BLE001 - a planner failure is recorded
            failure = _error_text(error)
            workdir.mkdir(parents=True, exist_ok=True)
            (workdir / "error.txt").write_text(traceback.format_exc())
        logger.warning(
            "The %s planner failed (%s); a rule-based planner proposes this trial",
            self.planner.name,
            failure,
        )
        fallback = RuleBasedPlanner()
        try:
            proposal = fallback.propose(context, workdir)
        except Exception as error:  # noqa: BLE001 - stop with both reasons
            return (
                Proposal(
                    stop=True,
                    rationale=f"it failed too ({_error_text(error)})",
                    planner=fallback.name,
                ),
                failure,
            )
        note = (
            f"the {self.planner.name} planner failed ({failure}); the "
            f"{fallback.name} planner proposed instead"
        )
        return (
            replace(proposal, planner=proposal.planner or fallback.name, note=note),
            failure,
        )

    def _record_proposal(
        self, number: int, proposal: Proposal, failure: str | None
    ) -> dict[str, Any]:
        """Write ``proposal.json``, with the planner's error if it failed."""
        proposal_json = {
            "changes": dict(proposal.changes),
            "rationale": proposal.rationale,
            "stop": proposal.stop,
            "planner": proposal.planner or self.planner.name,
            "note": proposal.note,
            "edits": {
                name: [dict(edit) for edit in edits]
                for name, edits in proposal.edits.items()
            },
        }
        if failure is not None:
            proposal_json["planner_error"] = failure
            self._log_event("planner_failed", trial=f"{number:04d}", error=failure)
        write_json(self._trial_dir(number) / "proposal.json", proposal_json)
        return proposal_json

    # ------------------------------------------------------ running a trial

    def _run_trial(
        self,
        number: int,
        proposal: Proposal,
        proposal_json: dict[str, Any],
        search: _SearchState,
        knobs: Mapping[str, Knob],
        editable_files: Mapping[str, EditableFile],
        timing: Timing,
        bounds: Mapping[str, tuple[str, float]],
    ) -> dict[str, Any]:
        """Check the proposal, then measure and decide the candidate it leads to.

        Returns the trial's record with its decision. An invalid proposal is
        recorded with the decision ``invalid`` and never measured.
        """
        directory = self._trial_dir(number)
        best = search.best
        record: dict[str, Any] = {
            "id": f"{number:04d}",
            "parent": best["id"],
            "proposal": proposal_json,
        }
        try:
            config = self._config_from_proposal(
                proposal, best["config"], search.trials, knobs, editable_files
            )
        except ValueError as error:
            record.update(
                config=best["config"],
                changes=dict(proposal.changes),
                digest=f"invalid-{number}",
                decision={
                    "decision": "invalid",
                    "reason": f"proposal rejected: {error}",
                },
            )
            write_json(directory / "trial.json", record)
            write_json(directory / "decision.json", record["decision"])
            return record
        changes = difference(best["config"], config, knobs)
        record.update(
            config=config,
            changes=changes,
            # What the planner chose, without the settings that came along
            # as requirements: those stay untried on their own.
            proposed=chosen_changes(changes, knobs),
            digest=digest(config),
        )
        record["result"] = self._measure_trial(number, record, timing)
        record["decision"] = self._decide(number, record, best, timing, bounds)
        write_json(directory / "decision.json", record["decision"])
        self._log_event(
            "decision",
            trial=record["id"],
            decision=record["decision"]["decision"],
            reason=record["decision"]["reason"],
        )
        return record

    def _config_from_proposal(
        self,
        proposal: Proposal,
        best_config: Config,
        trials: Sequence[Mapping[str, Any]],
        knobs: Mapping[str, Knob],
        editable_files: Mapping[str, EditableFile],
    ) -> Config:
        """Check a proposal before anything is built; return the config it leads to.

        Applies the proposal's edits to the best config's files (storing each
        edited file), checks its changes against the knobs (see
        :func:`check_proposal`), and refuses a config that uses a knob value
        whose preparation failed or a config that was already measured.
        Raises ValueError saying why the proposal is invalid.
        """
        edited = self._apply_proposed_edits(proposal, best_config, editable_files)
        config = check_proposal(
            replace(proposal, changes={**proposal.changes, **edited}),
            best_config,
            knobs,
        )
        unusable = [
            entry for entry in self._prepared_values(config) if entry["status"] != "ok"
        ]
        if unusable:
            raise ValueError(
                "the config uses "
                + "; ".join(_unusable_reason(entry) for entry in unusable)
            )
        if digest(config) in {record["digest"] for record in trials}:
            raise ValueError("this config was already measured")
        return config

    def _apply_proposed_edits(
        self,
        proposal: Proposal,
        config: Mapping[str, Any],
        editable_files: Mapping[str, EditableFile],
    ) -> dict[str, str]:
        """Apply the proposal's edits to the best config's files; store each edited file.

        Returns the config changes (``edit.<name>`` -> digest, or ``""`` for
        the original file). Raises ValueError for an unknown file, an edit
        outside the fences, a file that does not compile, edits that change
        nothing, or an ``edit.<name>`` change naming no stored edited file.
        """
        store = EditedFileStore(self.directory)
        for key, value in proposal.changes.items():
            if not is_edit_key(key):
                continue
            if key[len(EDIT_PREFIX) :] not in editable_files:
                raise ValueError(
                    f"Unknown editable file {key!r}; the workload has "
                    f"{sorted(editable.key for editable in editable_files.values())}"
                )
            if value and not store.path(value).is_file():
                raise ValueError(f"{key} names no stored edited file: {value!r}")
        changes = {}
        for name, edits in proposal.edits.items():
            editable = editable_files.get(name)
            if editable is None:
                raise ValueError(
                    f"Unknown editable file {name!r}; the workload has "
                    f"{sorted(editable_files)}"
                )
            source = store.source(editable, config)
            edited = apply_edits(source, edits)
            if edited == source:
                raise ValueError(f"the edits of {name} change nothing")
            check_compiles(edited, editable.path)
            # Edits that restore the original file select the original.
            original = Path(editable.path).read_text()
            changes[editable.key] = "" if edited == original else store.save(edited)
        return changes

    def _measure_trial(
        self, number: int, record: dict[str, Any], timing: Timing
    ) -> dict[str, Any]:
        """Write ``trial.json``, then measure the trial's config once, with a profile.

        This is the trial's first measurement, in a fresh worker process.
        """
        directory = self._trial_dir(number)
        write_json(directory / "trial.json", record)
        self._log_event("trial", trial=record["id"], changes=record.get("changes"))
        return self._measure_config(record["config"], timing, directory, profile=True)

    def _decide(
        self,
        number: int,
        record: dict[str, Any],
        best: dict[str, Any],
        timing: Timing,
        bounds: Mapping[str, tuple[str, float]],
    ) -> dict[str, Any]:
        """Accept or reject a measured trial.

        Its first measurement must beat the best config (see
        :meth:`check_result`). With head-to-head runs on, the candidate must
        then also win them (see :meth:`_head_to_head`). Returns the decision
        written to ``decision.json``: ``accepted`` or ``rejected`` with the
        reason and, after head-to-head runs, the first measurement's reason
        and the runs' summary. A build that needed kernels no preparation
        made measured nothing: its decision is ``invalid``, so it spends no
        trial and no patience, and the planner sees why in the history.
        """
        if record["result"].get("unprepared"):
            # Not a measurement: the config needs kernels no preparation made.
            return {
                "decision": "invalid",
                "reason": f"proposal rejected: {record['result']['error']}",
            }
        accepted, reason = self.check_result(record["result"], best["result"], bounds)
        if not (accepted and self.noise_control.head_to_head_pairs):
            return {
                "decision": "accepted" if accepted else "rejected",
                "reason": reason,
            }
        head_to_head = self._head_to_head(
            best,
            record,
            timing,
            bounds,
            self._trial_dir(number) / "head_to_head",
            self.noise_control.head_to_head_pairs,
            ("best", "candidate"),
        )
        gain = head_to_head["median_gain"]
        if head_to_head["passed"]:
            verdict = (
                f"won head-to-head: median paired gain {gain:+.1%} over "
                f"{len(head_to_head['gains'])} interleaved pairs (noise threshold "
                f"{self.noise_threshold:.1%})"
            )
        elif head_to_head["problems"]:
            verdict = "head-to-head failed: " + "; ".join(head_to_head["problems"])
        else:
            gains = ", ".join(f"{g:+.1%}" for g in head_to_head["gains"])
            verdict = (
                f"lost head-to-head: gains {gains} (every pair must gain and the "
                f"median must exceed {self.noise_threshold:.1%})"
            )
        return {
            "decision": "accepted" if head_to_head["passed"] else "rejected",
            "reason": verdict,
            "first_measurement": reason,
            "head_to_head": head_to_head,
        }

    def check_result(
        self,
        result: Mapping[str, Any],
        best: Mapping[str, Any] | None,
        bounds: Mapping[str, tuple[str, float]],
    ) -> tuple[bool, str]:
        """Whether ``result`` qualifies and, given ``best``, beats it in its own run.

        A result qualifies when its worker succeeded, it met the accuracy
        bounds (checked before and after timing) and the memory limit, and
        it was measured stably. Without ``best`` (for the baseline, a noise
        run, a head-to-head run or a final check run) that is all. With
        ``best``, the result must also improve the goal metric by more than
        the noise threshold; when head-to-head runs are off, its latency
        quartiles must also lie entirely on the better side of the best
        config's. With head-to-head runs on, passing here only earns the
        candidate its head-to-head runs. Returns ``(passed, reason)``. A
        failed result's reason is its stage and the first line of its
        error, with the signal that ended its worker, if any.
        """
        if result.get("status") != "ok":
            return False, failure_reason(result)
        missed = _accuracy_misses(result, bounds)
        if missed:
            return False, "accuracy: " + "; ".join(missed)
        memory = memory_mb(result)
        limit = self.limits.max_memory_mb
        if limit is not None and (memory is None or memory > limit):
            return False, f"memory {memory} MB exceeds max_memory_mb={limit:g}"
        stability = result.get("timing_stability") or {}
        if stability.get("stable") is False:
            spreads = stability.get("spreads") or [float("nan")]
            busy = ", the GPU was busy" if stability.get("gpu_quiet") is False else ""
            return False, (
                f"unstable measurement: interquartile spread {spreads[-1]:.1%} after "
                f"{len(spreads)} attempts{busy}"
            )
        if best is None:
            return True, "validated baseline"
        value = metric_value(result, self.goal)
        best_value = metric_value(best, self.goal)
        gain = self.goal.gain(best_value, value)
        head_to_head = self.noise_control.head_to_head_pairs > 0
        if gain > self.noise_threshold and (
            head_to_head or _quartiles_separated(result, best)
        ):
            return (
                True,
                f"{self.goal.metric} improved {gain:+.1%} "
                f"({best_value:.4g} -> {value:.4g})",
            )
        needs = f"needs more than {self.noise_threshold:.1%}" + (
            "" if head_to_head else " with non-overlapping quartiles"
        )
        return False, (
            f"no gain beyond noise: {self.goal.metric} {gain:+.1%} "
            f"({best_value:.4g} -> {value:.4g}); {needs}"
        )

    def _head_to_head(
        self,
        defender: dict[str, Any],
        challenger: dict[str, Any],
        timing: Timing,
        bounds: Mapping[str, tuple[str, float]],
        directory: Path,
        pairs: int,
        sides: tuple[str, str],
    ) -> dict[str, Any]:
        """Measure ``defender`` and ``challenger`` in ``pairs`` pairs of fresh runs.

        The two run back to back in each pair, and the order alternates
        between pairs, so a drift (clocks warming up, another job starting)
        does not favor one side. The challenger passes when every run
        qualifies, it gains in every pair and the median gain exceeds the
        noise threshold. ``sides`` names the two in the run directories
        (``<pair>-<side>``) and in the summary. The search uses it for
        head-to-head runs (the best config against a candidate), the final
        check for the baseline against the best config.
        """
        first, second = sides
        runs = []
        for index in range(pairs):
            order = [(first, defender), (second, challenger)]
            if index % 2:
                order.reverse()
            pair = {
                side: self._measure_config(
                    record["config"], timing, directory / f"{index}-{side}"
                )
                for side, record in order
            }
            runs.append(pair)
        gains, problems = [], []
        for index, pair in enumerate(runs):
            for side in sides:
                qualifies, reason = self.check_result(pair[side], None, bounds)
                if not qualifies:
                    problems.append(f"pair {index} {side}: {reason}")
            if pair[first].get("status") == "ok" and pair[second].get("status") == "ok":
                gains.append(
                    self.goal.gain(
                        metric_value(pair[first], self.goal),
                        metric_value(pair[second], self.goal),
                    )
                )
        median = statistics.median(gains) if gains else None
        passed = bool(
            not problems
            and gains
            and min(gains) > 0
            and median is not None
            and median > self.noise_threshold
        )
        return {
            "passed": passed,
            "median_gain": median,
            "gains": gains,
            "problems": problems,
            "noise_threshold": self.noise_threshold,
            "pairs": [
                {side: _run_summary(result, self.goal) for side, result in pair.items()}
                for pair in runs
            ],
        }

    # ------------------------------------------------- the end of a session

    def _final_check(
        self,
        baseline: dict[str, Any],
        best: dict[str, Any],
        timing: Timing,
        bounds: Mapping[str, tuple[str, float]],
    ) -> dict[str, Any] | None:
        """Run the best config head-to-head against the baseline, in fresh runs.

        Each accepted trial only beat the config before it, so the winner of
        the search must also beat the baseline directly. Returns the summary
        also written to ``final_check/summary.json``, or None when no
        candidate was accepted.
        """
        if best is baseline:
            return None
        final_check = self._head_to_head(
            baseline,
            best,
            timing,
            bounds,
            self.directory / "final_check",
            self.final_check_pairs,
            ("baseline", "best"),
        )
        write_json(self.directory / "final_check" / "summary.json", final_check)
        self._log_event(
            "final_check",
            passed=final_check["passed"],
            median_gain=final_check["median_gain"],
        )
        return final_check

    def _export(
        self, winner: dict[str, Any], bounds: Mapping[str, tuple[str, float]]
    ) -> dict[str, Any]:
        """Rebuild the winner in a worker that saves it, and check that rebuild.

        The worker exports into ``export/files``. Those files, with the
        winner's ``config.json``, replace ``artifact/`` only if the rebuilt
        candidate's outputs, checked before and after its short timing, meet
        the accuracy bounds. An export that does not pass leaves
        ``artifact/`` as the last export that passed left it, so a resume
        whose export fails (a busy GPU, a timeout) keeps a validated
        artifact. The winner's config is also written to
        ``export/config.json``. Returns the summary also written to
        ``export/summary.json``: ``status`` (``exported``,
        ``export_rejected``, ``export_unsupported`` when the workload cannot
        save the winner, or ``export_failed``) and ``reason``; when an
        earlier export stays in ``artifact/``, ``kept`` names the directory,
        the config it holds and its other files.
        """
        root = self.directory / "export"
        artifact = self.directory / "artifact"
        staging = root / "files"
        aside = root / "previous_artifact"
        _restore_aside(artifact, aside)
        _remove(staging)
        write_json(root / "config.json", winner["config"])
        result = self._run_worker(
            {
                "kind": "measure",
                "config": winner["config"],
                "timing": asdict(Timing(warmup=1, rounds=3, calls=1, profile_calls=0)),
                "profile": False,
                "export_dir": str(staging),
            },
            root,
        )
        summary: dict[str, Any] = {"trial": winner["id"], "config": winner["config"]}
        if result.get("status") != "ok":
            unsupported = result.get("stage") == "export" and bool(
                result.get("export_unsupported")
            )
            summary.update(
                status="export_unsupported" if unsupported else "export_failed",
                reason=failure_reason(result),
            )
        else:
            missed = _accuracy_misses(result, bounds)
            summary.update(
                accuracy=result["accuracy"],
                latency_ms=result["latency_ms"]["median"],
            )
            if missed:
                summary.update(
                    status="export_rejected",
                    reason=f"the rebuilt candidate missed the accuracy bounds "
                    f"({'; '.join(missed)}); its files are in {staging}",
                )
            else:
                write_json(staging / "config.json", winner["config"])
                _replace_directory(staging, artifact, aside)
                summary.update(
                    status="exported",
                    reason="the rebuilt candidate met the accuracy bounds",
                    files=list(result.get("exported") or ()),
                )
        if summary["status"] != "exported" and artifact.is_dir():
            summary["kept"] = _kept_artifact(artifact)
        write_json(root / "summary.json", summary)
        self._log_event("export", status=summary["status"], reason=summary["reason"])
        if summary["status"] != "exported":
            logger.warning(
                "The config of trial %s was not exported (%s: %s); the config is "
                "in %s%s",
                winner["id"],
                summary["status"],
                summary["reason"],
                root / "config.json",
                (
                    f", and {artifact} keeps an earlier export"
                    if "kept" in summary
                    else ""
                ),
            )
        return summary

    # ------------------------------------------------- workers and records

    def _measure_config(
        self, config: Config, timing: Timing, directory: Path, *, profile: bool = False
    ) -> dict[str, Any]:
        """One measurement of ``config`` in a fresh worker process."""
        return self._run_worker(
            {
                "kind": "measure",
                "config": config,
                "timing": asdict(timing),
                "profile": profile,
                "noise_control": self._noise_control_request(),
            },
            directory,
        )

    def _noise_control_request(self) -> dict[str, Any]:
        """The noise-control settings a worker needs to time a candidate."""
        return {
            "quiet_gpu_s": self.noise_control.quiet_gpu_s,
            "retries": self.noise_control.retries,
            "max_spread": self.noise_control.max_spread,
        }

    def _run_worker(self, request: dict[str, Any], directory: Path) -> dict[str, Any]:
        """Run one job in a fresh worker process; return its result.

        The job's ``request.json``, ``worker.log`` and ``result.json`` go in
        ``directory``. A measurement whose config uses prepared knob values
        is pinned: its worker may only load cached kernels
        (``"allow_generation": false``), so that a kernel missing from the
        cache fails the build at once instead of starting a search, and a
        result whose build ran, for a prepared generation task, another kernel
        than the task's pinned one fails at stage ``kernels``. So does a
        build that failed because it needed kernels for tasks no preparation
        made (``unprepared``; :meth:`_decide` then records an invalid
        proposal), and a config that uses a value whose preparation failed,
        without a worker.
        """
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / "result.json"
        result_path.unlink(missing_ok=True)
        pinned = (
            self._prepared_values(request.get("config") or {})
            if request["kind"] == "measure"
            else []
        )
        request = {
            **request,
            "workload_factory": self.workload_factory.to_json(),
            "reference_path": str(self.directory / "reference.pt"),
            "result_path": str(result_path),
            "session_dir": str(self.directory),
        }
        if pinned:
            request["allow_generation"] = False
        write_json(directory / "request.json", request)
        unusable = [entry for entry in pinned if entry["status"] != "ok"]
        if unusable:
            result: dict[str, Any] = {
                "status": "failed",
                "stage": "kernels",
                "error": "The config uses "
                + "; ".join(_unusable_reason(entry) for entry in unusable),
            }
            write_json(result_path, result)
            return result
        result = self._run_worker_process(request, directory)
        if not pinned:
            return result
        unprepared = [
            kernel
            for kernel in result.get("missing_kernels") or ()
            if isinstance(kernel, Mapping)
            and kernel.get("task_sha256") not in self._pins
        ]
        problems = _pinned_kernel_problems(self._pins, result)
        if result.get("status") != "ok" and unprepared:
            result = {
                **result,
                "stage": "kernels",
                "error": _unprepared_reason(unprepared, pinned),
                "build_error": result.get("error"),
                "unprepared": unprepared,
            }
        elif problems:
            result = {
                **result,
                "status": "failed",
                "stage": "kernels",
                "error": "The run did not use the prepared kernels: "
                + "; ".join(problems),
            }
        else:
            return result
        write_json(result_path, result)
        return result

    def _run_worker_process(
        self, request: dict[str, Any], directory: Path
    ) -> dict[str, Any]:
        """Run the worker process for ``directory/request.json``; return its result.

        The worker runs in its own process group, with ``-X faulthandler`` so
        that a crash, even while importing, leaves its Python stacks in
        ``worker.log``. After its time limit, ``prepare_timeout_s`` for a
        ``prepare`` job and ``trial_timeout_s`` for any other, it gets
        SIGUSR1, on which it logs its stacks, and then its process group gets
        SIGKILL.

        A worker that leaves no result, because it crashed or was killed,
        gets one written for it (``synthesized``): status ``failed``, stage
        ``crash`` or ``timeout``, its ``exit_status`` and the ``signal`` that
        ended it, and from its log the first line reporting an error
        (``log_error``), the Python stack dump it ends with (``log_stack``,
        starting with the thread that faulted or, for a timeout, the main
        thread) and its end (``log_tail``). A worker that wrote its result
        and then exited abnormally keeps its result, with ``exit_status``
        and ``signal``, and an ``exit_note`` if it ran past its time limit.
        """
        result_path = Path(request["result_path"])
        log_path = directory / "worker.log"
        command = [
            self.python,
            "-X",
            "faulthandler",
            "-P",
            "-m",
            "torch_tensorrt.autotune._worker",
            str(directory / "request.json"),
        ]
        environment = {**os.environ, **self.env}
        limit = (
            "prepare_timeout_s" if request["kind"] == "prepare" else "trial_timeout_s"
        )
        timeout = getattr(self.budget, limit)
        timed_out = killed = False
        with open(log_path, "wb") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=self.cwd,
                env=environment,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                returncode, killed = _kill_worker(process)
        ended = _exit_fields(returncode)
        if result_path.is_file():
            result: dict[str, Any] = json.loads(result_path.read_text())
            if timed_out or returncode:
                result.update(ended)
                if timed_out:
                    result["exit_note"] = (
                        f"killed after {limit}={timeout:g}, having written its "
                        "result"
                        if killed
                        else f"exceeded {limit}={timeout:g}, then exited before "
                        "it was killed"
                    )
                write_json(result_path, result)
            return result
        excerpt = _read_worker_log(log_path)
        if timed_out:
            stage, error = "timeout", f"The worker exceeded {limit}={timeout:g}"
        elif ended["signal"]:
            stage = "crash"
            error = (
                f"The worker was killed by {ended['signal']} (exit status "
                f"{returncode}) before writing a result"
            )
            if ended["signal"] == "SIGKILL":
                error += "; the session did not send it (an out-of-memory kill?)"
        else:
            stage = "crash"
            error = (
                f"The worker exited with status {returncode} without writing a result"
            )
        result = {
            "status": "failed",
            "stage": stage,
            "error": error,
            **ended,
            **excerpt,
            "synthesized": True,
        }
        write_json(result_path, result)
        return result

    def _log_event(self, event: str, **fields: Any) -> None:
        """Append one event to ``events.jsonl``, with the time."""
        line = json.dumps({"time": time.time(), "event": event, **fields}, default=str)
        with open(self.directory / "events.jsonl", "a") as stream:
            stream.write(line + "\n")

    def _trial_dir(self, number: int) -> Path:
        return self.directory / "trials" / f"{number:04d}"

    def _load_trials(self) -> list[dict[str, Any]]:
        """The recorded trials, in order, with their results and decisions.

        A measured trial without ``decision.json`` was interrupted; an
        invalid proposal also keeps its decision in ``trial.json``.
        """
        trials = []
        root = self.directory / "trials"
        for directory in sorted(root.glob("[0-9][0-9][0-9][0-9]")):
            record_path = directory / "trial.json"
            if not record_path.is_file():
                continue
            record = json.loads(record_path.read_text())
            result_path = directory / "result.json"
            decision_path = directory / "decision.json"
            if decision_path.is_file() and result_path.is_file():
                record["result"] = json.loads(result_path.read_text())
                record["decision"] = json.loads(decision_path.read_text())
            elif decision_path.is_file():
                record["decision"] = json.loads(decision_path.read_text())
            trials.append(record)
        return trials


def _is_validated_baseline(record: Mapping[str, Any]) -> bool:
    """Whether a recorded trial 0000 was measured and found to be a valid baseline."""
    return (
        record.get("result") is not None
        and (record.get("decision") or {}).get("decision") == "baseline"
    )


def _run_summary(result: Mapping[str, Any], goal: Goal) -> dict[str, Any]:
    """One run in a head-to-head summary: status, metric, latency and accuracy."""
    if result.get("status") != "ok":
        return {"status": result.get("status"), "stage": result.get("stage")}
    return {
        "status": "ok",
        "metric": metric_value(result, goal),
        "latency_ms": result["latency_ms"],
        "accuracy": result["accuracy"],
    }


def _preparation_names(preparations: Sequence[tuple[str, Any]]) -> list[str]:
    """The directory name of each prepared knob value: ``<knob>=<value>``.

    A string value appears as it is, any other value as JSON. Characters
    other than letters, digits and ``._=+-`` become ``_``; a name that needed
    that, is long, or is shared by two values (``"5"`` and ``5``) ends with
    the start of the SHA-256 of the knob and the value.
    """
    plain = [
        f"{name}={value if isinstance(value, str) else json.dumps(value)}"
        for name, value in preparations
    ]
    safe = [_UNSAFE_NAME.sub("_", label) for label in plain]
    names = []
    for (name, value), label, cleaned in zip(preparations, plain, safe):
        if cleaned != label or safe.count(cleaned) > 1 or len(cleaned) > 100:
            cleaned = f"{cleaned[:80]}-{digest([name, value])[:8]}"
        names.append(cleaned)
    return names


def _sets(config: Mapping[str, Any], name: str, value: Any) -> bool:
    """Whether ``config`` sets knob ``name`` to ``value`` (a non-default value)."""
    return name in config and is_member(config[name], (value,))


def _unprepared(
    entry: Mapping[str, Any], status: str, reason: str, config: Config | None = None
) -> dict[str, Any]:
    """The preparation entry of a knob value that has no prepared build."""
    return {
        **entry,
        "status": status,
        "reason": reason,
        "config": config,
        "build_s": None,
        "regions": [],
        "region_speedup": None,
        "warnings": [],
    }


def _preparation_entry(
    entry: Mapping[str, Any],
    config: Config,
    result: Mapping[str, Any],
    baseline_tasks: set[Any],
) -> dict[str, Any]:
    """The preparation entry of a ``prepare`` worker's result.

    ``regions`` are the kernels its build registered for generation tasks
    that the baseline's build did not register: the regions the value adds
    or changes. :func:`_use_final_kernels` later gives them their final
    kernels and sets ``region_speedup``.
    """
    ok = result.get("status") == "ok"
    return {
        **entry,
        "status": "ok" if ok else "failed",
        "reason": None if ok else failure_reason(result),
        "config": config,
        "build_s": result.get("build_s"),
        "regions": [
            dict(kernel)
            for kernel in result.get("kernels") or ()
            if isinstance(kernel, Mapping)
            and kernel.get("task_sha256") not in baseline_tasks
        ],
        "region_speedup": None,
        "warnings": [],
    }


def _finished_at(directory: Path) -> int:
    """When the job in ``directory`` wrote its ``result.json`` (0 if unknown)."""
    try:
        return (directory / "result.json").stat().st_mtime_ns
    except OSError:
        return 0


# The fields of a kernel record that describe the kernel itself, which a
# prepared value's region takes from its task's final kernel.
_KERNEL_FIELDS = (
    "kernel_sha256",
    "backend",
    "provider",
    "model",
    "origin",
    "latency_ms",
    "baseline",
    "baseline_latency_ms",
    "speedup",
    "manifest",
)


def _kernel_pins(
    finished: Sequence[tuple[int, int, Mapping[str, Any], Mapping[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """The kernel each prepared generation task must run, and the tasks that changed.

    ``finished`` holds, for each preparation with a result, when it finished,
    its index, its entry and its result. Taken in the order the preparations
    finished, the last kernel registered for a task is the task's final
    kernel: the one the kernel cache now holds, which every run that uses a
    prepared value must run for that task. A task that got different kernels
    gets a warning naming the values whose builds registered it.
    """
    registered: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for _, _, entry, result in sorted(finished, key=lambda item: item[:2]):
        for kernel in result.get("kernels") or ():
            if isinstance(kernel, Mapping) and kernel.get("task_sha256"):
                registered.setdefault(kernel["task_sha256"], []).append(
                    (dict(kernel), entry["name"])
                )
    pins: dict[str, dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []
    for task, kernels in registered.items():
        final = kernels[-1][0]
        values = list(dict.fromkeys(name for _, name in kernels))
        pins[task] = {
            "task_sha256": task,
            "region": final.get("region"),
            "kernel_sha256": final.get("kernel_sha256"),
            "kernel": final,
            "values": values,
        }
        earlier = [
            sha
            for sha in dict.fromkeys(
                kernel.get("kernel_sha256") for kernel, _ in kernels
            )
            if sha != final.get("kernel_sha256")
        ]
        if not earlier:
            continue
        replaced: list[dict[str, Any]] = [
            {
                "kernel_sha256": sha,
                "values": list(
                    dict.fromkeys(
                        name
                        for kernel, name in kernels
                        if kernel.get("kernel_sha256") == sha
                    )
                ),
            }
            for sha in earlier
        ]
        message = (
            f"The kernel of generation task {_short(task)} (region "
            f"{final.get('region')!r}) changed while knob values were prepared: "
            + "; ".join(
                f"{_short(item['kernel_sha256'])}, registered by "
                f"{', '.join(item['values'])}"
                for item in replaced
            )
            + f", was replaced by {_short(final.get('kernel_sha256'))}, registered "
            f"by {kernels[-1][1]}, which the kernel cache now holds; runs that use "
            f"{', '.join(values)} must run {_short(final.get('kernel_sha256'))}"
        )
        warnings.append(
            {
                "task_sha256": task,
                "region": final.get("region"),
                "kernel_sha256": final.get("kernel_sha256"),
                "replaced": replaced,
                "values": values,
                "message": message,
            }
        )
    return pins, warnings


def _use_final_kernels(
    entry: dict[str, Any],
    pins: Mapping[str, Mapping[str, Any]],
    warnings: Sequence[Mapping[str, Any]],
) -> None:
    """Give ``entry``'s regions their tasks' final kernels, with its warnings.

    A region whose kernel a later preparation replaced takes the final
    kernel's fields, and ``replaced_kernel_sha256`` keeps the kernel the
    value's own build registered. ``region_speedup`` is then the smallest
    measured speedup of its regions (None when none was measured).
    """
    regions = []
    for kernel in entry["regions"]:
        pin = pins.get(kernel.get("task_sha256"))
        if pin is not None and pin["kernel_sha256"] != kernel.get("kernel_sha256"):
            kernel = {
                **kernel,
                **{key: pin["kernel"].get(key) for key in _KERNEL_FIELDS},
                "replaced_kernel_sha256": kernel.get("kernel_sha256"),
            }
        regions.append(kernel)
    speedups = [
        kernel["speedup"]
        for kernel in regions
        if isinstance(kernel.get("speedup"), (int, float))
        and not isinstance(kernel["speedup"], bool)
    ]
    entry.update(
        regions=regions,
        region_speedup=min(speedups) if speedups else None,
        warnings=[
            warning["message"]
            for warning in warnings
            if entry["name"] in warning["values"]
        ],
    )


def _unusable_reason(entry: Mapping[str, Any]) -> str:
    """Why a run cannot use a prepared knob value, and where to look."""
    what = (
        "could not be prepared" if entry["status"] == "failed" else "was not prepared"
    )
    return (
        f"{entry['knob']}={entry['value']!r}, which {what} ({entry['reason']}; "
        f"see kernels/{entry['name']})"
    )


def _short(sha: Any) -> str:
    """The start of a digest, for messages."""
    return str(sha)[:_SHORT_DIGEST] if sha else "?"


def _pinned_kernel_problems(
    pins: Mapping[str, Mapping[str, Any]], result: Mapping[str, Any]
) -> list[str]:
    """How a run's region kernels differ from the kernels pinned to their tasks.

    Only kernels whose generation task a preparation registered are
    checked, each against its task's final kernel; a prepared task the run
    did not register is fine, since the config may not compile that region.
    A failed run is not checked: its own failure says what went wrong.
    """
    if result.get("status") != "ok":
        return []
    problems = []
    for kernel in result.get("kernels") or ():
        if not isinstance(kernel, Mapping):
            continue
        pin = pins.get(str(kernel.get("task_sha256")))
        if pin is not None and kernel.get("kernel_sha256") != pin["kernel_sha256"]:
            problems.append(
                f"region {kernel.get('region_id')!r} ran kernel "
                f"{_short(kernel.get('kernel_sha256'))} ({kernel.get('origin')}) "
                f"for generation task {_short(kernel.get('task_sha256'))}, not the "
                f"kernel {_short(pin['kernel_sha256'])} that "
                f"{', '.join(pin['values'])} prepared for it"
            )
    return problems


def _unprepared_reason(
    missing: Sequence[Mapping[str, Any]], pinned: Sequence[Mapping[str, Any]]
) -> str:
    """Why a run that uses prepared values needs kernels that were never prepared."""
    regions = "; ".join(
        f"region {kernel.get('region_id')!r} (generation task "
        f"{_short(kernel.get('task_sha256'))})"
        for kernel in missing
    )
    prepared = "; ".join(
        f"{entry['name']} in {json.dumps(entry['config'], sort_keys=True)}"
        for entry in pinned
    )
    return (
        f"The config needs region kernels that were never prepared: {regions}. Its "
        f"settings differ from those its prepared values were prepared in "
        f"({prepared}), and runs that use prepared values only load kernels from "
        "the cache"
    )


def _planner_kernels(
    entry: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """A prepared knob value as the planner context shows it.

    Its knob, value, status and reason, the config it was prepared in
    (``prepared_config``), its build time, its smallest region speedup,
    whether the best config ``config`` uses it, its warnings, and each
    region kernel it adds or changes, with its task's final kernel, without
    the local path of its ``accepted.json``.
    """
    return {
        "knob": entry["knob"],
        "value": entry["value"],
        "status": entry["status"],
        "reason": entry["reason"],
        "prepared_config": entry["config"],
        "build_s": entry["build_s"],
        "region_speedup": entry["region_speedup"],
        "used_by_best": _sets(config, entry["knob"], entry["value"]),
        "warnings": list(entry.get("warnings") or ()),
        "regions": [
            {key: value for key, value in kernel.items() if key != "manifest"}
            for kernel in entry["regions"]
        ],
    }


def _kill_worker(process: subprocess.Popen[bytes]) -> tuple[int, bool]:
    """Kill a worker's process group, after asking the worker to log its stacks.

    Returns the worker's exit status and whether it was still running when
    the group was killed.
    """
    if hasattr(signal, "SIGUSR1"):
        with contextlib.suppress(OSError):
            os.kill(process.pid, signal.SIGUSR1)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=_DUMP_GRACE_S)
    running = process.poll() is None
    with contextlib.suppress(OSError):
        os.killpg(process.pid, signal.SIGKILL)
    return process.wait(), running


def _exit_fields(returncode: int) -> dict[str, Any]:
    """A worker's exit status and the name of the signal that ended it, if one did."""
    name = None
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"signal {-returncode}"
    return {"exit_status": returncode, "signal": name}


def _read_worker_log(path: Path) -> dict[str, str | None]:
    """What a worker's log says about how it ended.

    ``log_error`` is the first line reporting an error (warnings skipped),
    ``log_stack`` the Python stack dump the log ends with (faulthandler's,
    the thread that matters first; see :func:`_stack_excerpt`), and
    ``log_tail`` the end of the log, without the list of extension modules a
    fatal error dump ends with. Only the last ``_LOG_SCAN_BYTES`` are read.
    """
    try:
        with open(path, "rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size - _LOG_SCAN_BYTES))
            text = stream.read().decode(errors="replace")
    except OSError:
        return {"log_error": None, "log_stack": None, "log_tail": ""}
    lines = text.splitlines()
    if size > _LOG_SCAN_BYTES:
        lines = lines[1:]  # the first line is cut
    lines = [line for line in lines if not line.startswith("Extension modules: ")]
    first = next(
        (
            line.strip()
            for line in lines
            if _ERROR_LINE.search(line) and not _WARNING_LINE.search(line)
        ),
        None,
    )
    if first is not None and len(first) > 500:
        first = first[:497] + "..."
    start = _stack_start(lines)
    return {
        "log_error": first,
        "log_stack": (
            None
            if start is None
            else _stack_excerpt(_dump_lines(lines[start:]))[:_LOG_STACK_CHARS]
        ),
        "log_tail": "\n".join(lines)[-_LOG_TAIL_CHARS:],
    }


def _stack_start(lines: list[str]) -> int | None:
    """Index of the first line of the Python stack dump in ``lines``.

    A dump is faulthandler's: a ``Fatal Python error`` line (and Python's
    ``Python runtime state`` line) or a thread header ending in ``(most
    recent call first):``, indented frames, blank lines, and the ``...``
    that ends a dump of more than 100 threads. The last fatal error's dump
    counts wherever it is: after writing it, faulthandler calls the handler
    it replaced, and a native library's crash handler can print more. A dump
    without a fatal error line (``SIGUSR1``, before a timeout kill) counts
    only when the log ends with it.
    """
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].startswith("Fatal Python error"):
            return index
    start = None
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        if line.startswith("Fatal Python error"):
            return index
        if line.endswith("(most recent call first):"):
            start = index
        elif (
            line.strip()
            and not line[0].isspace()
            and line != "..."
            and not line.startswith("Python runtime state:")
        ):
            break
    return start


def _dump_lines(lines: list[str]) -> list[str]:
    """The lines of the faulthandler dump that ``lines`` start with, up to its end."""
    end = len(lines)
    for index, line in enumerate(lines[1:], start=1):
        if (
            line.strip()
            and not line[0].isspace()
            and line != "..."
            and not line.endswith("(most recent call first):")
            and not line.startswith("Python runtime state:")
        ):
            end = index
            break
    return lines[:end]


def _stack_excerpt(dump: list[str]) -> str:
    """A faulthandler dump with the thread that explains the failure first.

    faulthandler lists the threads newest first, so the main thread last,
    and labels the thread the signal reached ``Current thread``. After a
    fatal error that is the thread that faulted; the dump a worker writes on
    the session's ``SIGUSR1`` (before a timeout kill) is about the main
    thread, where the job runs, whichever thread received the signal. The
    excerpt is the dump's ``Fatal Python error`` line, that thread's
    frames (innermost first), then the other threads in faulthandler's
    order. A fault outside any Python thread falls back to the main thread.
    """
    fatal = dump[0].startswith("Fatal Python error")
    preamble: list[str] = []
    blocks: list[list[str]] = []
    truncated = False
    for line in dump:
        if line.endswith("(most recent call first):") and not line[0].isspace():
            blocks.append([line])
        elif line == "...":
            truncated = True  # more than 100 threads: the main thread is cut
        elif blocks and line.strip():
            blocks[-1].append(line)
        elif line.strip():
            preamble.append(line)
    if not blocks:
        return "\n".join(preamble)
    python = [
        index
        for index, block in enumerate(blocks)
        if block[0].startswith(("Thread 0x", "Current thread 0x", "Stack ("))
    ]
    current = next(
        (index for index in python if blocks[index][0].startswith("Current thread")),
        None,
    )
    main = python[-1] if python and not truncated else None
    first = current if fatal else main
    if first is None:
        first = main if fatal else current
    if first is None:
        first = 0
    order = [first] + [index for index in range(len(blocks)) if index != first]
    parts = (["\n".join(preamble)] if preamble else []) + [
        "\n".join(blocks[index]) for index in order
    ]
    return "\n\n".join(parts)


def _error_excerpt(result: Mapping[str, Any], size: int = 500) -> str:
    """At most ``size`` characters of a failure's log or error for the planner.

    The start of the worker's Python stack dump when its log ends with one:
    the fatal error and the innermost frames of the thread that faulted, or
    of the main thread for a timeout (see :func:`_stack_excerpt`); else the
    end of its log or error.
    """
    if result.get("log_stack"):
        return str(result["log_stack"]).strip()[:size]
    return str(result.get("log_tail") or result.get("error") or "").strip()[-size:]


def _malformed(proposal: Proposal) -> str | None:
    """Why ``proposal``'s fields cannot be recorded and checked, or None.

    Its changes must map knob names to values, its edits map editable file
    names to lists of edit mappings, its rationale, planner and note be
    text, and everything be JSON data.
    """
    if not isinstance(proposal.changes, Mapping) or not all(
        isinstance(name, str) for name in proposal.changes
    ):
        return "changes must map knob names to values"
    if not all(
        isinstance(text, str)
        for text in (proposal.rationale, proposal.planner, proposal.note)
    ):
        return "rationale, planner and note must be text"
    edits = proposal.edits
    if not isinstance(edits, Mapping) or not all(
        isinstance(name, str)
        and isinstance(file_edits, Sequence)
        and not isinstance(file_edits, (str, bytes))
        and all(isinstance(edit, Mapping) for edit in file_edits)
        for name, file_edits in edits.items()
    ):
        return "edits must map editable file names to lists of {search, replace} edits"
    try:
        json.dumps(
            [
                dict(proposal.changes),
                {
                    name: [dict(edit) for edit in file_edits]
                    for name, file_edits in edits.items()
                },
                proposal.rationale,
                proposal.stop,
                proposal.planner,
                proposal.note,
            ]
        )
    except (TypeError, ValueError) as error:
        return f"it is not JSON data ({error})"
    return None


def _error_text(error: BaseException) -> str:
    """``Type: message`` of an exception, cut to 1000 characters."""
    text = f"{type(error).__name__}: {error}"
    return text if len(text) <= 1000 else text[:997] + "..."


def _remove(path: Path) -> None:
    """Remove a file, a symbolic link or a directory tree, if there is one."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _replace_directory(new: Path, target: Path, aside: Path) -> None:
    """Make ``new`` the ``target`` directory, deleting the old one only afterwards.

    The old ``target`` is renamed to ``aside`` first. If the process stops
    before ``new`` takes its place, :func:`_restore_aside` puts it back.
    """
    _remove(aside)
    if target.exists() or target.is_symlink():
        os.replace(target, aside)
    os.replace(new, target)
    _remove(aside)


def _restore_aside(target: Path, aside: Path) -> None:
    """Undo a :func:`_replace_directory` that stopped between its two renames."""
    if not (aside.exists() or aside.is_symlink()):
        return
    if target.exists() or target.is_symlink():
        _remove(aside)
    else:
        os.replace(aside, target)


def _kept_artifact(artifact: Path) -> dict[str, Any]:
    """What an earlier export left in ``artifact/``: its config and its other files."""
    try:
        kept: Any = json.loads((artifact / "config.json").read_text())
    except (OSError, ValueError):
        kept = None
    return {
        "artifact": str(artifact),
        "config": kept,
        "files": sorted(
            entry.name for entry in artifact.iterdir() if entry.name != "config.json"
        ),
    }


def _differences(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """What differs between a pinned workload or environment and the current one."""
    found: list[str] = []
    for name in sorted(set(recorded) | set(current)):
        before, after = recorded.get(name), current.get(name)
        if before == after:
            continue
        if name in ("workload_files", "knobs", "editable_files"):
            old, new = _by_name(before), _by_name(after)
            label = {"knobs": "knob ", "editable_files": "editable file "}.get(name, "")
            for item in sorted(set(old) | set(new)):
                if item not in new:
                    found.append(f"{label}{item} was removed")
                elif item not in old:
                    found.append(f"{label}{item} was added")
                elif old[item] != new[item]:
                    found.append(f"{label}{item} changed")
        elif isinstance(before, Mapping) and isinstance(after, Mapping):
            found.extend(
                f"{name}.{field} was {before.get(field)!r}, now {after.get(field)!r}"
                for field in sorted(set(before) | set(after))
                if before.get(field) != after.get(field)
            )
        elif before is None or after is None:
            found.append(
                f"{name} was {_abbreviated(before)}, now {_abbreviated(after)}"
            )
        elif isinstance(before, (Mapping, list)) or isinstance(after, (Mapping, list)):
            found.append(f"{name} changed")
        else:
            found.append(f"{name} was {before!r}, now {after!r}")
    return found


def _abbreviated(value: Any, size: int = 200) -> str:
    """``repr(value)``, cut to ``size`` characters."""
    text = repr(value)
    return text if len(text) <= size else text[: size - 3] + "..."


def _by_name(value: Any) -> dict[str, Any]:
    """Index a list of named records (knobs, editable files) or keep a mapping."""
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list):
        return {
            str(item.get("name")): item for item in value if isinstance(item, Mapping)
        }
    return {}
