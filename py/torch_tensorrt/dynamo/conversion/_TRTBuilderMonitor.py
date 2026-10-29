# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import threading
from typing import Any, Dict, Optional, Protocol

import tensorrt as trt
from torch_tensorrt.reporting import EventKind, Stage, current_emitter


class _TerminalProgressMonitor(Protocol):
    def __init__(self, engine_name: str = "") -> None: ...

    def phase_start(
        self, phase_name: str, parent_phase: Optional[str], num_steps: int
    ) -> None: ...

    def phase_finish(self, phase_name: str) -> None: ...

    def step_complete(self, phase_name: str, step: int) -> bool: ...


class _ASCIIMonitor(trt.IProgressMonitor):  # type: ignore
    def __init__(self, engine_name: str = "") -> None:
        trt.IProgressMonitor.__init__(self)
        self._active_phases: Dict[str, Dict[str, Any]] = {}
        self._step_result = True

        self._render = True
        if (ci_env_var := os.environ.get("CI_BUILD")) is not None:
            if ci_env_var == "1":
                self._render = False

    def phase_start(
        self, phase_name: str, parent_phase: Optional[str], num_steps: int
    ) -> None:
        try:
            if parent_phase is not None:
                nbIndents = 1 + self._active_phases[parent_phase]["nbIndents"]
            else:
                nbIndents = 0
            self._active_phases[phase_name] = {
                "title": phase_name,
                "steps": 0,
                "num_steps": num_steps,
                "nbIndents": nbIndents,
            }
            self._redraw()
        except KeyboardInterrupt:
            self._step_result = False

    def phase_finish(self, phase_name: str) -> None:
        try:
            del self._active_phases[phase_name]
            self._redraw(blank_lines=1)  # Clear the removed phase.
        except KeyboardInterrupt:
            self._step_result = False

    def step_complete(self, phase_name: str, step: int) -> bool:
        try:
            self._active_phases[phase_name]["steps"] = step
            self._redraw()
            return self._step_result
        except KeyboardInterrupt:
            return False

    def _redraw(self, *, blank_lines: int = 0) -> None:
        if self._render:

            def clear_line() -> None:
                print("\x1b[2K", end="")

            def move_to_start_of_line() -> None:
                print("\x1b[0G", end="")

            def move_cursor_up(lines: int) -> None:
                print("\x1b[{}A".format(lines), end="")

            def progress_bar(steps: int, num_steps: int) -> str:
                INNER_WIDTH = 10
                completed_bar_chars = int(INNER_WIDTH * steps / float(num_steps))
                return "[{}{}]".format(
                    "=" * completed_bar_chars, "-" * (INNER_WIDTH - completed_bar_chars)
                )

            # Set max_cols to a default of 200 if not run in interactive mode.
            max_cols = os.get_terminal_size().columns if sys.stdout.isatty() else 200

            move_to_start_of_line()
            for phase in self._active_phases.values():
                phase_prefix = "{indent}{bar} {title}".format(
                    indent=" " * phase["nbIndents"],
                    bar=progress_bar(phase["steps"], phase["num_steps"]),
                    title=phase["title"],
                )
                phase_suffix = "{steps}/{num_steps}".format(**phase)
                allowable_prefix_chars = max_cols - len(phase_suffix) - 2
                if allowable_prefix_chars < len(phase_prefix):
                    phase_prefix = phase_prefix[0 : allowable_prefix_chars - 3] + "..."
                clear_line()
                print(phase_prefix, phase_suffix)
            for line in range(blank_lines):
                clear_line()
                print()
            move_cursor_up(len(self._active_phases) + blank_lines)
            sys.stdout.flush()


try:
    from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn

    class _RichMonitor(trt.IProgressMonitor):  # type: ignore
        def __init__(self, engine_name: str = "") -> None:
            trt.IProgressMonitor.__init__(self)
            self._active_phases: Dict[str, TaskID] = {}
            self._step_result = True

            self._progress_monitors = Progress(
                TextColumn("  "),
                TimeElapsedColumn(),
                TextColumn("{task.description}: "),
                BarColumn(),
                TextColumn(" {task.percentage:.0f}% ({task.completed}/{task.total})"),
            )

            self._render = True
            if (ci_env_var := os.environ.get("CI_BUILD")) is not None:
                if ci_env_var == "1":
                    self._render = False

            if self._render:
                self._progress_monitors.start()

        def phase_start(
            self, phase_name: str, parent_phase: Optional[str], num_steps: int
        ) -> None:
            try:
                self._active_phases[phase_name] = self._progress_monitors.add_task(
                    phase_name, total=num_steps
                )
                self._progress_monitors.refresh()
            except KeyboardInterrupt:
                # The phase_start callback cannot directly cancel the build, so request the cancellation from within step_complete.
                self._step_result = False

        def phase_finish(self, phase_name: str) -> None:
            try:
                self._progress_monitors.update(
                    self._active_phases[phase_name], visible=False
                )
                self._progress_monitors.stop_task(self._active_phases[phase_name])
                self._progress_monitors.remove_task(self._active_phases[phase_name])
                self._progress_monitors.refresh()
            except KeyboardInterrupt:
                self._step_result = False

        def step_complete(self, phase_name: str, step: int) -> bool:
            try:
                self._progress_monitors.update(
                    self._active_phases[phase_name], completed=step
                )
                self._progress_monitors.refresh()
                return self._step_result
            except KeyboardInterrupt:
                # There is no need to propagate this exception to TensorRT. We can simply cancel the build.
                return False

        def __del__(self) -> None:
            if self._progress_monitors:
                self._progress_monitors.stop()

    _TerminalMonitorClass: type[_TerminalProgressMonitor] = _RichMonitor
except ImportError:
    _TerminalMonitorClass = _ASCIIMonitor


class _ProgressEvents:
    """Emit bounded, thread-safe progress events from native builder callbacks."""

    _BUCKETS = 100

    def __init__(self, engine_name: str = "") -> None:
        self._event_emitter = current_emitter()
        if not engine_name:
            from torch_tensorrt.dynamo._reporting import current_collector

            collector = current_collector()
            engine_name = collector.active_engine if collector else ""
        self._engine_name = engine_name
        self._phases: Dict[str, tuple[Optional[str], int, int]] = {}
        self._lock = threading.RLock()

    def _emit(
        self,
        phase_name: str,
        parent_phase: Optional[str],
        completed: int,
        total: int,
    ) -> None:
        if self._event_emitter is not None:
            self._event_emitter.emit(
                Stage.BUILD,
                EventKind.PROGRESS,
                self._engine_name,
                {
                    "phase": phase_name,
                    "parent_phase": parent_phase,
                    "completed": completed,
                    "total": total,
                },
            )

    def phase_start(
        self, phase_name: str, parent_phase: Optional[str], num_steps: int
    ) -> None:
        with self._lock:
            self._phases[phase_name] = (parent_phase, num_steps, 0)
            self._emit(phase_name, parent_phase, 0, num_steps)

    def step_complete(self, phase_name: str, step: int) -> None:
        with self._lock:
            phase = self._phases.get(phase_name)
            if phase is None:
                return
            parent_phase, total, last_bucket = phase
            if total <= 0:
                return
            bucket = max(0, min(self._BUCKETS, step * self._BUCKETS // total))
            # phase_finish emits the final state; intermediate events are limited
            # to one per percentage bucket to avoid slowing TensorRT callbacks.
            if bucket <= last_bucket or bucket >= self._BUCKETS:
                return
            self._phases[phase_name] = (parent_phase, total, bucket)
            self._emit(phase_name, parent_phase, step, total)

    def phase_finish(self, phase_name: str) -> None:
        with self._lock:
            parent_phase, total, _ = self._phases.pop(phase_name, (None, 0, 0))
            self._emit(phase_name, parent_phase, total, total)


class ReportingBuilderMonitor(trt.IProgressMonitor):  # type: ignore[misc]
    """Forward terminal progress while emitting report-aware progress events."""

    def __init__(self, engine_name: str = "") -> None:
        super().__init__()
        self._progress_events = _ProgressEvents(engine_name)
        self._terminal_monitor = _TerminalMonitorClass(engine_name)

    def phase_start(
        self, phase_name: str, parent_phase: Optional[str], num_steps: int
    ) -> None:
        self._progress_events.phase_start(phase_name, parent_phase, num_steps)
        self._terminal_monitor.phase_start(phase_name, parent_phase, num_steps)

    def phase_finish(self, phase_name: str) -> None:
        self._progress_events.phase_finish(phase_name)
        self._terminal_monitor.phase_finish(phase_name)

    def step_complete(self, phase_name: str, step: int) -> bool:
        self._progress_events.step_complete(phase_name, step)
        return self._terminal_monitor.step_complete(phase_name, step)


TRTBulderMonitor = ReportingBuilderMonitor


class EventBuilderMonitor(trt.IProgressMonitor):  # type: ignore[misc]
    """Native progress events without terminal rendering for report-aware jobs."""

    def __init__(self) -> None:
        super().__init__()
        self._progress_events = _ProgressEvents()

    def phase_start(
        self, phase_name: str, parent_phase: Optional[str], num_steps: int
    ) -> None:
        self._progress_events.phase_start(phase_name, parent_phase, num_steps)

    def phase_finish(self, phase_name: str) -> None:
        self._progress_events.phase_finish(phase_name)

    def step_complete(self, phase_name: str, step: int) -> bool:
        self._progress_events.step_complete(phase_name, step)
        return True
