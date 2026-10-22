# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from concurrent.futures import ThreadPoolExecutor

from torch_tensorrt.dynamo.conversion._TRTBuilderMonitor import (
    EventBuilderMonitor,
    _ASCIIMonitor,
    _ProgressEvents,
)
from torch_tensorrt.reporting import EventKind, Stage, event_context


def test_ascii_monitor_remembers_keyboard_interrupt():
    monitor = _ASCIIMonitor()

    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monitor._redraw = interrupt
    monitor.phase_start("build", None, 1)
    assert monitor._step_result is False
    monitor._step_result = True
    monitor.phase_finish("build")
    assert monitor._step_result is False


def test_progress_events_are_bounded_and_keep_phase_context():
    with event_context("job") as emitter:
        progress = _ProgressEvents("engine")
        progress.phase_start("child", "parent", 1000)
        for step in range(1, 1001):
            progress.step_complete("child", step)
        progress.phase_finish("child")

    events = emitter.events
    assert len(events) == 101
    assert all(event.stage == Stage.BUILD for event in events)
    assert all(event.kind == EventKind.PROGRESS for event in events)
    assert all(event.stage_id == "engine" for event in events)
    assert all(event.details["parent_phase"] == "parent" for event in events)
    assert events[0].details["completed"] == 0
    assert events[-1].details == {
        "phase": "child",
        "parent_phase": "parent",
        "completed": 1000,
        "total": 1000,
    }


def test_progress_state_is_safe_for_concurrent_native_callbacks():
    with event_context("job") as emitter:
        progress = _ProgressEvents("engine")
        progress.phase_start("build", None, 1000)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(
                pool.map(
                    lambda step: progress.step_complete("build", step), range(1000)
                )
            )
        progress.phase_finish("build")

    completed = [event.details["completed"] for event in emitter.events]
    assert completed[0] == 0
    assert completed[-1] == 1000
    assert completed == sorted(completed)
    assert len(completed) <= 101


def test_event_only_monitor_uses_shared_event_format():
    with event_context("job") as emitter:
        monitor = EventBuilderMonitor()
        monitor.phase_start("build", "parent", 2)
        assert monitor.step_complete("build", 1)
        monitor.phase_finish("build")

    assert [event.details for event in emitter.events] == [
        {
            "phase": "build",
            "parent_phase": "parent",
            "completed": 0,
            "total": 2,
        },
        {
            "phase": "build",
            "parent_phase": "parent",
            "completed": 1,
            "total": 2,
        },
        {
            "phase": "build",
            "parent_phase": "parent",
            "completed": 2,
            "total": 2,
        },
    ]
