# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import random
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from torch_tensorrt.dynamo import _compiler, _reporting
from torch_tensorrt.reporting import (
    AnalysisReport,
    CompilationFailed,
    EventKind,
    FailureCategory,
    NodeKind,
    ReportStatus,
    Stage,
    event_context,
)


def test_engine_stage_events_report_stage_specific_elapsed_time():
    collector = _reporting.ReportCollector("job")
    token = _reporting._CONTEXT.set(collector)

    def build(**kwargs):
        _reporting.stage_end(Stage.CONVERSION)
        _reporting.stage_start(Stage.BUILD)
        return object()

    try:
        with (
            event_context("job") as emitter,
            mock.patch.object(
                _reporting.time,
                "perf_counter",
                side_effect=[0.0, 0.0, 1.0, 1.0, 4.0, 4.0],
            ),
        ):
            _reporting.build_engine_with_reporting(build, name="engine")
    finally:
        _reporting._CONTEXT.reset(token)

    ended = {
        event.stage: event for event in emitter.events if event.kind == EventKind.END
    }
    assert ended[Stage.CONVERSION].details["elapsed_seconds"] == 1.0
    assert ended[Stage.BUILD].details["elapsed_seconds"] == 3.0
    assert collector.engines[0].elapsed_seconds == 4.0


def test_analysis_uses_stable_source_ids_and_does_not_label_constants_as_buffers():
    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta[_reporting._SOURCE_ID] = "original_x"
    operation = graph.call_function(torch.neg, (source,))
    operation.meta["from_node"] = (source,)
    graph.output(operation)
    module = torch.fx.GraphModule(torch.nn.Module(), graph)
    support = SimpleNamespace(
        node_support={operation.name: True},
        node_reasons={},
    )
    settings = SimpleNamespace(dryrun=True, require_full_compilation=False)

    report = _reporting.partition_analysis(
        module, module, support, settings, skipped=True
    )
    assert report is not None
    decision = next(node for node in report.nodes if node.node_id == operation.name)
    assert decision.source_node_ids == ["original_x"]

    constant = torch.fx.Graph().create_node("get_attr", "constant")
    assert _reporting._node_kind(constant, set(), set()) == NodeKind.STRUCTURAL


def test_report_compilation_preserves_all_loaded_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    with _reporting._preserve_rng_state():
        random.random()
        np.random.random()
        torch.rand(1)

    assert random.getstate() == python_state
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_state[0]
    assert np.array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_final_compilation_callback_failure_is_retained_as_a_diagnostic():
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    inputs = (torch.ones(1),)
    program = torch.export.export(Identity(), inputs)

    def fake_compile(exported_program, **kwargs):
        collector = _reporting.current_collector()
        assert collector is not None
        collector.analysis = AnalysisReport(job_id=collector.job_id)
        return exported_program.module()

    def fail_on_final_end(event):
        if event.stage == Stage.COMPILATION and event.kind == EventKind.END:
            raise RuntimeError("final callback failure")

    with mock.patch.object(_compiler, "compile", side_effect=fake_compile):
        result = _reporting.compile_with_report(
            program, inputs, event_sink=fail_on_final_end
        )

    assert [diagnostic.code for diagnostic in result.report.diagnostics] == [
        "event_callback_failed"
    ]


def test_keyboard_interrupt_produces_structured_report_api_cancellation():
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    inputs = (torch.ones(1),)
    program = torch.export.export(Identity(), inputs)

    with mock.patch.object(_compiler, "compile", side_effect=KeyboardInterrupt):
        analysis = _reporting.analyze(program, inputs)
    assert analysis.status == ReportStatus.CANCELLED
    assert analysis.failure is not None
    assert analysis.failure.category == FailureCategory.CANCELLED

    with mock.patch.object(_compiler, "compile", side_effect=KeyboardInterrupt):
        with pytest.raises(CompilationFailed) as caught:
            _reporting.compile_with_report(program, inputs)
    assert caught.value.report.status == ReportStatus.CANCELLED
    assert caught.value.report.failure is not None
    assert caught.value.report.failure.category == FailureCategory.CANCELLED


def test_keyboard_interrupt_marks_active_engine_cancelled_but_remains_transparent():
    def interrupt(**kwargs):
        raise KeyboardInterrupt

    # Without a collector this is the ordinary compile path: preserve Ctrl+C.
    with pytest.raises(KeyboardInterrupt):
        _reporting.build_engine_with_reporting(interrupt)

    collector = _reporting.ReportCollector("job")
    token = _reporting._CONTEXT.set(collector)
    try:
        with pytest.raises(KeyboardInterrupt):
            _reporting.build_engine_with_reporting(interrupt, name="engine")
    finally:
        _reporting._CONTEXT.reset(token)

    assert collector.engines[0].status == ReportStatus.CANCELLED
