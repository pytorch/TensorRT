"""Native compiler/report integration; skip cleanly when CUDA is unavailable."""

from contextlib import contextmanager
from unittest import mock

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required for native compiler reporting", allow_module_level=True
    )

import torch_tensorrt
from torch_tensorrt.dynamo import _compiler, _reporting, partitioning
from torch_tensorrt.dynamo.conversion._ConverterRegistry import DYNAMO_CONVERTERS
from torch_tensorrt.reporting import (
    AnalysisReport,
    CompilationFailed,
    EventKind,
    FailureCategory,
    NodeAssignment,
    NodeKind,
    ReportStatus,
    Stage,
)


class Supported(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "bias", torch.arange(6, device="cuda", dtype=torch.float16).reshape(2, 3)
        )

    def forward(self, x):
        return torch.relu(x + self.bias)


class Mixed(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + torch.sin(x)


def exported(module=None):
    module = (module if module is not None else Supported()).eval()
    x = torch.randn(2, 3, device="cuda", dtype=torch.float16)
    return module, torch.export.export(module, (x,)), x


def assignments(report):
    return [
        (node.node_id, node.operator, node.kind, node.eligible, node.assignment)
        for node in report.nodes
    ]


def partitions_signature(report):
    return [
        (partition.partition_id, partition.node_ids, partition.kind)
        for partition in report.partitions
    ]


def test_source_node_tags_preserve_existing_provenance():
    graph = torch.fx.symbolic_trace(torch.nn.ReLU())
    node = next(node for node in graph.graph.nodes if node.op == "call_function")
    node.meta["_torchtrt_report_node_id"] = "original_node"
    _reporting.tag_source_nodes(graph)
    assert node.meta["_torchtrt_report_node_id"] == "original_node"


def test_analysis_then_compilation_preserves_source_rng_and_partition_evidence():
    model, program, x = exported()
    original_graph = str(program.graph_module.graph)
    original_state = {name: value.clone() for name, value in program.state_dict.items()}
    original_input = x.clone()
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    events = []
    report = torch_tensorrt.dynamo.analyze(
        program,
        inputs=(x,),
        job_id="analysis",
        event_sink=events.append,
        min_block_size=1,
    )
    assert report.status == ReportStatus.SUCCEEDED
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    compile_events = []
    result = torch_tensorrt.dynamo.compile_with_report(
        program,
        inputs=(x,),
        job_id="compile",
        min_block_size=1,
        optimization_level=0,
        event_sink=compile_events.append,
    )
    assert result.report.status == ReportStatus.SUCCEEDED
    assert any(
        event.stage == Stage.BUILD and event.kind == EventKind.PROGRESS
        for event in compile_events
    )
    assert any(
        event.stage == Stage.CONVERSION and event.kind == EventKind.START
        for event in compile_events
    )
    assert any(
        event.stage == Stage.CONVERSION and event.kind == EventKind.END
        for event in compile_events
    )
    assert result.report.engines[0].metadata["serialized_size_bytes"] > 0
    assert len(result.report.engines[0].metadata["serialized_sha256"]) == 64
    assert assignments(report) == assignments(result.report.analysis)
    assert partitions_signature(report) == partitions_signature(result.report.analysis)
    assert str(program.graph_module.graph) == original_graph
    assert all(
        torch.equal(value, original_state[name])
        for name, value in program.state_dict.items()
    )
    assert torch.equal(x, original_input)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    with torch.inference_mode():
        torch.testing.assert_close(result.module(x), model(x), atol=1e-3, rtol=1e-3)
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert all(event.job_id == "analysis" for event in events)
    assert not any(event.stage == Stage.BUILD for event in events)
    assert AnalysisReport.from_json(report.to_json()) == report


def test_full_compilation_fastpath_emits_one_partition_even_below_minimum():
    _, program, x = exported()
    with mock.patch.object(
        partitioning,
        "fast_partition",
        side_effect=AssertionError("partitioner must be skipped"),
    ):
        report = torch_tensorrt.dynamo.analyze(
            program, (x,), require_full_compilation=True, min_block_size=100
        )
    assert report.status == ReportStatus.SUCCEEDED
    assert report.requested_settings_satisfiable is True
    assert report.coverage.assigned == report.coverage.total
    assert len(report.partitions) == 1
    assert report.partitions[0].partition_id == "_run_on_acc_0"


def test_full_compilation_constructs_partition_inputs_once():
    _, program, x = exported()
    with mock.patch.object(
        partitioning,
        "construct_submodule_inputs",
        wraps=partitioning.construct_submodule_inputs,
    ) as construct_inputs:
        torch_tensorrt.dynamo.compile_with_report(
            program,
            (x,),
            require_full_compilation=True,
            min_block_size=100,
            optimization_level=0,
        )
    assert construct_inputs.call_count == 1


def test_zero_support_explicit_exclusion_and_full_compilation_unsatisfiable():
    _, program, x = exported(Mixed())
    all_ops = {
        node.target
        for node in program.graph_module.graph.nodes
        if node.op == "call_function"
    }
    report = torch_tensorrt.dynamo.analyze(
        program, (x,), torch_executed_ops=all_ops, min_block_size=1
    )
    assert report.status == ReportStatus.SUCCEEDED
    assert report.coverage.total > 0
    assert report.coverage.eligible == report.coverage.assigned == 0
    assert report.coverage.excluded == report.coverage.total
    assert report.partitions == []
    assert all(
        node.assignment == NodeAssignment.PYTORCH
        for node in report.nodes
        if node.kind == NodeKind.COMPUTATIONAL
    )
    strict = torch_tensorrt.dynamo.analyze(
        program,
        (x,),
        torch_executed_ops=all_ops,
        min_block_size=1,
        require_full_compilation=True,
    )
    assert strict.requested_settings_satisfiable is False
    assert any(
        check.name == "requested_settings" and check.status.value == "failed"
        for check in strict.checks
    )
    with pytest.raises(CompilationFailed) as caught:
        torch_tensorrt.dynamo.compile_with_report(
            program,
            (x,),
            torch_executed_ops=all_ops,
            min_block_size=1,
            require_full_compilation=True,
        )
    assert caught.value.report.status == ReportStatus.FAILED
    assert caught.value.report.failure is not None


def test_below_minimum_predicts_actual_fallback():
    _, program, x = exported(Mixed())
    settings = {
        "torch_executed_ops": {"torch.ops.aten.sin.default"},
        "min_block_size": 100,
    }
    analysis = torch_tensorrt.dynamo.analyze(program, (x,), **settings)
    result = torch_tensorrt.dynamo.compile_with_report(program, (x,), **settings)
    assert analysis.status == ReportStatus.SUCCEEDED
    assert analysis.coverage.eligible > 0
    assert analysis.coverage.assigned == 0
    assert analysis.coverage.below_min_block_size == analysis.coverage.eligible
    assert assignments(analysis) == assignments(result.report.analysis)
    assert result.report.engines == []


@pytest.mark.parametrize("resource", [False, True])
def test_global_and_resource_partitions_preserve_node_linkage(resource):
    _, program, x = exported(Mixed())
    report = torch_tensorrt.dynamo.analyze(
        program,
        (x,),
        min_block_size=1,
        use_fast_partitioner=False,
        enable_resource_partitioning=resource,
        cpu_memory_budget=8 << 30,
        torch_executed_ops={"torch.ops.aten.sin.default"},
    )
    assert report.status == ReportStatus.SUCCEEDED, report.failure
    assert report.coverage.assigned > 0
    assert report.coverage.excluded == 1
    node_ids = {node.node_id for node in report.nodes}
    assert all(set(partition.node_ids) <= node_ids for partition in report.partitions)
    assert (
        sum(
            len(
                [
                    node
                    for node in report.nodes
                    if node.node_id in partition.node_ids
                    and node.kind == NodeKind.COMPUTATIONAL
                ]
            )
            for partition in report.partitions
            if partition.kind == "tensorrt"
        )
        == report.coverage.assigned
    )


def test_fast_partitioner_fallback_is_visible_and_restores_converter_settings():
    _, program, x = exported(Mixed())
    previous_settings = DYNAMO_CONVERTERS.compilation_settings
    previous_targets = DYNAMO_CONVERTERS.disallowed_targets
    with mock.patch.object(
        partitioning,
        "fast_partition",
        side_effect=torch.fx.passes.splitter_base.FxNetSplitterInternalError(
            "forced fallback"
        ),
    ):
        report = torch_tensorrt.dynamo.analyze(
            program,
            (x,),
            min_block_size=1,
            torch_executed_ops={"torch.ops.aten.sin.default"},
        )
    assert report.status == ReportStatus.SUCCEEDED
    assert report.coverage.assigned > 0
    assert DYNAMO_CONVERTERS.compilation_settings is previous_settings
    assert DYNAMO_CONVERTERS.disallowed_targets is previous_targets


def test_conversion_failure_retains_analysis_diagnostics_and_typed_events(tmp_path):
    _, program, x = exported()
    events = []
    with mock.patch.object(
        _compiler, "convert_module", side_effect=MemoryError("simulated capacity")
    ):
        with pytest.raises(CompilationFailed) as caught:
            torch_tensorrt.dynamo.compile_with_report(
                program,
                (x,),
                min_block_size=1,
                job_id="failed",
                event_sink=events.append,
                diagnostics_dir=tmp_path,
            )
    report = caught.value.report
    assert report.status == ReportStatus.FAILED
    assert report.failure.category == FailureCategory.OUT_OF_MEMORY
    assert report.analysis is not None
    assert report.engines[-1].status == ReportStatus.FAILED
    assert (tmp_path / "compiler.log").is_file()
    assert (tmp_path / "exported_graph.txt").is_file()
    assert events[-1].kind == EventKind.FAILURE
    assert events[-1].stage == Stage.CONVERSION
    assert AnalysisReport.from_json(report.analysis.to_json()) == report.analysis


def test_optional_event_callback_failure_is_reported_without_failing_analysis():
    _, program, x = exported()

    def callback(event):
        raise RuntimeError("observer failure")

    report = torch_tensorrt.dynamo.analyze(
        program, (x,), min_block_size=1, event_sink=callback
    )
    assert report.status == ReportStatus.SUCCEEDED
    assert any(
        diagnostic.code == "event_callback_failed" for diagnostic in report.diagnostics
    )


def test_unknown_settings_produce_structured_analysis_failure():
    _, program, x = exported()
    report = torch_tensorrt.dynamo.analyze(program, (x,), typo_setting=True)
    assert report.status == ReportStatus.FAILED
    assert report.failure.category == FailureCategory.INVALID_INPUT
    assert report.to_dict()["failure"]["stage"] == "analysis"


def test_diagnostic_destination_failure_is_a_typed_compilation_failure(tmp_path):
    _, program, x = exported()
    with mock.patch.object(
        _reporting.logging, "FileHandler", side_effect=PermissionError("output denied")
    ):
        with pytest.raises(CompilationFailed) as caught:
            torch_tensorrt.dynamo.compile_with_report(
                program, (x,), diagnostics_dir=tmp_path
            )
    assert caught.value.report.status == ReportStatus.FAILED
    assert caught.value.report.failure.category == FailureCategory.ARTIFACT_ERROR


def test_critical_event_sink_failure_does_not_replace_failed_report():
    _, program, x = exported()

    class BrokenSink:
        critical = True

        def __call__(self, event):
            raise OSError("event persistence failed")

    report = torch_tensorrt.dynamo.analyze(program, (x,), event_sink=BrokenSink())
    assert report.status == ReportStatus.FAILED
    assert report.failure is not None
    with pytest.raises(CompilationFailed) as caught:
        torch_tensorrt.dynamo.compile_with_report(
            program, (x,), event_sink=BrokenSink()
        )
    assert caught.value.report.status == ReportStatus.FAILED


def test_diagnostics_close_failure_restores_compiler_context(tmp_path):
    _, program, x = exported()
    previous_settings = DYNAMO_CONVERTERS.compilation_settings
    previous_targets = DYNAMO_CONVERTERS.disallowed_targets

    @contextmanager
    def broken_diagnostics(directory):
        yield
        raise PermissionError("diagnostic flush denied")

    with mock.patch.object(_reporting, "_diagnostics", broken_diagnostics):
        with pytest.raises(CompilationFailed) as caught:
            torch_tensorrt.dynamo.compile_with_report(
                program,
                (x,),
                min_block_size=1,
                optimization_level=0,
                diagnostics_dir=tmp_path,
            )
    assert caught.value.report.failure.category == FailureCategory.ARTIFACT_ERROR
    assert DYNAMO_CONVERTERS.compilation_settings is previous_settings
    assert DYNAMO_CONVERTERS.disallowed_targets is previous_targets
    assert _reporting.current_collector() is None


def test_invalid_nonfinite_settings_still_produce_finite_failure_json():
    _, program, x = exported()
    report = torch_tensorrt.dynamo.analyze(program, (x,), workspace_size=float("nan"))
    assert report.status == ReportStatus.FAILED
    assert report.failure.category == FailureCategory.INVALID_INPUT
    assert AnalysisReport.from_json(report.to_json()) == report
