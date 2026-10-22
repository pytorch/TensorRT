import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "_contract_test_loader", Path(__file__).with_name("isolation.py")
)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
r = loader.load_reporting()
VERSION_DIRECTORY = f"v{r.SCHEMA_VERSION}"


def test_reporting_public_exports_exclude_implementation_modules():
    assert not {"events", "failures", "models", "serialization"} & set(r.__all__)
    assert all(hasattr(r, name) for name in r.__all__)


def test_reporting_import_has_no_runtime_dependencies():
    import subprocess

    source = f"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location('loader', {str(Path(__file__).with_name("isolation.py"))!r})
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
loader.load_reporting()
assert 'torch' not in sys.modules
assert 'tensorrt' not in sys.modules
assert 'mcp' not in sys.modules
assert 'torch_tensorrt' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", source], check=True)


def test_golden_documents_match_schemas_and_roundtrip():
    jsonschema = pytest.importorskip("jsonschema")
    root = Path(__file__).parent
    golden_root = root / "golden" / VERSION_DIRECTORY
    schema_root = (
        Path(__file__).resolve().parents[3]
        / "schemas"
        / "reporting"
        / VERSION_DIRECTORY
    )
    expected_files = {f"{report_type}.json" for report_type in r.models.REPORT_TYPES}
    assert {path.name for path in golden_root.glob("*.json")} == expected_files
    assert {path.name for path in schema_root.glob("*.json")} == expected_files

    for golden_path in sorted(golden_root.glob("*.json")):
        document = json.loads(golden_path.read_text(encoding="utf-8"))
        contract = r.from_dict(document)
        schema = json.loads(
            (schema_root / golden_path.name).read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        ).validate(document)
        assert r.json_schema(type(contract)) == schema
        assert contract.to_dict() == document
        assert r.from_json(contract.to_json()) == contract
        assert contract.to_json() == r.to_json(document)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_rejected_everywhere(value):
    with pytest.raises(r.ContractError):
        r.AnalysisReport(settings={"bad": [value]}).to_json()
    with pytest.raises(r.ContractError):
        r.from_json(
            '{"report_type":"analysis","schema_version":"0.1","settings":{"bad":NaN}}'
        )


def test_unknown_versions_fields_types_and_duplicate_keys_are_rejected():
    valid = r.AnalysisReport().to_dict()
    for bad in (
        {**valid, "schema_version": "2.0"},
        {**valid, "unknown": 1},
        {**valid, "capture_performed": 1},
        {**valid, "coverage": {"total": True}},
    ):
        with pytest.raises(r.ContractError):
            r.from_dict(bad)
    with pytest.raises(r.ContractError):
        r.from_json(
            '{"report_type":"analysis","schema_version":"0.1","schema_version":"0.1"}'
        )
    with pytest.raises(r.ContractError):
        r.AnalysisReport.from_dict(r.CompilationReport().to_dict())
    with pytest.raises(r.ContractError):
        r.from_dict({"report_type": "analysis"})


def test_canonical_order_and_numerical_roundtrip():
    timestamp = "2026-10-12T00:00:00Z"
    left = r.AnalysisReport(
        created_at=timestamp, settings={"z": 1, "a": {"y": 2, "x": 3}}
    )
    right = r.AnalysisReport(
        created_at=timestamp, settings={"a": {"x": 3, "y": 2}, "z": 1}
    )
    assert left.to_json() == right.to_json()
    measured = r.TimingMeasurements(wall_samples_ms=[1, 2.0])
    assert (
        measured.to_json()
        == r.TimingMeasurements.from_json(measured.to_json()).to_json()
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("throughput_method", "custom"),
        ("gpu_isolation", "exclusive"),
        ("timing_scope", "end_to_end"),
        ("seed_reset", "never"),
        ("model_mode", "train"),
        ("input_reset", "reuse"),
        ("synchronization", "none"),
        ("speedup_method", "mean_ratio"),
        ("inference_mode", 1),
        ("cuda_graphs", 0),
    ],
)
def test_benchmark_protocol_rejects_unknown_methods(field, value):
    with pytest.raises(ValueError):
        r.BenchmarkProtocol(**{field: value})


def test_manifest_paths_and_duplicate_ids_are_checked():
    for path in ("../outside", "/outside", "a/../outside", "a//b", "a\\b", "./a"):
        with pytest.raises(ValueError):
            r.ArtifactEntry("id", path, "0" * 64, 0)
    entry = r.ArtifactEntry("id", "artifacts/model.pt2", "0" * 64, 1)
    with pytest.raises(ValueError):
        r.ArtifactManifest(entries=[entry, entry])
    with pytest.raises(ValueError):
        r.Coverage(total=1, eligible=2)
    with pytest.raises(ValueError):
        r.Event(timestamp="2026-10-12T01:00:00+01:00")


def test_nested_async_jobs_have_separate_sequences_and_callbacks():
    async def run():
        parent_events, left_events, right_events = [], [], []

        async def child(job_id, captured):
            with r.event_context(job_id, captured.append):
                r.emit_event(r.Stage.BUILD, r.EventKind.START)
                await asyncio.sleep(0)
                r.emit_event(r.Stage.BUILD, r.EventKind.END)

        with r.event_context("parent", parent_events.append):
            r.emit_event(r.Stage.ANALYSIS, r.EventKind.START)
            await asyncio.gather(
                child("left", left_events), child("right", right_events)
            )
            r.emit_event(r.Stage.ANALYSIS, r.EventKind.END)
        assert r.current_emitter() is None
        for job_id, events in (
            ("parent", parent_events),
            ("left", left_events),
            ("right", right_events),
        ):
            assert [event.sequence for event in events] == [1, 2]
            assert {event.job_id for event in events} == {job_id}

    asyncio.run(run())


def test_event_callback_failure_policy_and_threads():
    from concurrent.futures import ThreadPoolExecutor

    def fails(event):
        raise OSError("disk failure")

    with r.event_context("job", fails) as emitter:
        emitter.emit("build", "start")
        assert emitter.callback_failures == ["builtins.OSError"]
    fails.critical = True
    with r.event_context("critical", fails):
        with pytest.raises(r.EventPersistenceError):
            r.emit_event("build", "end")
    emitter = r.EventEmitter("threaded")
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: emitter.emit("build", "progress"), range(30)))
    assert [event.sequence for event in emitter.events] == list(range(1, 31))


def test_event_details_are_snapshotted_and_invalid_events_do_not_consume_sequence():
    emitter = r.EventEmitter("immutable")
    details = {"progress": {"completed": 1}}
    event = emitter.emit("build", "progress", details=details)
    details["progress"]["completed"] = 2
    assert event.details == {"progress": {"completed": 1}}

    with pytest.raises(r.ContractError):
        emitter.emit("build", "progress", details={"invalid": object()})
    assert emitter.emit("build", "end").sequence == 2


def test_typed_failure_classification_and_bounded_sanitized_message():
    assert (
        r.classify_exception(MemoryError("capacity"), "build").category
        == r.FailureCategory.OUT_OF_MEMORY
    )
    assert (
        r.classify_exception(TimeoutError("limit"), "build").category
        == r.FailureCategory.TIMEOUT
    )
    assert (
        r.classify_exception(r.InvalidInputError("bad"), "analysis").category
        == r.FailureCategory.INVALID_INPUT
    )
    assert (
        r.classify_exception(ValueError("bug"), "analysis").category
        == r.FailureCategory.INTERNAL_ERROR
    )
    assert (
        r.classify_exception(AssertionError("invariant"), "build").category
        == r.FailureCategory.INTERNAL_ERROR
    )
    unknown = r.classify_exception(
        RuntimeError("/private/model.py token=secret " + "x" * 3000 + "\ntraceback"),
        "build",
    )
    assert unknown.category == r.FailureCategory.INTERNAL_ERROR
    assert len(unknown.message) <= 2048
    assert "/private" not in unknown.message and "token=secret" not in unknown.message
    assert "traceback" not in unknown.message
    report = r.CompilationReport(status=r.ReportStatus.FAILED, failure=unknown)
    assert r.CompilationFailed(report).report is report


def test_wrapped_typed_failure_and_unprintable_exception_are_reportable():
    try:
        try:
            raise MemoryError("capacity")
        except MemoryError as exc:
            raise RuntimeError("converter wrapper") from exc
    except RuntimeError as exc:
        assert (
            r.classify_exception(exc, "build").category
            == r.FailureCategory.OUT_OF_MEMORY
        )
    try:
        try:
            raise MemoryError("capacity")
        except MemoryError as exc:
            raise ValueError("broad wrapper") from exc
    except ValueError as exc:
        assert (
            r.classify_exception(exc, "build").category
            == r.FailureCategory.OUT_OF_MEMORY
        )

    class UnprintableError(RuntimeError):
        def __str__(self):
            raise RuntimeError("formatting failed")

    assert (
        r.classify_exception(UnprintableError(), "build").message == "UnprintableError"
    )


def test_event_retention_is_bounded_without_dropping_sink_events():
    events = []
    emitter = r.EventEmitter("bounded", events.append, max_events=2)
    for _ in range(10):
        emitter.emit("build", "progress")
    assert len(events) == 10
    assert [event.sequence for event in emitter.events] == [9, 10]
    assert emitter.dropped_events == 8

    def callback(event):
        raise OSError("unavailable")

    emitter = r.EventEmitter("failed-callback", callback)
    for _ in range(10):
        emitter.emit("build", "progress")
    assert emitter.callback_failures == ["builtins.OSError"]
    assert emitter.callback_failure_count == 10
