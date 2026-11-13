import json
import os
import re
import time

import pytest
from helpers import factory_import_core, fake_core
from torch_tensorrt_mcp.jobs import JobManager, _output_dir_name
from torch_tensorrt_mcp.policy import PolicyError
from torch_tensorrt_mcp.storage import write_json


def test_output_directory_uses_utc_time():
    job_id = "a" * 32
    request = {"model": "model"}
    assert _output_dir_name("compile", request, job_id, "2026-11-04T04:05:40Z") == (
        "20261104_040540_compile_model_aaaaaaaa"
    )


def test_non_linux_execution_is_explicitly_unsupported(policy, monkeypatch):
    monkeypatch.setattr("torch_tensorrt_mcp.jobs.sys.platform", "darwin")
    with pytest.raises(RuntimeError, match="requires Linux"):
        JobManager(policy, executor=fake_core)


def wait_terminal(manager, job_id, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = manager.status(job_id)
        if status["status"] in {"succeeded", "failed", "cancelled"}:
            return status
        time.sleep(0.02)
    pytest.fail("worker did not reach a terminal state")


def request(mode="success", timeout=5.0):
    return {
        "model": "model",
        "inputs": "inputs",
        "model_parameters": {"mode": mode},
        "timeout_seconds": timeout,
    }


def test_success_manifest_report_integrity_and_root_lock(policy):
    manager = JobManager(policy, executor=fake_core)
    try:
        with pytest.raises(RuntimeError, match="another adapter"):
            JobManager(policy, executor=fake_core)
        job_id = manager.submit("compile", request())["job_id"]
        assert wait_terminal(manager, job_id)["status"] == "succeeded"
        manifest = manager.get_manifest(job_id)
        compiled = next(
            entry
            for entry in manifest["entries"]
            if entry["artifact_id"] == "compiled_model"
        )
        assert compiled["path"] == "artifacts/compiled.pt2"
        assert manager.get_report(job_id, "compilation")["status"] == "succeeded"
        output_dir = manager.status(job_id)["output_dir"]
        assert re.fullmatch(rf"\d{{8}}_\d{{6}}_compile_model_{job_id[:8]}", output_dir)
        events = [
            json.loads(line)
            for line in (policy.job_path(job_id) / "events.jsonl")
            .read_text()
            .splitlines()
        ]
        assert [event["sequence"] for event in events] == list(
            range(1, len(events) + 1)
        )
        (policy.job_path(job_id) / "compilation.json").write_text("{}")
        with pytest.raises(PolicyError, match="manifest"):
            manager.get_report(job_id, "compilation")
    finally:
        manager.close()


def test_each_worker_imports_generated_candidate_in_a_fresh_process(policy):
    graph_transformation_candidate_id = "c" * 32
    generated = policy.output_root / ".graph_transformation_candidates"
    (
        generated
        / f"graph_transformation_candidate_{graph_transformation_candidate_id}.py"
    ).write_text(
        "import os\n"
        "import uuid\n"
        "IMPORT_COUNT = globals().get('IMPORT_COUNT', 0) + 1\n"
        "WORKER_PID = os.getpid()\n"
        "IMPORT_TOKEN = uuid.uuid4().hex\n"
    )
    (policy.workspace_root / "factories.py").write_text(
        "def model(**kwargs): return None\ndef inputs(**kwargs): return ()\n"
    )
    manager = JobManager(policy, executor=factory_import_core)
    try:
        reports = []
        for _ in range(2):
            candidate_request = request()
            candidate_request["graph_transformation_candidate_id"] = (
                graph_transformation_candidate_id
            )
            job_id = manager.submit("compile", candidate_request)["job_id"]
            assert wait_terminal(manager, job_id)["status"] == "succeeded"
            reports.append(manager.get_report(job_id, "compilation"))

        assert [report["candidate_import_count"] for report in reports] == [1, 1]
        assert all(
            report["candidate_worker_pid"] == report["worker_pid"] for report in reports
        )
        assert (
            reports[0]["candidate_import_token"] != reports[1]["candidate_import_token"]
        )
    finally:
        manager.close()


@pytest.mark.parametrize(
    ("mode", "timeout", "category"),
    [
        ("crash", 5.0, "worker_crash"),
        ("error", 5.0, "invalid_input"),
        ("block", 0.2, "timeout"),
    ],
)
def test_parent_finalizes_worker_failures(policy, mode, timeout, category):
    manager = JobManager(policy, executor=fake_core)
    try:
        job_id = manager.submit("compile", request(mode, timeout))["job_id"]
        status = wait_terminal(manager, job_id)
        assert status["status"] == "failed"
        assert status["failure"]["category"] == category
        assert manager.get_report(job_id, "failure")["failure"]["category"] == category
        assert manager.get_manifest(job_id)["status"] == "failed"
        assert (policy.job_path(job_id) / "diagnostics").is_dir()
    finally:
        manager.close()


def test_worker_spawn_failure_reports_concrete_exception(policy, monkeypatch):
    manager = JobManager(policy, executor=fake_core, autostart=False)

    class FailingProcess:
        def start(self):
            raise OSError("spawn denied")

    try:
        monkeypatch.setattr(
            manager._context, "Process", lambda *args, **kwargs: FailingProcess()
        )
        job_id = manager.submit("compile", request())["job_id"]
        manager._start_next()
        failure = manager.status(job_id)["failure"]
        assert failure["exception_type"] == "OSError"
        assert failure["message"] == (
            "Could not spawn execution worker: OSError: spawn denied"
        )
    finally:
        manager.close()


def test_started_worker_is_owned_before_live_record_persistence(policy, monkeypatch):
    manager = JobManager(policy, executor=fake_core, autostart=False)
    job_id = manager.submit("compile", request("block"))["job_id"]
    original_persist = manager._persist

    def fail_persist(record):
        if record["pid"] is not None:
            raise OSError("live-state storage failed")
        original_persist(record)

    monkeypatch.setattr(manager, "_persist", fail_persist)
    try:
        with pytest.raises(OSError, match="live-state"):
            manager._start_next()
        assert manager._active is not None
        assert manager._active["job_id"] == job_id
        process = manager._active["process"]
        deadline = time.monotonic() + 3
        while process.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not process.is_alive()
    finally:
        monkeypatch.setattr(manager, "_persist", original_persist)
        manager.close()


def test_pre_spawn_event_failure_finalizes_dequeued_job(policy, monkeypatch):
    manager = JobManager(policy, executor=fake_core, autostart=False)
    job_id = manager.submit("compile", request())["job_id"]
    original_event = manager._event

    def fail_start_event(record, event):
        if event.get("message") == "Execution worker starting":
            raise OSError("event storage failed")
        original_event(record, event)

    monkeypatch.setattr(manager, "_event", fail_start_event)
    manager._thread.start()
    try:
        status = wait_terminal(manager, job_id)
        assert status["status"] == "failed"
        assert status["failure"]["category"] == "artifact_error"
        assert manager._active is None
        assert manager.get_manifest(job_id)["status"] == "failed"
    finally:
        manager.close()


def test_published_manifest_remains_authoritative_when_live_persist_fails(
    policy, monkeypatch
):
    manager = JobManager(policy, executor=fake_core, autostart=False)
    original_persist = manager._persist

    def fail_after_publish(record):
        final_path = policy.output_root / record["output_dir"]
        if (
            record["status"] in {"succeeded", "failed", "cancelled"}
            and final_path.is_dir()
        ):
            raise OSError("live-state storage failed after publish")
        original_persist(record)

    monkeypatch.setattr(manager, "_persist", fail_after_publish)
    manager._thread.start()
    try:
        job_id = manager.submit("compile", request())["job_id"]
        status = wait_terminal(manager, job_id)
        assert status["status"] == "succeeded"
        assert status["live_state_persistence_failed"] is True
        assert "retained_staging_dir" not in status
        assert manager.get_manifest(job_id)["status"] == "succeeded"
        assert manager.get_report(job_id, "compilation")["status"] == "succeeded"
    finally:
        monkeypatch.setattr(manager, "_persist", original_persist)
        manager.close()


def test_cancel_reaps_completed_worker_without_overwriting_success(policy):
    manager = JobManager(policy, executor=fake_core, autostart=False)
    try:
        job_id = manager.submit("compile", request())["job_id"]
        manager._start_next()
        assert manager._active is not None
        process = manager._active["process"]
        deadline = time.monotonic() + 5
        while process.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not process.is_alive()
        assert manager.cancel(job_id)["status"] == "succeeded"
    finally:
        manager.close()


def test_queued_and_running_cancellation(policy):
    manager = JobManager(policy, executor=fake_core)
    try:
        running = manager.submit("compile", request("block"))["job_id"]
        deadline = time.monotonic() + 3
        while (
            manager.status(running)["status"] == "queued"
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        queued = manager.submit("analyze", request())["job_id"]
        queued_status = manager.cancel(queued)
        assert queued_status["status"] == "cancelled"
        assert queued_status["failure"]["retry_unchanged"]
        assert not queued_status["failure"]["retry_after_change"]
        cancelled = manager.cancel(running)
        assert cancelled["status"] == "cancellation_requested"
        terminal = wait_terminal(manager, running)
        assert terminal["status"] == "cancelled"
        assert terminal["failure"]["retry_unchanged"]
        assert not terminal["failure"]["retry_after_change"]
        assert manager.cancel(running)["status"] == "cancelled"
    finally:
        manager.close()


def test_restart_finalizes_queued_job_and_does_not_kill_unrelated_pid(policy):
    manager = JobManager(policy, executor=fake_core, autostart=False)
    job_id = manager.submit("compile", request())["job_id"]
    path = policy.output_root / ".jobs" / (job_id + ".json")
    record = json.loads(path.read_text())
    record["pid"] = os.getpid()
    record["process_token"] = "wrong-token"
    write_json(path, record)
    # Simulate a dead manager's lock release without invoking graceful shutdown.
    manager._lock_file.close()
    restarted = JobManager(policy, executor=fake_core)
    try:
        status = restarted.status(job_id)
        assert status["status"] == "failed"
        assert status["failure"]["category"] == "worker_crash"
        assert restarted.get_manifest(job_id)["status"] == "failed"
    finally:
        restarted.close()


def test_followup_requires_successful_compile_and_bounded_reports(policy):
    manager = JobManager(policy, executor=fake_core)
    try:
        with pytest.raises(PolicyError):
            manager.submit("validate", {"source_job_id": "a" * 32})
        source = manager.submit("compile", request())["job_id"]
        assert wait_terminal(manager, source)["status"] == "succeeded"
        followup = manager.submit("validate", {"source_job_id": source})["job_id"]
        assert wait_terminal(manager, followup)["status"] == "succeeded"
        assert re.fullmatch(
            rf"\d{{8}}_\d{{6}}_validate_model_{followup[:8]}",
            manager.status(followup)["output_dir"],
        )
        assert manager.get_manifest(followup)["lineage"] == {"source_job_id": source}
        assert manager.bounded(
            {"job_id": source, "value": "x" * policy.max_response_bytes}
        )["response_omitted"]
    finally:
        manager.close()


def test_numerical_failure_report_makes_job_failed(policy):
    manager = JobManager(policy, executor=fake_core)
    try:
        job_id = manager.submit("compile", request("report_failed"))["job_id"]
        status = wait_terminal(manager, job_id)
        assert status["status"] == "failed"
        assert status["failure"]["category"] == "validation_error"
        assert manager.get_manifest(job_id)["status"] == "failed"
    finally:
        manager.close()


def test_supervisor_finalization_failure_reaches_terminal_after_child_exit(policy):
    manager = JobManager(policy, executor=fake_core)
    try:
        job_id = manager.submit("compile", request("symlink"))["job_id"]
        status = wait_terminal(manager, job_id)
        assert status["status"] == "failed"
        assert status["failure"]["category"] == "artifact_error"
        assert status["retained_staging_dir"] == "." + job_id + ".tmp"
        assert manager._active is None
        with pytest.raises(PolicyError, match="finalization failed"):
            manager.get_manifest(job_id)
    finally:
        manager.close()
    # Fatal artifact errors still release the output-root ownership lock.
    restarted = JobManager(policy, executor=fake_core)
    restarted.close()
