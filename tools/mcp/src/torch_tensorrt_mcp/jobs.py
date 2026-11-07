"""One CUDA-safe spawned worker at a time, with supervisor-owned finalization."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import signal
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from .models import REQUEST_TYPES
from .policy import Policy, PolicyError, confined_path, file_hash
from .storage import read_json, utc_now, write_json
from .worker import worker_main

TERMINAL = {"succeeded", "failed", "cancelled"}
REPORT_SCHEMA_VERSION = "0.1"
_STAGES = {
    "collect_environment": "environment",
    "analyze": "analysis",
    "compile": "compilation",
    "validate": "validation",
    "benchmark": "benchmark",
}


def _output_dir_name(
    operation: str,
    request: dict[str, Any],
    job_id: str,
    created_at: str,
    source: dict[str, Any] | None = None,
) -> str:
    """Build a sortable, human-readable directory name for a job."""
    model = request.get("model")
    if model is None and source is not None:
        model = source["request"].get("model")
    model = model or "none"
    timestamp = created_at[:19].replace("-", "").replace(":", "").replace("T", "_")
    return f"{timestamp}_{operation}_{model}_{job_id[:8]}"


class _PublishedFinalizationError(RuntimeError):
    """The immutable result exists, but its mutable live record was not updated."""


def process_token(pid: int) -> str | None:
    """Linux start time prevents terminating an unrelated process after PID reuse."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        return text[text.rfind(")") + 2 :].split()[19]
    except (OSError, IndexError):
        return None


def _failure(stage: str, category: str, message: str) -> dict[str, Any]:
    retry_unchanged = category == "cancelled"
    return {
        "stage": stage,
        "category": category,
        "message": message[:2048],
        "retry_unchanged": retry_unchanged,
        "retry_after_change": not retry_unchanged,
        "suggested_actions": (
            []
            if category == "cancelled"
            else ["Inspect retained diagnostics before submitting another job."]
        ),
        "exception_type": "",
    }


class JobManager:
    """Output-root locking makes concurrency and recovery apply across servers."""

    def __init__(
        self, policy: Policy, *, executor: Any = None, autostart: bool = True
    ) -> None:
        if not sys.platform.startswith("linux"):
            raise RuntimeError(
                "Unsupported environment: the job manager requires Linux "
                "process-group and /proc identity tracking for audited crash recovery"
            )
        self.policy = policy
        self.executor = executor
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._queue: deque[str] = deque()
        self._records: dict[str, dict[str, Any]] = {}
        self._active: dict[str, Any] | None = None
        self._context = multiprocessing.get_context("spawn")
        self._live = confined_path(policy.output_root, ".jobs", existing=False)
        self._live.mkdir(mode=0o700, exist_ok=True)
        lock_path = confined_path(self._live, "manager.lock", existing=False)
        self._lock_file = lock_path.open("a+b")
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt

                self._lock_file.write(b"0")
                self._lock_file.flush()
                self._lock_file.seek(0)
                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self._lock_file.close()
            raise RuntimeError("another adapter owns this output root") from exc
        try:
            self._recover()
        except BaseException:
            self._lock_file.close()
            raise
        self._thread = threading.Thread(
            target=self._run, name="torch-tensorrt-jobs", daemon=True
        )
        if autostart:
            self._thread.start()

    def _staging(self, job_id: str) -> Path:
        return confined_path(self.policy.output_root, f".{job_id}.tmp", existing=False)

    def _persist(self, record: dict[str, Any]) -> None:
        record["updated_at"] = utc_now()
        write_json(
            confined_path(self._live, record["job_id"] + ".json", existing=False),
            record,
        )

    def _event(self, record: dict[str, Any], event: dict[str, Any]) -> None:
        record["sequence"] += 1
        stage = event.get("stage", _STAGES[record["operation"]])
        if stage not in {
            "environment",
            "capture",
            "analysis",
            "lowering",
            "partitioning",
            "compilation",
            "conversion",
            "build",
            "serialization",
            "save",
            "validation",
            "benchmark",
            "finalization",
        }:
            stage = _STAGES[record["operation"]]
        kind = event.get("kind", "progress")
        if kind not in {"start", "end", "progress", "failure"}:
            kind = "progress"
        details = dict(event.get("details", {}))
        if "message" in event:
            details["message"] = str(event["message"])[:2048]
        document = {
            "report_type": "event",
            "schema_version": REPORT_SCHEMA_VERSION,
            "sequence": record["sequence"],
            "job_id": record["job_id"],
            "stage": stage,
            "kind": kind,
            "timestamp": utc_now(),
            "stage_id": str(event.get("stage_id", stage))[:256],
            "details": details,
        }
        serialized = json.dumps(document, sort_keys=True, allow_nan=False)
        if len(serialized.encode()) > 65536:
            document["details"] = {
                "message": "Event details exceeded 64 KiB and were omitted"
            }
            serialized = json.dumps(document, sort_keys=True)
        path = self._staging(record["job_id"]) / "events.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        record["stage"] = stage
        self._persist(record)

    def _recover(self) -> None:
        for path in sorted(self._live.glob("*.json")):
            record = read_json(confined_path(self._live, path.name))
            job_id = record["job_id"]
            if path.name != job_id + ".json":
                raise PolicyError("live record has mismatched job identity")
            # Validate identifiers before using persisted paths or processes.
            from .models import JobRequest

            JobRequest(job_id=job_id)
            self._records[job_id] = record
            if record["status"] in TERMINAL:
                continue
            pid = record.get("pid")
            token = record.get("process_token")
            if pid and token and process_token(pid) == token:
                self._signal_pid(pid, signal.SIGTERM)
                deadline = time.monotonic() + 1.0
                while process_token(pid) == token and time.monotonic() < deadline:
                    time.sleep(0.02)
                if process_token(pid) == token:
                    self._signal_pid(pid, signal.SIGKILL)
                    while process_token(pid) == token:
                        time.sleep(0.02)
            final_path = confined_path(
                self.policy.output_root, record["output_dir"], existing=False
            )
            if final_path.is_dir():
                manifest = read_json(final_path / "manifest.json")
                record["status"] = manifest["status"]
                record["failure"] = manifest.get("failure")
                record["finished_at"] = utc_now()
                self._persist(record)
                continue
            self._finalize(
                record,
                "failed",
                _failure(
                    _STAGES[record["operation"]],
                    "worker_crash",
                    "Adapter restarted before the job reached a terminal state",
                ),
            )

    def submit(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation not in REQUEST_TYPES:
            raise ValueError("unknown execution operation")
        request = (
            REQUEST_TYPES[operation].model_validate(payload).model_dump(mode="json")
        )
        self.policy.validate_request(operation, request)
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("job manager is shutting down")
            if (
                len(self._queue) + int(self._active is not None)
                >= self.policy.max_pending_jobs
            ):
                raise RuntimeError("administrator job queue limit reached")
            source = None
            if operation in ("validate", "benchmark"):
                source_status = self.status(request["source_job_id"])
                if (
                    source_status["status"] != "succeeded"
                    or source_status["operation"] != "compile"
                ):
                    raise PolicyError(
                        "source_job_id must identify a successful immutable compilation"
                    )
                source = self._records[request["source_job_id"]]
            job_id = uuid.uuid4().hex
            created_at = utc_now()
            staging = self._staging(job_id)
            staging.mkdir(mode=0o700)
            (staging / "diagnostics").mkdir(mode=0o700)
            (staging / "artifacts").mkdir(mode=0o700)
            write_json(
                staging / "request.json", {"operation": operation, "request": request}
            )
            record = {
                "job_id": job_id,
                "operation": operation,
                "status": "queued",
                "stage": _STAGES[operation],
                "sequence": 0,
                "created_at": created_at,
                "updated_at": created_at,
                "output_dir": _output_dir_name(
                    operation,
                    request,
                    job_id,
                    created_at,
                    source,
                ),
                "started_at": None,
                "finished_at": None,
                "pid": None,
                "process_token": None,
                "failure": None,
                "request": request,
            }
            self._records[job_id] = record
            self._event(record, {"kind": "start", "message": "Job queued"})
            self._queue.append(job_id)
            return self.status(job_id)

    def status(self, job_id: str) -> dict[str, Any]:
        from .models import JobRequest

        JobRequest(job_id=job_id)
        with self._lock:
            if job_id not in self._records:
                raise PolicyError("unknown job ID")
            record = self._records[job_id]
            return {
                key: value
                for key, value in record.items()
                if key not in {"request", "pid", "process_token"}
            }

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            self.status(job_id)
            record = self._records[job_id]
            if (
                self._active is not None
                and self._active["job_id"] == job_id
                and not self._active["process"].is_alive()
            ):
                self._tick()
                record = self._records[job_id]
            if (
                record["status"] in TERMINAL
                or record["status"] == "cancellation_requested"
            ):
                return self.status(job_id)
            record["status"] = "cancellation_requested"
            self._event(record, {"message": "Cancellation requested"})
            if job_id in self._queue:
                self._queue.remove(job_id)
                self._finalize(
                    record,
                    "cancelled",
                    _failure(
                        record["stage"],
                        "cancelled",
                        "Job cancelled before its worker started",
                    ),
                )
            elif self._active is not None and self._active["job_id"] == job_id:
                self._active["termination"] = "cancelled"
                self._active["terminate_at"] = time.monotonic()
                self._terminate(self._active["process"])
            return self.status(job_id)

    @staticmethod
    def _signal_pid(pid: int, sig: int) -> None:
        try:
            if os.name == "posix" and os.getpgid(pid) == pid:
                os.killpg(pid, sig)
            else:
                os.kill(pid, sig)
        except ProcessLookupError:
            pass

    def _terminate(self, process: Any, *, kill: bool = False) -> None:
        if os.name == "posix":
            self._signal_pid(process.pid, signal.SIGKILL if kill else signal.SIGTERM)
        elif kill:
            process.kill()
        else:
            process.terminate()

    def _start_next(self) -> None:
        job_id = self._queue[0]
        record = self._records[job_id]
        record["status"] = "running"
        record["started_at"] = utc_now()
        self._event(record, {"kind": "start", "message": "Execution worker starting"})
        self._queue.popleft()
        receiver, sender = self._context.Pipe(duplex=False)
        start_receiver, start_sender = self._context.Pipe(duplex=False)
        process = self._context.Process(
            target=worker_main,
            args=(
                self.policy,
                record["operation"],
                record["request"],
                str(self._staging(job_id)),
                job_id,
                sender,
                start_receiver,
                self.executor,
            ),
            name="torch-tensorrt-" + job_id,
        )
        try:
            process.start()
        except Exception as exc:
            receiver.close()
            sender.close()
            start_receiver.close()
            start_sender.close()
            failure = _failure(
                record["stage"],
                "worker_crash",
                (
                    "Could not spawn execution worker: "
                    f"{type(exc).__name__}: {str(exc) or '<no message>'}"
                ),
            )
            failure["exception_type"] = type(exc).__name__
            self._finalize(
                record,
                "failed",
                failure,
            )
            return
        sender.close()
        start_receiver.close()
        record["pid"] = process.pid
        record["process_token"] = process_token(process.pid)
        self._active = {
            "job_id": job_id,
            "process": process,
            "receiver": receiver,
            "deadline": time.monotonic() + record["request"]["timeout_seconds"],
            "termination": None,
            "terminate_at": None,
        }
        try:
            self._persist(record)
            start_sender.send("start")
        finally:
            start_sender.close()

    def _tick(self) -> None:
        with self._lock:
            if self._active is None:
                if self._queue and not self._stop.is_set():
                    self._start_next()
                return
            active = self._active
            record = self._records[active["job_id"]]
            process, receiver = active["process"], active["receiver"]
            for _ in range(256):
                if not receiver.poll():
                    break
                try:
                    event = receiver.recv()
                except EOFError:
                    break
                if isinstance(event, dict):
                    self._event(record, event)
            now = time.monotonic()
            if (
                process.is_alive()
                and active["termination"] is None
                and now >= active["deadline"]
            ):
                active["termination"] = "timeout"
                active["terminate_at"] = now
                record["status"] = "cancellation_requested"
                self._event(record, {"message": "Execution timeout reached"})
                self._terminate(process)
            if (
                process.is_alive()
                and active["termination"]
                and now - active["terminate_at"] >= 1.0
            ):
                self._terminate(process, kill=True)
            if process.is_alive():
                return
            while receiver.poll():
                try:
                    event = receiver.recv()
                except EOFError:
                    break
                if isinstance(event, dict):
                    self._event(record, event)
            process.join(timeout=0)
            receiver.close()
            termination = active["termination"]
            if termination:
                status = "cancelled" if termination == "cancelled" else "failed"
                failure = _failure(
                    record["stage"],
                    termination,
                    (
                        "Job cancelled after its worker exited"
                        if termination == "cancelled"
                        else "Execution worker exceeded the configured timeout"
                    ),
                )
            else:
                result_path = self._staging(record["job_id"]) / "worker-result.json"
                if process.exitcode != 0 or not result_path.is_file():
                    status, failure = (
                        "failed",
                        _failure(
                            record["stage"],
                            "worker_crash",
                            f"Worker exited without a result (exit code {process.exitcode})",
                        ),
                    )
                else:
                    result = read_json(result_path)
                    status, failure = result["status"], result.get("failure")
                    if failure:
                        failure = {
                            **_failure(
                                record["stage"], failure["category"], failure["message"]
                            ),
                            **failure,
                        }
            self._finalize(record, status, failure)
            process.close()
            self._active = None

    def _finalize(
        self, record: dict[str, Any], status: str, failure: dict[str, Any] | None = None
    ) -> None:
        job_id = record["job_id"]
        path = self._staging(job_id)
        path.mkdir(mode=0o700, exist_ok=True)
        for folder in ("diagnostics", "artifacts"):
            (path / folder).mkdir(mode=0o700, exist_ok=True)
        self._event(
            record,
            {
                "stage": "finalization",
                "kind": "end" if status == "succeeded" else "failure",
                "message": "Job finalized as " + status,
            },
        )
        if failure:
            write_json(
                path / "failure.json",
                {
                    "report_type": "base",
                    "schema_version": REPORT_SCHEMA_VERSION,
                    "job_id": job_id,
                    "status": status,
                    "created_at": utc_now(),
                    "failure": failure,
                    "diagnostics": [],
                },
            )
        entries = []
        for file in sorted(path.rglob("*")):
            if file.is_symlink():
                raise PolicyError("worker outputs contain a symlink")
            if file.is_file() and file.name != "manifest.json":
                relative = file.relative_to(path).as_posix()
                artifact_id = (
                    "compiled_model"
                    if relative == "artifacts/compiled.pt2"
                    else relative
                )
                entries.append(
                    {
                        "artifact_id": artifact_id,
                        "path": relative,
                        "sha256": file_hash(file),
                        "size_bytes": file.stat().st_size,
                        "kind": (
                            "report"
                            if file.suffix == ".json"
                            and file.stem
                            in {
                                "environment",
                                "analysis",
                                "compilation",
                                "validation",
                                "benchmark",
                                "failure",
                            }
                            else "file"
                        ),
                    }
                )
        request = (
            read_json(path / "request.json")
            if (path / "request.json").is_file()
            else {"operation": record["operation"], "request": record["request"]}
        )
        lineage = (
            {"source_job_id": record["request"]["source_job_id"]}
            if "source_job_id" in record["request"]
            else {}
        )
        environment = (
            read_json(path / "environment.json")
            if (path / "environment.json").is_file()
            else None
        )
        manifest = {
            "report_type": "artifact_manifest",
            "schema_version": REPORT_SCHEMA_VERSION,
            "job_id": job_id,
            "status": status,
            "created_at": utc_now(),
            "failure": failure,
            "diagnostics": [],
            "entries": entries,
            "environment": environment,
            "request_fingerprint": hashlib.sha256(
                json.dumps(
                    request,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
            "lineage": lineage,
            "absent_stage_reasons": {
                stage: "Stage not requested or not reached"
                for stage in (
                    "environment",
                    "analysis",
                    "compilation",
                    "validation",
                    "benchmark",
                )
                if not (path / (stage + ".json")).is_file()
            },
        }
        write_json(path / "manifest.json", manifest)
        final_path = confined_path(
            self.policy.output_root, record["output_dir"], existing=False
        )
        if final_path.exists():
            raise PolicyError("job output is create-only")
        os.rename(path, final_path)
        record["status"] = status
        record["failure"] = failure
        record["finished_at"] = utc_now()
        try:
            self._persist(record)
        except Exception as exc:
            raise _PublishedFinalizationError(
                "job result was published but live-state persistence failed"
            ) from exc

    def _run(self) -> None:
        while not self._stop.wait(0.05):
            try:
                self._tick()
            except Exception as exc:
                with self._lock:
                    if self._active is not None:
                        active = self._active
                        record = self._records[active["job_id"]]
                        if isinstance(exc, _PublishedFinalizationError):
                            record["live_state_persistence_failed"] = True
                        else:
                            self._terminate(active["process"], kill=True)
                            active["process"].join(timeout=2)
                            failure = _failure(
                                "finalization",
                                "artifact_error",
                                "Supervisor could not persist or finalize worker output; partial diagnostics remain in staging",
                            )
                            # Report a terminal outcome only after the child is gone.
                            record["failure"] = failure
                            record["status"] = (
                                "cancellation_requested"
                                if active["process"].is_alive()
                                else "failed"
                            )
                            record["retained_staging_dir"] = self._staging(
                                record["job_id"]
                            ).name
                            if record["status"] == "failed":
                                record["finished_at"] = utc_now()
                            try:
                                write_json(
                                    self._staging(record["job_id"]) / "failure.json",
                                    {
                                        "report_type": "base",
                                        "schema_version": REPORT_SCHEMA_VERSION,
                                        "job_id": record["job_id"],
                                        "status": "failed",
                                        "created_at": utc_now(),
                                        "failure": failure,
                                        "diagnostics": [],
                                    },
                                )
                                self._persist(record)
                            except OSError:
                                # Polling still exposes the in-memory failure when
                                # storage itself is unusable; never claim a manifest.
                                pass
                        if not active["process"].is_alive():
                            active["receiver"].close()
                            active["process"].close()
                            self._active = None
                    for job_id in list(self._queue):
                        self._queue.remove(job_id)
                        record = self._records[job_id]
                        try:
                            self._finalize(
                                record,
                                "failed",
                                _failure(
                                    "finalization",
                                    "artifact_error",
                                    "Supervisor stopped after an artifact persistence failure",
                                ),
                            )
                        except _PublishedFinalizationError:
                            record["live_state_persistence_failed"] = True
                        except (OSError, PolicyError):
                            record["status"] = "failed"
                            record["failure"] = _failure(
                                "finalization",
                                "artifact_error",
                                "Supervisor stopped and could not finalize queued job",
                            )
                self._stop.set()

    def close(self) -> None:
        with self._lock:
            self._stop.set()
            for job_id in list(self._queue):
                self.cancel(job_id)
            if self._active is not None:
                self.cancel(self._active["job_id"])
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        while self._active is not None:
            self._tick()
            time.sleep(0.02)
        self._lock_file.close()

    def get_manifest(self, job_id: str) -> dict[str, Any]:
        status = self.status(job_id)
        if status["status"] not in TERMINAL:
            raise PolicyError("manifest is available after the worker exits")
        if "retained_staging_dir" in status:
            raise PolicyError(
                "manifest finalization failed; inspect retained staging diagnostics from get_job_status"
            )
        return self.bounded(
            read_json(confined_path(self.policy.job_path(job_id), "manifest.json"))
        )

    def get_report(self, job_id: str, report: str) -> dict[str, Any]:
        from .models import ReportRequest

        ReportRequest(job_id=job_id, report=report)
        if self.status(job_id)["status"] not in TERMINAL:
            raise PolicyError("report is available after the worker exits")
        root = self.policy.job_path(job_id)
        manifest = read_json(confined_path(root, "manifest.json"))
        relative = report + ".json"
        entry = next(
            (item for item in manifest["entries"] if item["path"] == relative), None
        )
        if entry is None:
            raise PolicyError("requested report is absent from this job")
        path = confined_path(root, relative)
        if (
            path.stat().st_size != entry["size_bytes"]
            or file_hash(path) != entry["sha256"]
        ):
            raise PolicyError("report does not match its immutable manifest")
        return self.bounded(read_json(path))

    def bounded(self, document: dict[str, Any]) -> dict[str, Any]:
        size = len(json.dumps(document, allow_nan=False).encode())
        if size <= self.policy.max_response_bytes:
            return document
        return {
            "job_id": document.get("job_id"),
            "report_type": document.get("report_type"),
            "status": document.get("status"),
            "response_omitted": True,
            "size_bytes": size,
            "reason": "Document exceeds administrator response size limit; inspect the manifest-linked local report.",
        }
