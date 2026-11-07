"""Spawn-importable fake core, exercising real supervisor/process boundaries."""

import os
import time

from torch_tensorrt_mcp.storage import utc_now, write_json


def fake_core(policy, operation, request, path, job_id, emit):
    mode = request.get("model_parameters", {}).get("mode", "success")
    if mode == "crash":
        os._exit(23)
    if mode == "block":
        time.sleep(30)
    if mode == "error":
        raise ValueError("invalid fake model")
    if mode == "symlink":
        (path / "artifacts" / "outside").symlink_to(
            policy.workspace_root / "factories.py"
        )
    emit({"stage": "build", "message": "fake engine built"})
    report = {
        "report_type": "base",
        "schema_version": "0.1",
        "job_id": job_id,
        "created_at": utc_now(),
        "status": "succeeded",
        "failure": None,
        "diagnostics": [],
    }
    if mode == "report_failed":
        report["status"] = "failed"
        report["failure"] = {
            "stage": "validation",
            "category": "validation_error",
            "message": "Numerical validation failed",
            "retry_unchanged": False,
            "retry_after_change": True,
            "suggested_actions": [],
            "exception_type": "",
        }
    name = {
        "collect_environment": "environment",
        "analyze": "analysis",
        "compile": "compilation",
        "validate": "validation",
        "benchmark": "benchmark",
    }[operation]
    write_json(path / (name + ".json"), report)
    if operation == "compile":
        (path / "artifacts" / "compiled.pt2").write_bytes(b"fake compiled program")
    return name
