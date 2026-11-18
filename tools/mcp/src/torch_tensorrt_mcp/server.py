"""Optional official MCP v2 adapter. This executable serves stdio only."""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from . import __version__
from .jobs import JobManager
from .models import (
    BenchmarkRequest,
    EnvironmentRequest,
    JobRequest,
    ModelRequest,
    ReportRequest,
    ValidationRequest,
)
from .policy import Policy


def create_server(policy: Policy, *, manager: JobManager | None = None) -> Any:
    # Keep this import out of lifecycle/policy modules and unit tests.
    from mcp.server import MCPServer

    active: dict[str, JobManager] = {}
    lifecycle_state = "not_started"

    @asynccontextmanager
    async def lifespan(server: Any) -> Any:
        nonlocal lifecycle_state
        lifecycle_state = "starting"
        active["manager"] = manager or JobManager(policy)
        lifecycle_state = "running"
        try:
            yield active["manager"]
        finally:
            lifecycle_state = "stopping"
            try:
                active["manager"].close()
            finally:
                active.clear()
                lifecycle_state = "stopped"

    server = MCPServer(
        "torch-tensorrt",
        version=__version__,
        instructions="Execution tools return job IDs immediately. Poll get_job_status; use immutable completed job reports and manifests for evidence. validate and benchmark reference source_job_id from a successful compile job. Factories are configured by the administrator.",
        lifespan=lifespan,
    )

    def get_job_manager() -> JobManager:
        if "manager" not in active:
            raise RuntimeError(
                "Torch-TensorRT JobManager is unavailable: MCP server lifecycle "
                f"state is {lifecycle_state!r}; tools may only be called while "
                "the server lifecycle state is 'running'"
            )
        return active["manager"]

    @server.tool()
    def collect_environment(request: EnvironmentRequest) -> dict[str, Any]:
        """Collect runtime provenance in an isolated worker; returns a job ID."""
        return get_job_manager().submit(
            "collect_environment", request.model_dump(mode="json")
        )

    @server.tool()
    def analyze(request: ModelRequest) -> dict[str, Any]:
        """Preflight an administrator-configured model without building engines."""
        return get_job_manager().submit("analyze", request.model_dump(mode="json"))

    @server.tool(name="compile")
    def compile_model(request: ModelRequest) -> dict[str, Any]:
        """Compile a configured model and persist a reloadable .pt2 artifact."""
        return get_job_manager().submit("compile", request.model_dump(mode="json"))

    @server.tool()
    def get_job_status(request: JobRequest) -> dict[str, Any]:
        """Read durable job status, most recent stage, and event sequence."""
        return get_job_manager().status(request.job_id)

    @server.tool()
    def cancel_job(request: JobRequest) -> dict[str, Any]:
        """Request cancellation; cancelled is reported only after worker exit."""
        return get_job_manager().cancel(request.job_id)

    @server.tool()
    def validate(request: ValidationRequest) -> dict[str, Any]:
        """Compare the persisted compile artifact with its configured reference."""
        return get_job_manager().submit("validate", request.model_dump(mode="json"))

    @server.tool()
    def benchmark(request: BenchmarkRequest) -> dict[str, Any]:
        """Benchmark the persisted compile artifact against the same reference."""
        return get_job_manager().submit("benchmark", request.model_dump(mode="json"))

    @server.tool()
    def get_report(request: ReportRequest) -> dict[str, Any]:
        """Read a named, manifest-verified report from a completed job."""
        return get_job_manager().get_report(request.job_id, request.report)

    @server.tool()
    def get_artifact_manifest(request: JobRequest) -> dict[str, Any]:
        """Read an immutable manifest; large documents receive bounded metadata."""
        return get_job_manager().get_manifest(request.job_id)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Policy-controlled Torch-TensorRT MCP over stdio"
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="administrator JSON configuration"
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    create_server(Policy.load(arguments.config)).run(transport="stdio")


if __name__ == "__main__":
    main()
