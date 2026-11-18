import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from helpers import fake_core
from torch_tensorrt_mcp.jobs import JobManager
from torch_tensorrt_mcp.server import create_server

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mcp") is None, reason="optional MCP SDK is not installed"
)


def test_in_memory_tools_and_strict_requests(policy):
    from mcp import Client

    async def run():
        manager = JobManager(policy, executor=fake_core)
        server = create_server(policy, manager=manager)
        async with Client(server, read_timeout_seconds=5.0) as client:
            tools = await client.list_tools()
            names = {tool.name for tool in tools.tools}
            assert names == {
                "collect_environment",
                "analyze",
                "compile",
                "get_job_status",
                "cancel_job",
                "validate",
                "benchmark",
                "get_report",
                "get_artifact_manifest",
            }
            invalid = await client.call_tool(
                "compile",
                {
                    "request": {
                        "model": "model",
                        "inputs": "inputs",
                        "settings": {"engine_cache_dir": "/tmp/escape"},
                    }
                },
            )
            assert invalid.is_error
            result = await client.call_tool(
                "compile", {"request": {"model": "model", "inputs": "inputs"}}
            )
            assert not result.is_error
            job_id = result.structured_content["job_id"]
            while manager.status(job_id)["status"] not in {
                "succeeded",
                "failed",
                "cancelled",
            }:
                await asyncio.sleep(0.02)
            status = await client.call_tool(
                "get_job_status", {"request": {"job_id": job_id}}
            )
            assert status.structured_content["status"] == "succeeded"
            manifest = await client.call_tool(
                "get_artifact_manifest", {"request": {"job_id": job_id}}
            )
            assert manifest.structured_content["report_type"] == "artifact_manifest"

    asyncio.run(run())


def test_stdio_discovery_without_loading_gpu_libraries(policy, tmp_path):
    from mcp import Client, StdioServerParameters

    config = tmp_path / "stdio.json"
    config.write_text(
        json.dumps(
            {
                "workspace_root": str(policy.workspace_root),
                "output_root": str(policy.output_root),
                "models": {},
                "inputs": {},
            }
        )
    )
    source = str(Path(__file__).resolve().parents[1] / "src")

    async def run():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "torch_tensorrt_mcp.server", "--config", str(config)],
            env={**os.environ, "PYTHONPATH": source},
        )
        async with Client(parameters, read_timeout_seconds=5.0) as client:
            tools = await client.list_tools()
            assert len(tools.tools) == 9

    asyncio.run(run())
