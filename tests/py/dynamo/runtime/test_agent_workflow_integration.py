# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Real local-worker workflow, including dynamic persisted artifact guards."""

import json
import sys
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dynamic", [False, True])
def test_agent_workflow_uses_immutable_reloaded_artifacts(tmp_path, dynamic):
    pytest.importorskip("mcp")
    source = Path(__file__).resolve().parents[4] / "tools" / "mcp" / "src"
    sys.path.insert(0, str(source))
    from torch_tensorrt.artifacts import verify_manifest
    from torch_tensorrt.reporting import ArtifactManifest, from_json
    from torch_tensorrt_mcp.jobs import JobManager
    from torch_tensorrt_mcp.policy import Policy

    workspace = tmp_path / "workspace"
    output = tmp_path / "output"
    workspace.mkdir()
    output.mkdir()
    factory = workspace / "model.py"
    factory.write_text(
        """
import torch
from torch_tensorrt.verification import InputSample

class AddRelu(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + x)

def model(dynamic=False):
    module = AddRelu().cuda().eval()
    if not dynamic:
        return module
    sample = torch.randn(2, 3, device="cuda", dtype=torch.float16)
    return torch.export.export(module, (sample,), dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=4)},))

def inputs(seed, dynamic=False):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    points = [(1, "min"), (2, "opt"), (4, "max")] if dynamic else [(2, "opt")]
    return [InputSample(args=(torch.randn(batch, 3, generator=generator, device="cuda", dtype=torch.float16),), sample_id=str(batch), profile=label) for batch, label in points]
""",
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "workspace_root": str(workspace),
                "output_root": str(output),
                "models": {
                    "small": {
                        "path": "model.py",
                        "function": "model",
                        "parameters": ["dynamic"],
                    }
                },
                "inputs": {
                    "sample": {
                        "path": "model.py",
                        "function": "inputs",
                        "parameters": ["dynamic"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manager = JobManager(Policy.load(config))

    def run(operation, request):
        submitted = manager.submit(operation, request)
        job_id = submitted["job_id"]
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            status = manager.status(job_id)
            if status["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        job_root = output / status["output_dir"]
        assert status["status"] == "succeeded", (
            status,
            (job_root / "diagnostics" / "stderr.log").read_text()[-5000:],
        )
        manifest = ArtifactManifest.from_json((job_root / "manifest.json").read_text())
        assert verify_manifest(manifest, job_root) == []
        events = [
            from_json(line)
            for line in (job_root / "events.jsonl").read_text().splitlines()
        ]
        assert [event.sequence for event in events] == list(range(1, len(events) + 1))
        return job_id

    request = {
        "model": "small",
        "inputs": "sample",
        "model_parameters": {"dynamic": dynamic},
        "input_parameters": {"dynamic": dynamic},
        "settings": {"min_block_size": 1},
        "timeout_seconds": 90.0,
    }
    try:
        run("collect_environment", {"timeout_seconds": 90.0})
        analysis_id = run("analyze", request)
        assert manager.get_report(analysis_id, "analysis")["coverage"]["assigned"] == 2
        compilation_id = run("compile", request)
        compilation_root = output / manager.status(compilation_id)["output_dir"]
        before = (compilation_root / "manifest.json").read_bytes()
        validation_id = run(
            "validate",
            {
                "source_job_id": compilation_id,
                "atol": 1e-3,
                "rtol": 1e-3,
                "timeout_seconds": 90.0,
            },
        )
        validation = manager.get_report(validation_id, "validation")
        assert validation["passed"]
        assert validation["sample_count"] == (3 if dynamic else 1)
        benchmark_id = run(
            "benchmark",
            {
                "source_job_id": compilation_id,
                "warmup": 2,
                "iterations": 5,
                "sample_index": 1 if dynamic else 0,
                "timeout_seconds": 90.0,
            },
        )
        timing = manager.get_report(benchmark_id, "benchmark")
        assert len(timing["compiled"]["device_samples_ms"]) == 5
        assert timing["protocol"]["gpu_isolation"] == "unknown"
        assert (
            timing["speedup"]["device"]
            == timing["reference"]["device_statistics"]["median"]
            / timing["compiled"]["device_statistics"]["median"]
        )
        assert (compilation_root / "manifest.json").read_bytes() == before
        assert (
            manager.get_manifest(validation_id)["lineage"]["source_job_id"]
            == compilation_id
        )
    finally:
        manager.close()
