"""Regenerate versioned fixtures when reviewing a reporting contract change."""

import json
from pathlib import Path

from isolation import load_reporting

r = load_reporting()


def fixtures():
    timestamp = "2026-10-12T00:00:00Z"
    common = {"job_id": "example-job", "created_at": timestamp}
    environment = r.EnvironmentReport(
        **common,
        python_version="3.12.11",
        platform="Linux",
        torch_version="2.15.0",
        torch_tensorrt_version="2.15.0",
        tensorrt_version="11.3.0",
        torch_cuda_build_version="13.0",
        cuda_runtime_version="13.0",
        cuda_driver_version="13.0",
        nvidia_driver_version="580.65.06",
        cuda_available=True,
        gpus=[
            r.GPUInfo(
                name="Example GPU",
                uuid="GPU-example",
                compute_capability="9.0",
                total_memory_bytes=1024,
            )
        ],
    )
    analysis = r.AnalysisReport(
        **common,
        settings={"min_block_size": 1},
        nodes=[
            r.NodeDecision(
                node_id="relu",
                operator="torch.ops.aten.relu.default",
                eligible=True,
                assignment=r.NodeAssignment.TENSORRT,
                confidence=r.Confidence.CONFIRMED,
                source_node_ids=["source_relu"],
            )
        ],
        partitions=[
            r.Partition(
                partition_id="engine-0",
                node_ids=["relu"],
                input_profiles=[{"shape": [2, 3]}],
            )
        ],
        coverage=r.Coverage(total=1, eligible=1, assigned=1),
        checks=[
            r.Check(
                name="converter_validator",
                status=r.CheckStatus.PASSED,
            ),
            r.Check(
                name="engine_build",
                status=r.CheckStatus.SKIPPED,
                detail="Analysis does not build engines",
            ),
        ],
        requested_settings_satisfiable=True,
    )
    compilation = r.CompilationReport(
        **common,
        analysis=analysis,
        settings={"min_block_size": 1},
        engines=[
            r.EngineReport(
                engine_id="engine-0",
                partition_id="engine-0",
                elapsed_seconds=0.25,
            )
        ],
        artifact_ids=["artifacts/compiled.pt2"],
    )
    validation = r.ValidationReport(
        **common,
        seed=42,
        samples=[
            r.ValidationSample(
                sample_id="sample-0",
                label="opt",
                profile="profile-0",
            )
        ],
        outputs=[
            r.OutputComparison(
                sample_id="sample-0",
                path="$",
                passed=True,
                reference_shape=[2, 3],
                actual_shape=[2, 3],
                reference_dtype="torch.float32",
                actual_dtype="torch.float32",
                numel=6,
                max_abs_error=0.0,
                mean_abs_error=0.0,
                max_relative_error=0.0,
                mean_relative_error=0.0,
            )
        ],
        sample_count=1,
        passed_samples=1,
        profile_points=["profile-0"],
    )
    benchmark = r.BenchmarkReport(
        **common,
        device="cuda:0",
        seed=42,
        protocol=r.BenchmarkProtocol(warmup=1, iterations=2),
        reference=r.TimingMeasurements(
            wall_samples_ms=[2.0, 2.1],
            device_samples_ms=[1.0, 1.1],
            wall_statistics={"median": 2.05},
            device_statistics={"median": 1.05},
        ),
        compiled=r.TimingMeasurements(
            wall_samples_ms=[1.0, 1.1],
            device_samples_ms=[0.5, 0.6],
            wall_statistics={"median": 1.05},
            device_statistics={"median": 0.55},
        ),
        speedup={"wall": 2.05 / 1.05, "device": 1.05 / 0.55},
    )
    manifest = r.ArtifactManifest(
        **common,
        environment=environment,
        entries=[
            r.ArtifactEntry(
                artifact_id="artifacts/compiled.pt2",
                path="artifacts/compiled.pt2",
                sha256="0" * 64,
                size_bytes=16,
            )
        ],
        request_fingerprint="1" * 64,
        lineage={"source_graph_sha256": "2" * 64},
        absent_stage_reasons={"capture": "Input was already an ExportedProgram"},
    )
    event = r.Event(
        job_id="example-job",
        timestamp=timestamp,
        sequence=1,
        stage=r.Stage.BUILD,
        kind=r.EventKind.PROGRESS,
        stage_id="engine-0",
        details={"completed": 1, "total": 2},
    )
    return [
        r.BaseReport(**common),
        environment,
        analysis,
        compilation,
        validation,
        benchmark,
        manifest,
        event,
    ]


def main():
    root = Path(__file__).parent
    repository_root = Path(__file__).resolve().parents[3]
    version_directory = f"v{r.SCHEMA_VERSION}"
    for instance in fixtures():
        name = instance.report_type + ".json"
        for directory, value in (
            (root / "golden" / version_directory, instance.to_dict()),
            (
                repository_root / "schemas" / "reporting" / version_directory,
                r.json_schema(type(instance)),
            ),
        ):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / name).write_text(
                json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
