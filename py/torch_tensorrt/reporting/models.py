# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned, dependency-free evidence contracts for compiler workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar

SCHEMA_VERSION = "0.1"


def utc_now() -> str:
    """Return the current UTC timestamp in the contract's ISO-8601 format."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


class ReportStatus(str, Enum):
    """Lifecycle or final outcome of a report-producing workflow."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class Stage(str, Enum):
    """Workflow step where an event or failure occurred."""

    ENVIRONMENT = "environment"
    CAPTURE = "capture"
    ANALYSIS = "analysis"
    COMPILATION = "compilation"
    LOWERING = "lowering"
    PARTITIONING = "partitioning"
    CONVERSION = "conversion"
    BUILD = "build"
    SERIALIZATION = "serialization"
    SAVE = "save"
    VALIDATION = "validation"
    BENCHMARK = "benchmark"
    FINALIZATION = "finalization"


class FailureCategory(str, Enum):
    """Machine-readable classification of why a workflow failed."""

    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    UNSUPPORTED_ENVIRONMENT = "unsupported_environment"
    BUILDER_ERROR = "builder_error"
    OUT_OF_MEMORY = "out_of_memory"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    WORKER_CRASH = "worker_crash"
    SERIALIZATION_ERROR = "serialization_error"
    ARTIFACT_ERROR = "artifact_error"
    VALIDATION_ERROR = "validation_error"
    BENCHMARK_ERROR = "benchmark_error"
    INTERNAL_ERROR = "internal_error"


class NodeKind(str, Enum):
    """Semantic role of a node in the analyzed graph."""

    COMPUTATIONAL = "computational"
    STRUCTURAL = "structural"
    PARAMETER = "parameter"
    BUFFER = "buffer"


class NodeAssignment(str, Enum):
    """Runtime selected to execute an analyzed graph node."""

    TENSORRT = "tensorrt"
    PYTORCH = "pytorch"
    SKIPPED = "skipped"


class Confidence(str, Enum):
    """Whether evidence confirms a decision or only predicts it."""

    CONFIRMED = "confirmed"
    PREDICTED = "predicted"


class CheckStatus(str, Enum):
    """Outcome of one analysis or contract check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EventKind(str, Enum):
    """Lifecycle transition represented by an emitted event."""

    START = "start"
    END = "end"
    PROGRESS = "progress"
    FAILURE = "failure"


class Model:
    """Base class for strict JSON-serializable reporting contracts."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize this contract model to a validated JSON-compatible object."""
        from .serialization import to_dict

        return to_dict(self)

    def to_json(self) -> str:
        """Serialize this contract model to deterministic compact JSON."""
        from .serialization import to_json

        return to_json(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Any:
        """Decode and validate a JSON-compatible object as this model type."""
        from .serialization import from_dict

        return from_dict(value, expected_type=cls)

    @classmethod
    def from_json(cls, text: str) -> Any:
        """Decode and validate a JSON document as this model type."""
        from .serialization import from_json

        return from_json(text, expected_type=cls)


@dataclass(frozen=True)
class Diagnostic(Model):
    """Non-fatal, optionally node- or artifact-scoped diagnostic."""

    code: str
    message: str
    severity: str = "warning"
    node_id: str | None = None
    artifact_id: str | None = None


@dataclass(frozen=True)
class Failure(Model):
    """Machine-readable failure with retry guidance and safe diagnostics."""

    stage: Stage
    category: FailureCategory
    message: str
    retry_unchanged: bool = False
    retry_after_change: bool = True
    suggested_actions: list[str] = field(default_factory=list)
    exception_type: str = ""

    def __post_init__(self) -> None:
        """Enforce the bounded diagnostic-message contract."""
        if len(self.message) > 2048:
            raise ValueError("Failure messages are limited to 2048 characters")


@dataclass(frozen=True)
class BaseReport(Model):
    """Common version, identity, status, failure, and diagnostic fields."""

    report_type: ClassVar[str] = "base"
    schema_version: str = SCHEMA_VERSION
    job_id: str = ""
    status: ReportStatus = ReportStatus.SUCCEEDED
    created_at: str = field(default_factory=utc_now)
    failure: Failure | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the schema version and report creation timestamp."""
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version: {self.schema_version}")
        _validate_timestamp(self.created_at)


@dataclass(frozen=True)
class GPUInfo(Model):
    """Portable description of one visible GPU."""

    name: str = ""
    uuid: str | None = None
    compute_capability: str = ""
    total_memory_bytes: int | None = None
    index: int = 0


@dataclass(frozen=True)
class EnvironmentReport(BaseReport):
    """Software, driver, hardware, and feature evidence for one job."""

    report_type: ClassVar[str] = "environment"
    python_version: str = ""
    platform: str = ""
    torch_version: str | None = None
    torch_tensorrt_version: str | None = None
    tensorrt_version: str | None = None
    torch_cuda_build_version: str | None = None
    cuda_runtime_version: str | None = None
    cuda_driver_version: str | None = None
    nvidia_driver_version: str | None = None
    cuda_available: bool = False
    gpus: list[GPUInfo] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Coverage(Model):
    """Counts describing graph eligibility and TensorRT assignment coverage."""

    total: int = 0
    eligible: int = 0
    assigned: int = 0
    structural: int = 0
    parameters: int = 0
    excluded: int = 0
    unsupported: int = 0
    below_min_block_size: int = 0

    def __post_init__(self) -> None:
        """Validate nonnegative and internally consistent coverage counts."""
        counts = (
            self.total,
            self.eligible,
            self.assigned,
            self.structural,
            self.parameters,
            self.excluded,
            self.unsupported,
            self.below_min_block_size,
        )
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts):
            raise ValueError("Coverage counts must be nonnegative integers")
        if not self.assigned <= self.eligible <= self.total:
            raise ValueError("Coverage must satisfy assigned <= eligible <= total")
        if self.excluded + self.unsupported > self.total:
            raise ValueError("Excluded and unsupported counts cannot exceed total")
        if self.below_min_block_size > self.eligible:
            raise ValueError("Below-minimum count cannot exceed eligible count")

    @property
    def ratio(self) -> float:
        """Return the fraction of total nodes eligible for assignment."""
        return self.eligible / self.total if self.total else 0.0


@dataclass(frozen=True)
class NodeDecision(Model):
    """Per-node eligibility and runtime-assignment decision."""

    node_id: str
    operator: str
    kind: NodeKind = NodeKind.COMPUTATIONAL
    eligible: bool = False
    assignment: NodeAssignment = NodeAssignment.PYTORCH
    reason_codes: list[str] = field(default_factory=list)
    confidence: Confidence = Confidence.PREDICTED
    source_node_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Partition(Model):
    """One ordered group of graph nodes assigned to a runtime."""

    partition_id: str
    node_ids: list[str] = field(default_factory=list)
    input_profiles: list[dict[str, Any]] = field(default_factory=list)
    kind: str = "tensorrt"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Check(Model):
    """Named analysis check and its human-readable result."""

    name: str
    status: CheckStatus
    detail: str = ""


@dataclass(frozen=True)
class AnalysisReport(BaseReport):
    """Graph analysis, partitioning decisions, and coverage evidence."""

    report_type: ClassVar[str] = "analysis"
    settings: dict[str, Any] = field(default_factory=dict)
    nodes: list[NodeDecision] = field(default_factory=list)
    partitions: list[Partition] = field(default_factory=list)
    coverage: Coverage = field(default_factory=Coverage)
    checks: list[Check] = field(default_factory=list)
    capture_performed: bool = False
    model_execution_performed: bool = False
    graph_stage: str = "lowered"
    requested_settings_satisfiable: bool | None = None


@dataclass(frozen=True)
class EngineReport(Model):
    """Build result for one TensorRT engine partition."""

    engine_id: str
    partition_id: str
    status: ReportStatus = ReportStatus.SUCCEEDED
    cache_hit: bool = False
    elapsed_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompilationReport(BaseReport):
    """Compilation settings, analysis lineage, engines, and artifacts."""

    report_type: ClassVar[str] = "compilation"
    analysis: AnalysisReport | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    engines: list[EngineReport] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationSample(Model):
    """Identity and fingerprint of one concrete validation input."""

    sample_id: str
    label: str = ""
    profile: str | None = None
    input_fingerprint: str = ""


@dataclass(frozen=True)
class OutputComparison(Model):
    """Comparison evidence for one output pytree leaf."""

    sample_id: str
    path: str
    passed: bool
    reference_shape: list[int] = field(default_factory=list)
    actual_shape: list[int] = field(default_factory=list)
    reference_dtype: str = ""
    actual_dtype: str = ""
    numel: int = 0
    mismatch_count: int = 0
    reference_nan_count: int = 0
    actual_nan_count: int = 0
    reference_inf_count: int = 0
    actual_inf_count: int = 0
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    max_relative_error: float | None = None
    mean_relative_error: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class ValidationReport(BaseReport):
    """Aggregate correctness result across samples and output leaves."""

    report_type: ClassVar[str] = "validation"
    passed: bool = True
    seed: int | None = None
    atol: float = 1e-5
    rtol: float = 1e-5
    relative_floor: float = 1e-12
    samples: list[ValidationSample] = field(default_factory=list)
    outputs: list[OutputComparison] = field(default_factory=list)
    sample_count: int = 0
    passed_samples: int = 0
    profile_points: list[str] = field(default_factory=list)
    pass_rule: str = "abs(actual-reference) <= atol + rtol * abs(reference)"


@dataclass(frozen=True)
class BenchmarkProtocol(Model):
    """Declared timing method needed to interpret benchmark measurements."""

    warmup: int = 10
    iterations: int = 100
    percentiles: list[float] = field(default_factory=lambda: [50.0, 90.0, 99.0])
    inference_mode: bool = True
    throughput_method: str = "inverse_serial_latency"
    gpu_isolation: str = "unknown"
    timing_scope: str = "forward_only"
    cuda_graphs: bool = False
    seed_reset: str = "per_invocation"
    model_mode: str = "eval"
    input_reset: str = "alias_preserving_snapshot"
    synchronization: str = "per_invocation"
    speedup_method: str = "reference_median_over_compiled_median"

    def __post_init__(self) -> None:
        """Reject protocol identifiers not implemented by the current contract."""
        allowed = {
            "throughput_method": {"inverse_serial_latency"},
            "gpu_isolation": {"unknown", "isolated", "shared"},
            "timing_scope": {"forward_only"},
            "seed_reset": {"per_invocation"},
            "model_mode": {"eval", "eval_or_frozen_export_capture"},
            "input_reset": {"alias_preserving_snapshot"},
            "synchronization": {"per_invocation", "synchronous_cpu"},
            "speedup_method": {"reference_median_over_compiled_median"},
        }
        for name, choices in allowed.items():
            value = getattr(self, name)
            if not isinstance(value, str) or value not in choices:
                raise ValueError(
                    f"BenchmarkProtocol {name} must be one of: "
                    f"{', '.join(sorted(choices))}"
                )
        if type(self.inference_mode) is not bool or type(self.cuda_graphs) is not bool:
            raise ValueError(
                "BenchmarkProtocol inference_mode and cuda_graphs must be booleans"
            )


@dataclass(frozen=True)
class TimingMeasurements(Model):
    """Raw latency samples and their derived statistics."""

    wall_samples_ms: list[float] = field(default_factory=list)
    device_samples_ms: list[float] = field(default_factory=list)
    wall_statistics: dict[str, float] = field(default_factory=dict)
    device_statistics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkReport(BaseReport):
    """Reference and compiled timing evidence plus derived performance."""

    report_type: ClassVar[str] = "benchmark"
    device: str = ""
    seed: int | None = None
    protocol: BenchmarkProtocol = field(default_factory=BenchmarkProtocol)
    reference: TimingMeasurements | None = None
    compiled: TimingMeasurements | None = None
    speedup: dict[str, float] = field(default_factory=dict)
    throughput: dict[str, float] = field(default_factory=dict)
    transfers: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactEntry(Model):
    """Integrity metadata for one job-relative artifact file."""

    artifact_id: str
    path: str
    sha256: str
    size_bytes: int
    kind: str = "file"

    def __post_init__(self) -> None:
        """Validate the portable path, digest, and artifact size."""
        path = PurePosixPath(self.path)
        if (
            not self.path
            or path.is_absolute()
            or "\\" in self.path
            or any(part in (".", "..", "") for part in self.path.split("/"))
        ):
            raise ValueError("Artifact paths must be normalized relative POSIX paths")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("Artifact hashes must be lowercase SHA-256 hex strings")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("Artifact sizes must be nonnegative integers")


@dataclass(frozen=True)
class ArtifactManifest(BaseReport):
    """Integrity manifest and provenance for all persisted job artifacts."""

    report_type: ClassVar[str] = "artifact_manifest"
    entries: list[ArtifactEntry] = field(default_factory=list)
    environment: EnvironmentReport | None = None
    request_fingerprint: str = ""
    lineage: dict[str, Any] = field(default_factory=dict)
    absent_stage_reasons: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate base fields, fingerprints, and manifest entry uniqueness."""
        super().__post_init__()
        if self.request_fingerprint and not re.fullmatch(
            r"[0-9a-f]{64}", self.request_fingerprint
        ):
            raise ValueError("Request fingerprint must be a SHA-256 hex string")
        if len({entry.artifact_id for entry in self.entries}) != len(self.entries):
            raise ValueError("Artifact IDs must be unique")
        if len({entry.path for entry in self.entries}) != len(self.entries):
            raise ValueError("Artifact paths must be unique")
        if any(entry.path == "manifest.json" for entry in self.entries):
            raise ValueError("A manifest cannot hash itself")


@dataclass(frozen=True)
class Event(Model):
    """Sequenced lifecycle event for one stage of a job."""

    report_type: ClassVar[str] = "event"
    schema_version: str = SCHEMA_VERSION
    sequence: int = 0
    job_id: str = ""
    stage: Stage = Stage.ANALYSIS
    kind: EventKind = EventKind.START
    timestamp: str = field(default_factory=utc_now)
    stage_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the event schema, timestamp, and sequence number."""
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version: {self.schema_version}")
        _validate_timestamp(self.timestamp)
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("Event sequence must be a nonnegative integer")


@dataclass(frozen=True)
class CompilationResult:
    """Successful compiled module paired with its evidence report."""

    module: Any
    report: CompilationReport


class CompilationFailed(RuntimeError):
    """Compilation exception that retains the structured failure report."""

    def __init__(self, report: CompilationReport) -> None:
        """Expose the failed compilation report through an exception."""
        self.report = report
        super().__init__(
            report.failure.message if report.failure else "Compilation failed"
        )


def _validate_timestamp(value: str) -> None:
    """Require a real UTC ISO-8601 timestamp ending in ``Z``."""
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", value
    ):
        raise ValueError("Timestamps must use UTC ISO-8601 with a Z suffix")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("Invalid UTC timestamp") from exc


REPORT_TYPES = {
    cls.report_type: cls
    for cls in (
        BaseReport,
        EnvironmentReport,
        AnalysisReport,
        CompilationReport,
        ValidationReport,
        BenchmarkReport,
        ArtifactManifest,
        Event,
    )
}
