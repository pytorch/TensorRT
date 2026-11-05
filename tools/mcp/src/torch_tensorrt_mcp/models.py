"""Strict, JSON-only requests. Compiler path/object settings are never exposed."""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

JobId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
FactoryName = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class CompilationOptions(StrictModel):
    min_block_size: Annotated[int, Field(ge=1, le=100000)] = 5
    require_full_compilation: bool = False
    disable_tf32: bool = False
    use_fp32_acc: bool = False
    optimization_level: Annotated[int, Field(ge=0, le=5)] | None = None
    workspace_size: Annotated[int, Field(ge=0, le=64 * 1024**3)] = 0
    version_compatible: bool = False
    hardware_compatible: bool = False
    truncate_double: bool = False
    use_fast_partitioner: bool = True
    enable_experimental_decompositions: bool = False
    assume_dynamic_shape_support: bool = False
    sparse_weights: bool = False
    num_avg_timing_iters: Annotated[int, Field(ge=1, le=100)] = 1
    max_aux_streams: Annotated[int, Field(ge=0, le=1024)] | None = None
    offload_module_to_cpu: bool = False
    decompose_attention: bool = False
    fallback_data_dependent_ops: bool = False


class ExecutionRequest(StrictModel):
    timeout_seconds: Annotated[float, Field(gt=0, le=86400)] = 600.0


class ModelRequest(ExecutionRequest):
    model: FactoryName
    inputs: FactoryName
    model_parameters: dict[str, Any] = Field(default_factory=dict)
    input_parameters: dict[str, Any] = Field(default_factory=dict)
    seed: Annotated[int, Field(ge=0, lt=2**64)] = 0
    settings: CompilationOptions = Field(default_factory=CompilationOptions)

    @field_validator("model_parameters", "input_parameters")
    @classmethod
    def json_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        def check(item: Any, depth: int = 0) -> None:
            if depth > 16:
                raise ValueError("factory parameters exceed maximum JSON depth")
            if item is None or type(item) in (str, bool, int):
                return
            if type(item) is float and math.isfinite(item):
                return
            if type(item) is list:
                for child in item:
                    check(child, depth + 1)
                return
            if type(item) is dict and all(type(key) is str for key in item):
                for child in item.values():
                    check(child, depth + 1)
                return
            raise ValueError("factory parameters must contain only finite JSON values")

        check(value)
        if len(json.dumps(value).encode("utf-8")) > 65536:
            raise ValueError("factory parameters exceed 64 KiB")
        if "seed" in value:
            raise ValueError("seed is a reserved factory argument")
        return value


class EnvironmentRequest(ExecutionRequest):
    pass


class ValidationRequest(ExecutionRequest):
    source_job_id: JobId
    atol: Annotated[float, Field(ge=0)] = 1e-5
    rtol: Annotated[float, Field(ge=0)] = 1e-5
    relative_floor: Annotated[float, Field(gt=0)] = 1e-12


class BenchmarkRequest(ExecutionRequest):
    source_job_id: JobId
    sample_index: Annotated[int, Field(ge=0, le=1000000)] = 0
    warmup: Annotated[int, Field(ge=0, le=10000)] = 10
    iterations: Annotated[int, Field(ge=1, le=100000)] = 100
    percentiles: list[Annotated[float, Field(ge=0, le=100)]] = Field(
        default_factory=lambda: [50.0, 90.0, 99.0]
    )
    batch_size: Annotated[int, Field(ge=1, le=1000000)] | None = None

    @field_validator("percentiles")
    @classmethod
    def bounded_percentiles(cls, values: list[float]) -> list[float]:
        if not values or len(values) > 20 or len(set(values)) != len(values):
            raise ValueError("provide 1–20 unique percentiles")
        return values


class JobRequest(StrictModel):
    job_id: JobId


class ReportRequest(JobRequest):
    report: Literal[
        "environment", "analysis", "compilation", "validation", "benchmark", "failure"
    ]


REQUEST_TYPES = {
    "collect_environment": EnvironmentRequest,
    "analyze": ModelRequest,
    "compile": ModelRequest,
    "validate": ValidationRequest,
    "benchmark": BenchmarkRequest,
}
