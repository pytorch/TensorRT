# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Report collection shared by preflight and the AOT compiler.

The compiler owns support decisions; this module snapshots those decisions before
engine conversion. Public report APIs serialize access to compiler global state.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import hashlib
import inspect
import logging
import math
import operator
import random
import sys
import threading
import time
from contextvars import ContextVar
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, cast

import torch
from torch_tensorrt.reporting import (
    AnalysisReport,
    Check,
    CheckStatus,
    CompilationFailed,
    CompilationReport,
    CompilationResult,
    Confidence,
    Coverage,
    Diagnostic,
    EngineReport,
    EventKind,
    InvalidInputError,
    NodeAssignment,
    NodeDecision,
    NodeKind,
    Partition,
    Stage,
    classify_exception,
    emit_event,
    event_context,
    status_for_failure,
)

_COMPILER_LOCK = threading.RLock()
_CONTEXT: ContextVar[ReportCollector | None] = ContextVar(
    "torchtrt_report_collector", default=None
)
_SOURCE_ID = "_torchtrt_report_node_id"
_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class ReportCollector:
    job_id: str
    analyzing: bool = False
    diagnostics_dir: Path | None = None
    analysis: AnalysisReport | None = None
    engines: list[EngineReport] = dataclasses.field(default_factory=list)
    stage: Stage = Stage.COMPILATION
    active_engine: str = ""
    cache_hits: set[str] = dataclasses.field(default_factory=set)
    engine_metadata: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    stage_started_at: dict[tuple[Stage, str], float] = dataclasses.field(
        default_factory=dict
    )


def current_collector() -> ReportCollector | None:
    return _CONTEXT.get()


def compilation_lock() -> Any:
    return _COMPILER_LOCK


def with_compiler_lock(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Run a compiler entrypoint while holding the shared reentrant lock."""

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with _COMPILER_LOCK:
            return fn(*args, **kwargs)

    return wrapped


def is_analyzing() -> bool:
    collector = current_collector()
    return collector is not None and collector.analyzing


@contextlib.contextmanager
def _preserve_rng_state() -> Iterator[None]:
    """Preserve Torch, Python, and already-loaded NumPy random state."""
    python_state = random.getstate()
    numpy = cast(Any, sys.modules.get("numpy"))
    numpy_state = numpy.random.get_state() if numpy is not None else None
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)


def stage_start(stage: Stage, *, stage_id: str = "") -> None:
    collector = current_collector()
    if collector is not None:
        collector.stage = stage
        stage_id = stage_id or collector.active_engine
    emit_event(stage, EventKind.START, stage_id=stage_id)
    if collector is not None:
        # Start after dispatching START so a slow event sink is not stage work.
        collector.stage_started_at[(stage, stage_id)] = time.perf_counter()


def stage_end(stage: Stage, *, stage_id: str = "", **details: Any) -> None:
    collector = current_collector()
    if collector is not None:
        stage_id = stage_id or collector.active_engine
        started_at = collector.stage_started_at.pop((stage, stage_id), None)
        if started_at is not None:
            details.setdefault("elapsed_seconds", time.perf_counter() - started_at)
    emit_event(stage, EventKind.END, stage_id=stage_id, **details)


def record_partition_inputs(partition_id: str, inputs: Any) -> None:
    collector = current_collector()
    if collector is not None and collector.analysis is not None:
        collector.analysis = dataclasses.replace(
            collector.analysis,
            partitions=[
                (
                    dataclasses.replace(
                        p, input_profiles=[json_value(i) for i in inputs]
                    )
                    if p.partition_id == partition_id
                    else p
                )
                for p in collector.analysis.partitions
            ],
        )


def record_engine_metadata(name: str, result: Any) -> None:
    collector = current_collector()
    if collector is not None:
        collector.engine_metadata[name] = {
            "serialized_size_bytes": len(result.serialized_engine),
            "serialized_sha256": hashlib.sha256(result.serialized_engine).hexdigest(),
            "input_binding_names": list(result.input_names),
            "output_binding_names": list(result.output_names),
            "requires_output_allocator": result.requires_output_allocator,
            "requires_native_multidevice": result.requires_native_multidevice,
            "aliased_io": json_value(result.aliased_io),
            "symbolic_shape_expressions": json_value(result.symbolic_shape_expressions),
        }


def record_cache_hit() -> None:
    collector = current_collector()
    if collector is not None:
        collector.cache_hits.add(collector.active_engine)


def json_value(value: Any) -> Any:
    """Serialize settings without invoking arbitrary object repr or callbacks."""
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Report settings must contain finite numbers")
        return value
    if isinstance(value, Enum):
        return json_value(value.value)
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Report dictionaries must have string keys")
        return {key: json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (set, frozenset)):
        return sorted((json_value(v) for v in value), key=str)
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if dataclasses.is_dataclass(value):
        return {
            f.name: json_value(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    # Device/Input and torch operator objects have well-defined public fields.
    if hasattr(value, "gpu_id"):
        return {
            name: json_value(getattr(value, name))
            for name in ("gpu_id", "device_type", "dla_core", "allow_gpu_fallback")
            if hasattr(value, name)
        }
    if hasattr(value, "shape_mode"):
        return {
            name: json_value(getattr(value, name))
            for name in (
                "shape",
                "dtype",
                "shape_mode",
                "profiles",
                "shared_dims",
                "is_shape_tensor",
                "format",
                "name",
            )
            if hasattr(value, name)
        }
    if isinstance(value, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
        return str(value)
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def tag_source_nodes(gm: torch.fx.GraphModule) -> None:
    for node in gm.graph.nodes:
        node.meta.setdefault(_SOURCE_ID, node.name)


def _node_kind(
    node: torch.fx.Node, parameters: set[str], buffers: set[str]
) -> NodeKind:
    if node.op == "get_attr":
        if node.target in parameters:
            return NodeKind.PARAMETER
        if node.target in buffers:
            return NodeKind.BUFFER
        return NodeKind.STRUCTURAL
    if node.op not in ("call_function", "call_method", "call_module"):
        return NodeKind.STRUCTURAL
    name = str(node.target)
    if node.target in (operator.getitem, getattr) or name.startswith(
        (
            "aten.sym_size",
            "aten.sym_stride",
            "aten.sym_numel",
            "aten.sym_storage_offset",
            "aten._assert",
            "aten._check",
        )
    ):
        return NodeKind.STRUCTURAL
    # Scalar shape arithmetic is structural. Tensor-valued arithmetic counts.
    if "val" in node.meta and isinstance(
        node.meta["val"], (torch.SymInt, torch.SymFloat, torch.SymBool, int, bool)
    ):
        return NodeKind.STRUCTURAL
    return NodeKind.COMPUTATIONAL


def partition_analysis(
    gm: torch.fx.GraphModule,
    partitioned: torch.fx.GraphModule,
    support: Any,
    settings: Any,
    *,
    whole_graph: bool = False,
    skipped: bool = False,
) -> AnalysisReport | None:
    """Snapshot support and final assignment from the shared compiler pipeline."""
    collector = current_collector()
    if collector is None and not settings.dryrun:
        return None
    if collector is not None and collector.diagnostics_dir is not None:
        (collector.diagnostics_dir / "lowered_graph.txt").write_text(
            str(gm.graph), encoding="utf-8"
        )
        (collector.diagnostics_dir / "partitioned_graph.txt").write_text(
            str(partitioned.graph), encoding="utf-8"
        )
    parameters = set(dict(gm.named_parameters()))
    buffers = set(dict(gm.named_buffers()))
    assignments: dict[str, str] = {}
    assignment_reasons: dict[str, str] = {}
    partitions: list[Partition] = []
    if whole_graph:
        node_ids = [
            n.meta.get(_SOURCE_ID, n.name)
            for n in gm.graph.nodes
            if n.op in ("call_function", "call_method", "call_module")
        ]
        assignments = dict.fromkeys(node_ids, "_run_on_acc_0")
        partitions.append(Partition("_run_on_acc_0", node_ids))
    elif not skipped:
        for module in partitioned.modules():
            if isinstance(module, torch.fx.GraphModule):
                for node in module.graph.nodes:
                    reason = node.meta.get("_torchtrt_partition_fallback_reason")
                    if reason:
                        assignment_reasons[node.meta.get(_SOURCE_ID, node.name)] = (
                            reason
                        )
        for parent_node in partitioned.graph.nodes:
            if parent_node.op != "call_module":
                continue
            child = partitioned.get_submodule(str(parent_node.target))
            if not isinstance(child, torch.fx.GraphModule):
                continue
            node_ids = [
                n.meta.get(_SOURCE_ID, n.name)
                for n in child.graph.nodes
                if n.op in ("call_function", "call_method", "call_module")
            ]
            is_trt = "_run_on_acc" in str(parent_node.target)
            if is_trt:
                assignments.update(
                    {node_id: str(parent_node.target) for node_id in node_ids}
                )
            partitions.append(
                Partition(
                    str(parent_node.target),
                    node_ids,
                    kind="tensorrt" if is_trt else "pytorch",
                )
            )

    decisions: list[NodeDecision] = []
    node_support = getattr(support, "node_support", {})
    node_reasons = getattr(support, "node_reasons", {})
    module_dict = dict(gm.named_modules())
    for node in gm.graph.nodes:
        kind = _node_kind(node, parameters, buffers)
        node_id = node.meta.get(_SOURCE_ID, node.name)
        reasons = sorted(node_reasons.get(node.name, ()))
        computational = kind == NodeKind.COMPUTATIONAL
        eligible = bool(node_support.get(node.name, False)) if computational else False
        if computational and node.name not in node_support:
            eligible = bool(support.is_node_supported(module_dict, node))
            reasons = sorted(getattr(support, "node_reasons", {}).get(node.name, ()))
        assigned = node_id in assignments and eligible
        if eligible and not assigned:
            reasons.append(
                "below_min_block_size"
                if skipped
                else assignment_reasons.get(node_id, "not_assigned_by_partitioner")
            )
        decisions.append(
            NodeDecision(
                node_id=node_id,
                operator=str(node.target),
                kind=kind,
                eligible=eligible,
                assignment=(
                    NodeAssignment.TENSORRT
                    if assigned
                    else (
                        NodeAssignment.PYTORCH
                        if computational
                        else NodeAssignment.SKIPPED
                    )
                ),
                reason_codes=reasons,
                confidence=(
                    Confidence.CONFIRMED if "val" in node.meta else Confidence.PREDICTED
                ),
                source_node_ids=sorted(
                    {
                        getattr(source, "meta", {}).get(_SOURCE_ID, source.name)
                        for source in node.meta.get("from_node", ())
                        if isinstance(getattr(source, "name", None), str)
                    }
                ),
            )
        )

    computational_nodes = [n for n in decisions if n.kind == NodeKind.COMPUTATIONAL]

    def excluded(node: NodeDecision) -> bool:
        return any(reason.startswith("excluded by") for reason in node.reason_codes)

    coverage = Coverage(
        total=len(computational_nodes),
        eligible=sum(n.eligible for n in computational_nodes),
        assigned=sum(
            n.assignment == NodeAssignment.TENSORRT for n in computational_nodes
        ),
        structural=sum(n.kind == NodeKind.STRUCTURAL for n in decisions),
        parameters=sum(
            n.kind in (NodeKind.PARAMETER, NodeKind.BUFFER) for n in decisions
        ),
        excluded=sum(excluded(n) for n in computational_nodes),
        unsupported=sum(
            not n.eligible and not excluded(n) for n in computational_nodes
        ),
        below_min_block_size=sum(
            "below_min_block_size" in n.reason_codes for n in computational_nodes
        ),
    )
    satisfiable = (
        not settings.require_full_compilation or coverage.assigned == coverage.total
    )
    report = AnalysisReport(
        job_id=collector.job_id if collector else "",
        settings=json_value(settings),
        nodes=decisions,
        partitions=partitions,
        coverage=coverage,
        checks=[
            Check(
                "converter_validation",
                CheckStatus.PASSED,
                "Eligibility is confirmed only for lowered nodes and their recorded metadata.",
            ),
            Check(
                "requested_settings",
                CheckStatus.PASSED if satisfiable else CheckStatus.FAILED,
                "Final computational node assignment under requested min_block_size and full-compilation policy.",
            ),
            Check(
                "engine_build",
                CheckStatus.SKIPPED,
                "Partition eligibility does not prove that TensorRT can build an engine.",
            ),
        ],
        model_execution_performed=True,
        requested_settings_satisfiable=satisfiable,
    )
    if collector is not None:
        collector.analysis = report
    stage_end(
        Stage.PARTITIONING,
        eligible=coverage.eligible,
        assigned=coverage.assigned,
        total=coverage.total,
    )
    return report


def build_engine_with_reporting(
    builder: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    collector = current_collector()
    if collector is None:
        return builder(*args, **kwargs)
    name = kwargs.get("name", "_run_on_acc_0")
    collector.active_engine = name
    stage_start(Stage.CONVERSION)
    start = time.perf_counter()
    try:
        module = builder(*args, **kwargs)
    except (Exception, KeyboardInterrupt) as exc:
        failure = classify_exception(exc, collector.stage)
        collector.engines.append(
            EngineReport(
                name,
                name,
                status=status_for_failure(failure),
                elapsed_seconds=time.perf_counter() - start,
            )
        )
        raise
    elapsed_seconds = time.perf_counter() - start
    cache_hit = name in collector.cache_hits
    collector.engines.append(
        EngineReport(
            name,
            name,
            elapsed_seconds=elapsed_seconds,
            cache_hit=cache_hit,
            metadata={
                "runtime": f"{type(module).__module__}.{type(module).__qualname__}",
                **collector.engine_metadata.get(name, {}),
            },
        )
    )
    stage_end(
        collector.stage,
        stage_id=name,
        cache_hit=cache_hit,
    )
    collector.stage = Stage.COMPILATION
    collector.active_engine = ""
    return module


@contextlib.contextmanager
def _diagnostics(directory: str | Path | None) -> Iterator[None]:
    if directory is None:
        yield
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(directory / "compiler.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger = logging.getLogger("torch_tensorrt")
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        handler.close()


def _run(
    exported_program: torch.export.ExportedProgram,
    inputs: Any,
    arg_inputs: Any,
    kwarg_inputs: Any,
    job_id: str,
    event_sink: Any,
    diagnostics_dir: str | Path | None,
    analyzing: bool,
    settings: dict[str, Any],
) -> CompilationResult | AnalysisReport:
    from . import _compiler
    from .conversion._ConverterRegistry import DYNAMO_CONVERTERS

    collector = ReportCollector(
        job_id,
        analyzing,
        diagnostics_dir=Path(diagnostics_dir) if diagnostics_dir is not None else None,
    )
    effective_settings: dict[str, Any] = {}
    with _COMPILER_LOCK, event_context(job_id, event_sink) as emitter:
        token = _CONTEXT.set(collector)
        previous_settings = DYNAMO_CONVERTERS.compilation_settings
        previous_disallowed = DYNAMO_CONVERTERS.disallowed_targets
        diagnostic_context = _diagnostics(diagnostics_dir)
        diagnostic_opened = False
        try:
            collector.stage = (
                Stage.SAVE if diagnostics_dir is not None else Stage.COMPILATION
            )
            diagnostic_context.__enter__()
            diagnostic_opened = True
            stage_start(Stage.ANALYSIS if analyzing else Stage.COMPILATION)
            if not isinstance(exported_program, torch.export.ExportedProgram):
                raise InvalidInputError(
                    "Report APIs require a torch.export.ExportedProgram"
                )
            if settings.get("dryrun") and not analyzing:
                raise InvalidInputError(
                    "compile_with_report builds engines; use analyze for preflight"
                )
            allowed = set(inspect.signature(_compiler.compile).parameters) - {
                "kwargs",
                "inputs",
                "exported_program",
                "arg_inputs",
                "kwarg_inputs",
            }
            unknown = set(settings) - allowed
            if unknown:
                raise InvalidInputError(
                    f"Unknown compilation settings: {', '.join(sorted(unknown))}"
                )
            if (
                settings.get("enable_autocast")
                or settings.get("autocast_calibration_dataloader") is not None
            ):
                raise InvalidInputError(
                    "The current report APIs do not support execution of autocast calibration dataloaders"
                )
            try:
                json_value(settings)
            except ValueError as exc:
                raise InvalidInputError(str(exc)) from exc
            effective_settings = dict(settings)
            if analyzing:
                effective_settings.update(
                    dryrun=True,
                    cache_built_engines=False,
                    reuse_cached_engines=False,
                    custom_engine_cache=None,
                    offload_module_to_cpu=False,
                )
            # Copy graph and weights together so constant folding/partitioning cannot
            # invalidate the caller's ExportedProgram or reference model.
            with _preserve_rng_state():
                isolated = copy.deepcopy(exported_program)
                if diagnostics_dir is not None:
                    (Path(diagnostics_dir) / "exported_graph.txt").write_text(
                        str(isolated.graph_module.graph), encoding="utf-8"
                    )
                module = _compiler.compile(
                    isolated,
                    inputs=inputs,
                    arg_inputs=arg_inputs,
                    kwarg_inputs=kwarg_inputs,
                    **effective_settings,
                )
            # Close within the typed failure boundary, before announcing success.
            collector.stage = Stage.SAVE
            diagnostic_opened = False
            diagnostic_context.__exit__(None, None, None)
            collector.stage = Stage.ANALYSIS if analyzing else Stage.COMPILATION
            if collector.analysis is None:
                raise RuntimeError("Compiler completed without a partition analysis")
            analysis = collector.analysis
            if analyzing:
                # dryrun is an implementation detail; the report describes requested
                # settings and candidly marks engine-build checks as skipped.
                analysis_settings = dict(analysis.settings)
                analysis_settings.update(json_value(settings))
                analysis_settings["dryrun"] = False
                analysis = dataclasses.replace(analysis, settings=analysis_settings)
                stage_end(Stage.ANALYSIS)
                analysis = dataclasses.replace(
                    analysis,
                    diagnostics=[
                        Diagnostic("event_callback_failed", name)
                        for name in emitter.callback_failures
                    ],
                )
                return analysis
            stage_end(Stage.COMPILATION, engine_count=len(collector.engines))
            diagnostics = [
                Diagnostic("event_callback_failed", name)
                for name in emitter.callback_failures
            ]
            report = CompilationReport(
                job_id=job_id,
                analysis=analysis,
                settings=analysis.settings,
                engines=collector.engines,
                diagnostics=diagnostics,
            )
            return CompilationResult(module=module, report=report)
        except (Exception, KeyboardInterrupt) as exc:
            failure = classify_exception(exc, collector.stage)
            status = status_for_failure(failure)
            # A failed durable sink cannot be used to reliably record its own
            # failure, but it must not hide the typed report from the caller.
            with contextlib.suppress(Exception):
                emit_event(
                    collector.stage,
                    EventKind.FAILURE,
                    stage_id=collector.active_engine,
                    category=failure.category.value,
                )
            if analyzing:
                return (
                    dataclasses.replace(
                        collector.analysis, status=status, failure=failure
                    )
                    if collector.analysis
                    else AnalysisReport(
                        job_id=job_id,
                        status=status,
                        failure=failure,
                        settings=json_value(effective_settings),
                    )
                )
            raise CompilationFailed(
                CompilationReport(
                    job_id=job_id,
                    status=status,
                    failure=failure,
                    analysis=collector.analysis,
                    settings=json_value(effective_settings),
                    engines=collector.engines,
                )
            ) from exc
        finally:
            try:
                if diagnostic_opened:
                    diagnostic_context.__exit__(None, None, None)
            finally:
                DYNAMO_CONVERTERS.compilation_settings = previous_settings
                DYNAMO_CONVERTERS.set_disallowed_targets(previous_disallowed)
                _CONTEXT.reset(token)


def analyze(
    exported_program: torch.export.ExportedProgram,
    inputs: Any = None,
    *,
    arg_inputs: Any = None,
    kwarg_inputs: Any = None,
    job_id: str = "",
    event_sink: Any = None,
    diagnostics_dir: str | Path | None = None,
    **settings: Any,
) -> AnalysisReport:
    """Inspect lowered support/partitions without building engines.

    Graph/weights and RNG are isolated from the caller. Lowering may execute tensor
    operations on this private copy. CUDA/TensorRT are still required for compiler
    checks; a failed check returns a categorized AnalysisReport. Autocast calibration
    is outside the current ownership contract. Report APIs serialize compiler access.
    """
    return cast(
        AnalysisReport,
        _run(
            exported_program,
            inputs,
            arg_inputs,
            kwarg_inputs,
            job_id,
            event_sink,
            diagnostics_dir,
            True,
            settings,
        ),
    )


def compile_with_report(
    exported_program: torch.export.ExportedProgram,
    inputs: Any = None,
    *,
    arg_inputs: Any = None,
    kwarg_inputs: Any = None,
    job_id: str = "",
    event_sink: Any = None,
    diagnostics_dir: str | Path | None = None,
    **settings: Any,
) -> CompilationResult:
    """Compile an isolated ExportedProgram; failures carry a CompilationReport."""
    return cast(
        CompilationResult,
        _run(
            exported_program,
            inputs,
            arg_inputs,
            kwarg_inputs,
            job_id,
            event_sink,
            diagnostics_dir,
            False,
            settings,
        ),
    )
