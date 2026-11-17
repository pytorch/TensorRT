"""Apply Torch-TRT decompositions during export so compile can skip a second AOT retrace.

Stock ``torch.export`` captures without Torch-TRT's decomp table;
``dynamo.compile`` then calls ``ExportedProgram.run_decompositions``, which
re-traces. ``dynamo.trace`` uses this helper so the default
``torch_tensorrt.compile(nn.Module)`` path is one trace. ``compile`` skips
``run_decompositions`` when the exported program already carries a matching
table fingerprint — no user-facing flag.

``pre_export_lowering`` runs between the two, so ``trace`` leaves out of its table
the ops that those passes declare they must see undecomposed (see
``passes._undecomposed_ops``), plus the decompositions whose use depends on graph
shapes. ``compile`` decomposes them after the passes have run. A pass that needs
the whole graph undecomposed, such as autocast, makes ``trace`` export without
Torch-TRT decompositions, and ``compile`` decomposes once after lowering.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, Generator, Iterator, Mapping, Optional, cast

import torch
from torch.export import ExportedProgram
from torch.export._trace import _export
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering._decomposition_filter import (
    SHAPE_DEPENDENT_DECOMPOSITIONS,
    filter_decomposition_table,
)
from torch_tensorrt.dynamo.lowering._decompositions import get_decompositions
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
    pre_export_lowering_undecomposed_ops,
)
from torch_tensorrt.dynamo.lowering.passes._undecomposed_ops import ALL_OPS

logger = logging.getLogger(__name__)

_STAMP_KEY = "torch_tensorrt_decomp_fingerprint"
_STAMP_VERSION = 1
# Stamp entry naming the ops whose decomposition export left for compile.
_DEFERRED_KEY = "deferred"


def settings_from_kwargs(kwargs: Mapping[str, Any]) -> CompilationSettings:
    """The settings that the compile keyword arguments in ``kwargs`` give.

    ``dynamo.trace`` takes the same keyword arguments as ``compile``, and ignores
    the ones that are not settings.
    """
    names = {field.name for field in dataclasses.fields(CompilationSettings)}
    return CompilationSettings(**{k: v for k, v in kwargs.items() if k in names})


def decomp_fingerprint(
    settings: CompilationSettings = CompilationSettings(),
) -> dict[str, Any]:
    """Stable id for the decomp table ``compile`` would apply under ``settings``."""
    return {
        "v": _STAMP_VERSION,
        "experimental": bool(settings.enable_experimental_decompositions),
        "attention": bool(settings.decompose_attention),
        "distributed": bool(settings.use_distributed_mode_trace),
        "fp32_acc": bool(settings.use_fp32_acc),
    }


def stamp_exported_program(
    exported_program: ExportedProgram,
    fingerprint: dict[str, Any],
    deferred: tuple[str, ...] = (),
) -> ExportedProgram:
    """Record the decomp table applied to ``exported_program``.

    ``deferred`` names ops of that table whose decomposition was left for
    ``compile``: ones a pre-export pass must see, and ones whose use depends on
    shapes only known after tracing.
    """
    stamp = {**fingerprint, _DEFERRED_KEY: tuple(deferred)}
    exported_program.graph_module.meta[_STAMP_KEY] = dict(stamp)
    graph_meta = getattr(exported_program.graph_module.graph, "meta", None)
    if isinstance(graph_meta, dict):
        graph_meta[_STAMP_KEY] = dict(stamp)
    return exported_program


def _read_stamp(exported_program: ExportedProgram) -> Optional[dict[str, Any]]:
    for meta in (
        exported_program.graph_module.meta,
        getattr(exported_program.graph_module.graph, "meta", None),
    ):
        if isinstance(meta, dict) and isinstance(meta.get(_STAMP_KEY), dict):
            return cast(dict[str, Any], meta[_STAMP_KEY])
    return None


def _decomp_table(settings: CompilationSettings) -> dict[Any, Callable[..., Any]]:
    table = get_decompositions(
        settings.enable_experimental_decompositions,
        settings.decompose_attention,
        settings.use_distributed_mode_trace,
        use_fp32_acc=settings.use_fp32_acc,
    )
    return cast(dict[Any, Callable[..., Any]], table)


def _deferred(exported_program: ExportedProgram) -> set[str]:
    stamp = _read_stamp(exported_program)
    return set(stamp[_DEFERRED_KEY]) if stamp is not None else set()


def matching_decomp_stamp(
    exported_program: ExportedProgram, fingerprint: dict[str, Any]
) -> bool:
    """Whether ``exported_program`` was decomposed with the table ``fingerprint`` names."""
    stamp = _read_stamp(exported_program)
    if stamp is None:
        return False
    return {k: v for k, v in stamp.items() if k != _DEFERRED_KEY} == fingerprint


def _graph_targets(graph_module: torch.fx.GraphModule) -> set[Any]:
    # Includes subgraphs, such as the branches of torch.cond.
    return {
        node.target
        for module in graph_module.modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
        if node.op == "call_function"
    }


def _has_pending_deferred_ops(
    exported_program: ExportedProgram, table: Mapping[Any, Any]
) -> bool:
    """Whether the graph has an op that export deferred and ``table`` still decomposes."""
    deferred = _deferred(exported_program)
    if not deferred:
        return False
    return any(
        str(target) in deferred and target in table
        for target in _graph_targets(exported_program.graph_module)
    )


def _cia_split_apis() -> Optional[tuple[Callable[..., Any], Callable[..., Any]]]:
    try:
        from torch.export.exported_program import (
            _override_composite_implicit_decomp,
            _split_decomp_table_to_cia_and_python_decomp,
        )
    except ImportError:
        return None
    return (
        _override_composite_implicit_decomp,
        _split_decomp_table_to_cia_and_python_decomp,
    )


@contextmanager
def _aot_export_with_cia_split() -> Iterator[None]:
    """When AOT sees ``decompositions=table``, CIA-split like ``run_decompositions``."""
    import torch.export._trace as export_trace

    apis = _cia_split_apis()
    if apis is None:
        yield
        return
    override_composite_implicit_decomp, split_decomp_table_to_cia_and_python_decomp = (
        apis
    )
    original = export_trace._aot_export_joint_with_descriptors

    def _wrapped(stack: Any, mod: Any, fake_args: Any, **flags: Any) -> Any:
        table = flags.get("decompositions")
        if table is not None:
            # run_decompositions mutates the table via del; always copy.
            cia_to_decomp, python_decomp_table = (
                split_decomp_table_to_cia_and_python_decomp(dict(table))
            )
            logger.debug(
                "export_with_tensorrt_decomps: CIA-split decomp_table (cia=%d, python=%d)",
                len(cia_to_decomp),
                len(python_decomp_table),
            )
            stack.enter_context(override_composite_implicit_decomp(cia_to_decomp))
            flags["decompositions"] = python_decomp_table
        return original(stack, mod, fake_args, **flags)

    export_trace._aot_export_joint_with_descriptors = _wrapped
    try:
        yield
    finally:
        export_trace._aot_export_joint_with_descriptors = original


@contextmanager
def _export_to_aten_ir_with_decomp_table(
    decomp_table: dict[Any, Callable[..., Any]],
) -> Generator[None, None, None]:
    """Thread ``decomp_table`` into ``_export_to_aten_ir`` (public ``export`` omits it)."""
    import torch.export._trace as export_trace

    if not hasattr(export_trace, "_export_to_aten_ir"):
        yield
        return
    original = export_trace._export_to_aten_ir

    @functools.wraps(original)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        kwargs["decomp_table"] = decomp_table
        return original(*args, **kwargs)

    export_trace._export_to_aten_ir = _wrapped
    try:
        yield
    finally:
        export_trace._export_to_aten_ir = original


def maybe_run_decompositions(
    exported_program: ExportedProgram, settings: CompilationSettings
) -> ExportedProgram:
    """Run ``run_decompositions`` unless ``trace`` already applied the same table.

    Call after ``pre_export_lowering``. Runs anyway when a pre-export pass needs the
    graph undecomposed (autocast), since it may have inserted ops after ``trace``,
    and when the graph has an op whose decomposition ``trace`` deferred and
    ``filter_decomposition_table`` keeps for this graph.
    """
    fingerprint = decomp_fingerprint(settings)
    table = filter_decomposition_table(
        _decomp_table(settings), exported_program.graph_module
    )
    needed = pre_export_lowering_undecomposed_ops(settings)
    if needed == ALL_OPS:
        logger.info(
            "Running run_decompositions: a pre-export lowering pass needs the graph "
            "undecomposed and may have inserted ops after dynamo.trace"
        )
    elif matching_decomp_stamp(exported_program, fingerprint):
        decomposed_early = sorted(
            str(op)
            for op in needed
            if op in table and str(op) not in _deferred(exported_program)
        )
        if decomposed_early:
            logger.warning(
                "dynamo.trace decomposed %s, which pre-export lowering passes need to "
                "see under these settings, so those passes may not apply. Pass the "
                "same settings to dynamo.trace as to compile.",
                ", ".join(decomposed_early),
            )
        if not _has_pending_deferred_ops(exported_program, table):
            logger.info(
                "Skipping run_decompositions: exported program already has the matching "
                "Torch-TRT decomp table from dynamo.trace"
            )
            return exported_program
        logger.info(
            "Running run_decompositions: the graph has ops whose decomposition "
            "dynamo.trace deferred until after pre-export lowering"
        )

    exported_program = exported_program.run_decompositions(table)
    return stamp_exported_program(exported_program, fingerprint)


def export_with_tensorrt_decomps(
    mod: torch.nn.Module,
    args: tuple[Any, ...] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
    *,
    strict: bool = False,
    dynamic_shapes: Any = None,
    prefer_deferred_runtime_asserts_over_guards: bool = False,
    settings: CompilationSettings = CompilationSettings(),
) -> ExportedProgram:
    """Capture ``mod`` and apply Torch-TRT decompositions in a single AOT trace.

    Exports with stock ``torch.export`` (no stamp) instead when a pre-export
    lowering pass needs the whole graph undecomposed under ``settings``, or when
    this PyTorch build cannot thread a decomp table through export.
    """
    export_kwargs = dict(kwargs or {})
    needed = pre_export_lowering_undecomposed_ops(settings)
    can_inject = (
        _cia_split_apis() is not None
        and hasattr(torch.export._trace, "_export_to_aten_ir")
        and hasattr(torch.export._trace, "_aot_export_joint_with_descriptors")
    )
    if needed == ALL_OPS or not can_inject:
        if needed == ALL_OPS:
            logger.info(
                "Exporting without Torch-TRT decomps: a pre-export lowering pass "
                "needs the graph undecomposed; compile() will run_decompositions."
            )
        else:
            logger.warning(
                "Cannot apply Torch-TRT decomps during export on this PyTorch build; "
                "falling back to torch.export. compile() will run_decompositions."
            )
        # As MutableTorchTensorRTModule exported before one-shot decomposition.
        if prefer_deferred_runtime_asserts_over_guards:
            return _export(
                mod,
                args=args,
                kwargs=export_kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=strict,
                prefer_deferred_runtime_asserts_over_guards=True,
            )
        return torch.export.export(
            mod,
            args=args,
            kwargs=export_kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
        )

    table = _decomp_table(settings)
    # Left for compile: the ops pre-export passes must match, and decompositions
    # whose use depends on graph shapes (one that loops over a symbolic extent
    # would specialize it and fail export), which compile filters once it has the
    # traced graph.
    to_defer = {*SHAPE_DEPENDENT_DECOMPOSITIONS, *needed}
    deferred = tuple(sorted(str(op) for op in to_defer if op in table))
    for op in to_defer:
        table.pop(op, None)
    logger.info(
        "export_with_tensorrt_decomps: exporting with decomp_table (%d entries)",
        len(table),
    )
    with _aot_export_with_cia_split():
        with _export_to_aten_ir_with_decomp_table(table):
            exported_program = _export(
                mod,
                args=args,
                kwargs=export_kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=strict,
                prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
            )
    return stamp_exported_program(
        exported_program, decomp_fingerprint(settings), deferred
    )
