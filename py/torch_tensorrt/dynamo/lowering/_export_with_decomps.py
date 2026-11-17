"""Apply Torch-TRT decompositions during export so compile can skip a second AOT retrace.

Stock ``torch.export`` captures without Torch-TRT's decomp table;
``dynamo.compile`` then calls ``ExportedProgram.run_decompositions``, which
re-traces the whole program. ``dynamo.trace`` uses this helper so the default
``torch_tensorrt.compile(nn.Module)`` path is one trace.

``compile`` decides from the graph itself whether decomposition is still needed: it
runs ``run_decompositions`` unless the program is already post-dispatch ATen IR with
no op that its table would rewrite. That holds whatever produced the program and
whatever settings ``trace`` was given. An op that a table already tried and left in
place, such as ``_fused_rms_norm`` on CUDA, where its decomposition never applies,
is recorded on the program and not counted, since rerunning would not remove it.

``pre_export_lowering`` runs between the two. If any of its passes needs the graph
before Torch-TRT decompositions under the given settings (see
``passes._undecomposed_graph``), ``trace`` exports without them and ``compile``
decomposes after the passes have run.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, Generator, Mapping, Optional, cast

import torch
from torch.export import ExportedProgram
from torch.export._trace import _export
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering._decomposition_filter import (
    filter_decomposition_table,
)
from torch_tensorrt.dynamo.lowering._decompositions import get_decompositions
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
    pre_export_lowering_needs_undecomposed_graph,
)

logger = logging.getLogger(__name__)

# Graph meta naming the ops of the decomp table already applied to the program.
_APPLIED_KEY = "torch_tensorrt_applied_decompositions"


def settings_from_kwargs(kwargs: Mapping[str, Any]) -> CompilationSettings:
    """The settings that the compile keyword arguments in ``kwargs`` give.

    ``dynamo.trace`` takes the same keyword arguments as ``compile``, and ignores
    the ones that are not settings.
    """
    names = {field.name for field in dataclasses.fields(CompilationSettings)}
    return CompilationSettings(**{k: v for k, v in kwargs.items() if k in names})


def _decomp_table(settings: CompilationSettings) -> dict[Any, Callable[..., Any]]:
    table = get_decompositions(
        settings.enable_experimental_decompositions,
        settings.decompose_attention,
        settings.use_distributed_mode_trace,
        settings.use_fp32_acc,
    )
    return cast(dict[Any, Callable[..., Any]], table)


def _graph_targets(graph_module: torch.fx.GraphModule) -> set[Any]:
    # Includes subgraphs, such as the branches of torch.cond.
    return {
        node.target
        for module in graph_module.modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
        if node.op == "call_function"
    }


def _is_post_dispatch_aten(exported_program: ExportedProgram) -> bool:
    """Whether ``exported_program`` is functional post-dispatch ATen IR.

    Stock ``torch.export`` gives training IR, which can hold in-place and composite
    ops outside the decomp table; ``run_decompositions`` functionalizes those
    whatever the table holds, so the graph's ops alone cannot tell it is done.
    """
    verifiers = getattr(exported_program, "verifiers", None)
    return bool(verifiers) and getattr(verifiers[0], "dialect", None) == "ATEN"


def _record_applied(
    exported_program: ExportedProgram, table: Mapping[Any, Any]
) -> ExportedProgram:
    """Record that ``table`` has been applied to ``exported_program``."""
    exported_program.graph_module.meta[_APPLIED_KEY] = tuple(
        sorted(str(op) for op in table)
    )
    return exported_program


def _applied(exported_program: ExportedProgram) -> frozenset[str]:
    return frozenset(exported_program.graph_module.meta.get(_APPLIED_KEY, ()))


def pending_decompositions(
    exported_program: ExportedProgram, table: Mapping[Any, Any]
) -> list[str]:
    """The ops in ``exported_program``, subgraphs included, that ``table`` rewrites
    and that no table applied to the program has already tried."""
    applied = _applied(exported_program)
    return sorted(
        str(target)
        for target in _graph_targets(exported_program.graph_module)
        if target in table and str(target) not in applied
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
def _aot_export_with_cia_split() -> Generator[None, None, None]:
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
    """Run ``run_decompositions`` unless the graph needs nothing from it.

    Call after ``pre_export_lowering``. Skips only when the program is already
    post-dispatch ATen IR, as ``dynamo.trace`` produces, and has no op that the
    decomp table for ``settings``, filtered for this graph, would rewrite and that
    the table ``trace`` applied did not already try. Ops that a different
    ``settings`` decomposes were not in that table, so they rerun it.
    """
    table = filter_decomposition_table(
        _decomp_table(settings), exported_program.graph_module
    )
    if not _is_post_dispatch_aten(exported_program):
        logger.info("Running run_decompositions: the graph is not yet ATen IR")
    else:
        pending = pending_decompositions(exported_program, table)
        if not pending:
            logger.info(
                "Skipping run_decompositions: the graph has no op the Torch-TRT "
                "decomp table rewrites"
            )
            return exported_program
        logger.info(
            "Running run_decompositions: the graph still has %s", ", ".join(pending)
        )
    return _record_applied(exported_program.run_decompositions(table), table)


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

    Exports with stock ``torch.export`` instead when a pre-export lowering pass
    needs the graph undecomposed under ``settings``, or when this PyTorch build
    cannot thread a decomp table through export.
    """
    export_kwargs = dict(kwargs or {})
    needs_undecomposed = pre_export_lowering_needs_undecomposed_graph(settings)
    can_inject = (
        _cia_split_apis() is not None
        and hasattr(torch.export._trace, "_export_to_aten_ir")
        and hasattr(torch.export._trace, "_aot_export_joint_with_descriptors")
    )
    if needs_undecomposed or not can_inject:
        if needs_undecomposed:
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

    # A decomposition that cannot apply to a node, such as scatter_add over a
    # symbolic extent, declines and leaves the op in place; recording the table
    # keeps compile from retracing for it.
    table = _decomp_table(settings)
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
    return _record_applied(exported_program, table)
