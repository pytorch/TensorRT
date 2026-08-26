"""Apply Torch-TRT decompositions during export so compile can skip a second AOT retrace.

Stock ``torch.export`` captures without Torch-TRT's decomp table;
``dynamo.compile`` then calls ``ExportedProgram.run_decompositions``, which
re-traces. ``dynamo.trace`` uses this helper so the default
``torch_tensorrt.compile(nn.Module)`` path is one trace. ``compile`` skips
``run_decompositions`` when the exported program already carries a matching
table fingerprint — no user-facing flag.
"""

from __future__ import annotations

import functools
import inspect
import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, Optional, cast

import torch
from torch.export import ExportedProgram
from torch.export._trace import _export
from torch_tensorrt.dynamo.lowering._decompositions import get_decompositions

logger = logging.getLogger(__name__)

_STAMP_KEY = "torch_tensorrt_decomp_fingerprint"
_STAMP_VERSION = 1


def decomp_fingerprint(
    enable_experimental_decompositions: bool = False,
    decompose_attention: bool = False,
    use_distributed_mode_trace: bool = False,
    use_fp32_acc: bool = False,
) -> dict[str, Any]:
    """Stable id for the decomp table ``compile`` would apply."""
    return {
        "v": _STAMP_VERSION,
        "experimental": bool(enable_experimental_decompositions),
        "attention": bool(decompose_attention),
        "distributed": bool(use_distributed_mode_trace),
        "fp32_acc": bool(use_fp32_acc),
    }


def stamp_exported_program(
    exported_program: ExportedProgram, fingerprint: dict[str, Any]
) -> ExportedProgram:
    exported_program.graph_module.meta[_STAMP_KEY] = dict(fingerprint)
    graph_meta = getattr(exported_program.graph_module.graph, "meta", None)
    if isinstance(graph_meta, dict):
        graph_meta[_STAMP_KEY] = dict(fingerprint)
    return exported_program


def _decomp_table(
    enable_experimental_decompositions: bool,
    decompose_attention: bool,
    use_distributed_mode_trace: bool,
    use_fp32_acc: bool,
) -> dict[Any, Callable[..., Any]]:
    if "use_fp32_acc" in inspect.signature(get_decompositions).parameters:
        table = get_decompositions(
            enable_experimental_decompositions,
            decompose_attention,
            use_distributed_mode_trace,
            use_fp32_acc=use_fp32_acc,
        )
    else:
        table = get_decompositions(
            enable_experimental_decompositions,
            decompose_attention,
            use_distributed_mode_trace,
        )
    return cast(dict[Any, Callable[..., Any]], table)


def matching_decomp_stamp(
    exported_program: ExportedProgram, fingerprint: dict[str, Any]
) -> bool:
    stamped: object = exported_program.graph_module.meta.get(_STAMP_KEY)
    if stamped != fingerprint:
        graph_meta = getattr(exported_program.graph_module.graph, "meta", None)
        if isinstance(graph_meta, dict):
            stamped = graph_meta.get(_STAMP_KEY)
    return stamped == fingerprint


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
) -> Iterator[None]:
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
    exported_program: ExportedProgram,
    *,
    enable_experimental_decompositions: bool = False,
    decompose_attention: bool = False,
    use_distributed_mode_trace: bool = False,
    use_fp32_acc: bool = False,
) -> ExportedProgram:
    """Run ``run_decompositions`` unless ``trace`` already applied the same table."""
    fingerprint = decomp_fingerprint(
        enable_experimental_decompositions,
        decompose_attention,
        use_distributed_mode_trace,
        use_fp32_acc,
    )
    if matching_decomp_stamp(exported_program, fingerprint):
        logger.info(
            "Skipping run_decompositions: exported program already has the matching "
            "Torch-TRT decomp table from dynamo.trace"
        )
        return exported_program

    exported_program = exported_program.run_decompositions(
        _decomp_table(
            enable_experimental_decompositions,
            decompose_attention,
            use_distributed_mode_trace,
            use_fp32_acc,
        )
    )
    return stamp_exported_program(exported_program, fingerprint)


def export_with_tensorrt_decomps(
    mod: torch.nn.Module,
    args: tuple[Any, ...] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
    *,
    strict: bool = False,
    dynamic_shapes: Any = None,
    prefer_deferred_runtime_asserts_over_guards: bool = False,
    enable_experimental_decompositions: bool = False,
    decompose_attention: bool = False,
    use_distributed_mode_trace: bool = False,
    use_fp32_acc: bool = False,
) -> ExportedProgram:
    """Capture ``mod`` and apply Torch-TRT decompositions in a single AOT trace.

    Falls back to stock ``torch.export.export`` (no stamp) if this PyTorch build
    cannot thread a decomp table through export.
    """
    export_kwargs = dict(kwargs or {})
    fingerprint = decomp_fingerprint(
        enable_experimental_decompositions,
        decompose_attention,
        use_distributed_mode_trace,
        use_fp32_acc,
    )
    table = _decomp_table(
        enable_experimental_decompositions,
        decompose_attention,
        use_distributed_mode_trace,
        use_fp32_acc,
    )
    can_inject = (
        _cia_split_apis() is not None
        and hasattr(torch.export._trace, "_export_to_aten_ir")
        and hasattr(torch.export._trace, "_aot_export_joint_with_descriptors")
    )
    if not can_inject:
        logger.warning(
            "Cannot apply Torch-TRT decomps during export on this PyTorch build; "
            "falling back to torch.export. compile() will run_decompositions."
        )
        return torch.export.export(
            mod,
            args=args,
            kwargs=export_kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
            prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
        )

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
    return stamp_exported_program(exported_program, fingerprint)
