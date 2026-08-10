"""Export an ``nn.Module`` with Torch-TensorRT decompositions in one trace.

Stock ``torch.export`` captures without a backend decomp table; ``dynamo.compile``
then calls ``ExportedProgram.run_decompositions(get_decompositions())``, which
re-traces the graph. This module applies Torch-TRT's table during export using
the same CIA split that ``run_decompositions`` uses, so callers can pass
``skip_decompositions=True`` to ``compile`` and avoid the second trace.
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

_LOGGER = logging.getLogger(__name__)


def _require_cia_split_apis() -> tuple[Callable[..., Any], Callable[..., Any]]:
    try:
        from torch.export.exported_program import (
            _override_composite_implicit_decomp,
            _split_decomp_table_to_cia_and_python_decomp,
        )
    except ImportError as exc:
        raise RuntimeError(
            "export_for_tensorrt requires a PyTorch build that provides "
            "_split_decomp_table_to_cia_and_python_decomp and "
            "_override_composite_implicit_decomp on "
            "torch.export.exported_program (typically PyTorch 2.6+)."
        ) from exc
    return (
        _override_composite_implicit_decomp,
        _split_decomp_table_to_cia_and_python_decomp,
    )


@contextmanager
def _aot_export_with_cia_split() -> Iterator[None]:
    """When AOT sees ``decompositions=table``, CIA-split like ``run_decompositions``."""
    import torch.export._trace as export_trace

    (
        override_composite_implicit_decomp,
        split_decomp_table_to_cia_and_python_decomp,
    ) = _require_cia_split_apis()
    original = export_trace._aot_export_joint_with_descriptors

    def _wrapped(stack: Any, mod: Any, fake_args: Any, **flags: Any) -> Any:
        table = flags.get("decompositions")
        if table is not None:
            # run_decompositions mutates the table via del; always copy.
            cia_to_decomp, python_decomp_table = (
                split_decomp_table_to_cia_and_python_decomp(dict(table))
            )
            _LOGGER.debug(
                "export_for_tensorrt: CIA-split decomp_table (cia=%d, python=%d)",
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
    """Thread ``decomp_table`` into ``_export_to_aten_ir`` (public ``_export`` omits it)."""
    import torch.export._trace as export_trace

    if not hasattr(export_trace, "_export_to_aten_ir"):
        raise RuntimeError(
            "export_for_tensorrt requires torch.export._trace._export_to_aten_ir"
        )
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


def _get_decompositions_table(
    enable_experimental_decompositions: bool,
    decompose_attention: bool,
    use_distributed_mode_trace: bool,
    use_fp32_acc: bool,
) -> dict[Any, Callable[..., Any]]:
    kwargs: dict[str, Any] = {}
    if "use_fp32_acc" in inspect.signature(get_decompositions).parameters:
        kwargs["use_fp32_acc"] = use_fp32_acc
    return cast(
        dict[Any, Callable[..., Any]],
        get_decompositions(
            enable_experimental_decompositions,
            decompose_attention,
            use_distributed_mode_trace,
            **kwargs,
        ),
    )


def export_for_tensorrt(
    mod: torch.nn.Module,
    args: tuple[Any, ...] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
    *,
    strict: bool = False,
    prefer_deferred_runtime_asserts_over_guards: bool = True,
    enable_experimental_decompositions: bool = False,
    decompose_attention: bool = False,
    use_distributed_mode_trace: bool = False,
    use_fp32_acc: bool = False,
) -> ExportedProgram:
    """Capture ``mod`` and apply Torch-TRT decompositions in a single trace.

    Use with :func:`torch_tensorrt.dynamo.compile` and
    ``skip_decompositions=True`` (or prefer :func:`export_and_compile`) so
    ``run_decompositions`` is not paid a second time.
    """
    export_kwargs = dict(kwargs or {})
    table = _get_decompositions_table(
        enable_experimental_decompositions,
        decompose_attention,
        use_distributed_mode_trace,
        use_fp32_acc,
    )
    _LOGGER.info(
        "export_for_tensorrt: exporting with decomp_table (%d entries)",
        len(table),
    )

    with _aot_export_with_cia_split():
        with _export_to_aten_ir_with_decomp_table(table):
            return _export(
                mod,
                args=args,
                kwargs=export_kwargs,
                strict=strict,
                prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
            )
