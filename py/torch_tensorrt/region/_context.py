# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator, Optional, Sequence

import torch

from . import _marker_ops
from ._constants import (
    BACKEND_KEY,
    NAME_KEY,
    OPERATOR_KEY,
    PLUGIN_KEY,
    ROUTE_KEY,
    TOKEN_KEY,
)
from ._errors import RegionCaptureError
from ._marker_ops import BEGIN, END
from ._operators import check_operator, operator_name
from ._rules import (
    check_backend,
    check_kernel_options,
    check_name,
    check_operator_type,
)
from ._session import current_session

logger = logging.getLogger(__name__)

# Set on the Torch-TensorRT torch.compile backends, which consume the markers.
BACKEND_ATTRIBUTE = "_torch_tensorrt_records_regions"

_IGNORE_REGIONS: ContextVar[bool] = ContextVar(
    "torch_tensorrt_ignore_regions", default=False
)
# Kinds of trace that already warned about ignored annotations.
_warned_ignored: set[str] = set()
# One token per scope a trace enters, to pair its tags with its markers.
_TOKENS = itertools.count(1)

_IGNORED_BY = {
    "backend": (
        "torch_tensorrt.region annotations are ignored by this torch.compile "
        "backend, so the annotated code runs as ordinary operations. Only "
        "torch.export and the Torch-TensorRT backend, torch.compile(model, "
        "backend='torch_tensorrt'), record regions."
    ),
    "torchscript": (
        "torch_tensorrt.region annotations are ignored by TorchScript tracing, "
        "so execute_in_torch bodies can be converted to TensorRT and fused_op "
        "and fused_kernel do not use the supplied operator or kernel. Compile "
        "with ir='dynamo' or torch.compile(model, backend='torch_tensorrt')."
    ),
    "untraced": (
        "torch_tensorrt.region annotations are ignored in a forward that "
        "torch.compile did not trace, for example after a graph break inside "
        "a loop, so the annotated code runs as ordinary operations there."
    ),
}


def _warn_ignored(kind: str) -> None:
    if kind not in _warned_ignored:
        _warned_ignored.add(kind)
        logger.warning(_IGNORED_BY[kind])


@contextmanager
def ignore_regions() -> Iterator[None]:
    """Trace annotated code as ordinary operations inside this block.

    ``torch.export`` records region annotations in the exported program so that
    Torch-TensorRT can honor them later. Use this block when the program is
    exported for something else, such as ``torch.onnx.export``, that does not
    know the region markers.

    ``torch.compile`` traces at the first call of the compiled model, so the
    block must surround that call. The choice stays with the compiled code:
    later calls reuse it, inside or outside the block.
    """
    token = _IGNORE_REGIONS.set(True)
    try:
        yield
    finally:
        _IGNORE_REGIONS.reset(token)


def _backend_records_regions() -> bool:
    """Whether the torch.compile backend of the frame Dynamo is tracing is ours."""
    try:
        from torch._dynamo.symbolic_convert import InstructionTranslator

        output = InstructionTranslator.current_tx().output
        backend = output.compiler_fn
        traced = getattr(output, "root_tx", None)
    except Exception:
        _warn_ignored("backend")
        return False
    if traced is not None and traced.f_code.co_filename == __file__:
        # torch.compile skipped the forward that holds the scope and is now
        # tracing these helpers on their own: the scope runs as ordinary code.
        _warn_ignored("untraced")
        return False
    if not _is_region_backend(backend):
        _warn_ignored("backend")
        return False
    return True


def _is_region_backend(backend: Any) -> bool:
    # torch.compile wraps the backend for options, debugging and partial calls.
    seen: set[int] = set()
    while backend is not None and id(backend) not in seen:
        seen.add(id(backend))
        if getattr(backend, BACKEND_ATTRIBUTE, False):
            return True
        backend = next(
            (
                inner
                for name in (
                    "_torchdynamo_orig_backend",
                    "compiler_fn",
                    "func",
                    "__wrapped__",
                )
                if (inner := getattr(backend, name, None)) is not None
            ),
            None,
        )
    return False


@torch._dynamo.assume_constant_result  # type: ignore[misc]
def _trace_records_regions() -> bool:
    # Dynamo runs this at trace time, so it adds no guard, graph break or
    # traced logger call: strict torch.export drops those and
    # torch.compile(fullgraph=True) rejects them.
    if _IGNORE_REGIONS.get():
        return False
    if not torch.compiler.is_exporting() and not _backend_records_regions():
        return False
    # Raised here, at trace time: an exception from traced code would make
    # torch.compile skip the forward and run it without its regions.
    session = current_session()
    error = _marker_ops.CAPTURE_ERROR if session is None else session.capture_error
    if error is not None:
        raise RegionCaptureError(error)
    return True


@torch._dynamo.assume_constant_result  # type: ignore[misc]
def _next_token() -> int:
    return next(_TOKENS)


def _recording() -> bool:
    """Whether this call is a trace that records regions."""
    if not torch.compiler.is_compiling():
        if torch.jit.is_tracing() and not _IGNORE_REGIONS.get():
            _warn_ignored("torchscript")
        return False
    return bool(_trace_records_regions())


def _check_inputs(inputs: Optional[Sequence[torch.Tensor]], api: str) -> None:
    if inputs is not None and (
        not isinstance(inputs, (tuple, list))
        or not all(isinstance(value, torch.Tensor) for value in inputs)
    ):
        raise TypeError(f"{api} inputs must be a tuple or list of tensors")


@contextmanager
def execute_in_torch(
    *, name: Optional[str] = None, backend: str = "eager"
) -> Iterator[None]:
    """Keep a functional, exportable scope outside TensorRT.

    ``backend="eager"`` runs the captured PyTorch operations; ``"inductor"``
    compiles them on first execution. Nesting and graph breaks are
    unsupported. ``name`` is a diagnostic label.

    ``torch.export`` records the scope in the exported program, and
    ``torch_tensorrt.compile(..., ir="dynamo")`` and
    ``torch_tensorrt.dynamo.compile`` honor it; so does
    ``torch.compile(model, backend="torch_tensorrt")``. Eager execution runs
    the body as ordinary code, and so do other ``torch.compile`` backends and
    TorchScript tracing, which log a warning.

    To place a submodule you cannot edit in a region, pass
    ``regions={"path.to.module": ExecuteInTorch()}`` to the compile call.
    """
    check_name(name, "execute_in_torch")
    check_backend(backend)
    if not _recording():
        yield
        return
    with _scope(name=name, backend=backend):
        yield


@torch._dynamo.assume_constant_result  # type: ignore[misc]
def _expect_plugin(name: Optional[str]) -> None:
    session = current_session()
    if session is not None:
        # Dynamo converts _remember_plugin's arguments before calling it. If
        # that fails, this stays set and the session reports the scope's options.
        session.pending_plugin = name or ""


@torch._dynamo.assume_constant_result  # type: ignore[misc]
def _remember_plugin(
    kernel: Callable[[], Any],
    signature: Optional[dict[str, str]],
    constexprs: Optional[dict[str, Any]],
    grid: Optional[Callable[..., Any]],
    extra_args_fn: Optional[Callable[..., Any]],
    num_warps: Optional[int],
    num_stages: Optional[int],
) -> str:
    from ._plugin import prepare_plugin, remember_kernel

    session = current_session()
    if session is not None:
        session.pending_plugin = None
    # Resolve the factory outside tracing: Dynamo cannot pass a raw JIT kernel.
    resolved = kernel()
    return remember_kernel(
        prepare_plugin(
            resolved,
            signature=signature,
            constexprs=constexprs,
            grid=grid,
            extra_args_fn=extra_args_fn,
            num_warps=num_warps,
            num_stages=num_stages,
        ),
        (
            resolved,
            None if signature is None else tuple(signature.items()),
            None if constexprs is None else tuple(constexprs.items()),
            grid,
            extra_args_fn,
            num_warps,
            num_stages,
        ),
    )


@contextmanager
def fused_kernel(
    kernel: Callable[[], Any],
    *,
    inputs: Optional[Sequence[torch.Tensor]] = None,
    name: Optional[str] = None,
    signature: Optional[dict[str, str]] = None,
    constexprs: Optional[dict[str, Any]] = None,
    grid: Optional[Callable[..., Any]] = None,
    extra_args_fn: Optional[Callable[..., Any]] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> Iterator[None]:
    """Replace a functional scope with one TensorRT kernel plugin.

    The body defines the plugin's inputs and outputs, which must be static,
    contiguous CUDA tensors; outputs follow production order. The Triton
    options follow ``kernels.triton_op``; omit them for a CUDA ``KernelSpec``,
    which provides its own launch configuration.

    ``torch.export`` records the scope in the exported program, and
    ``torch_tensorrt.compile(..., ir="dynamo")`` and
    ``torch_tensorrt.dynamo.compile`` honor it; so does
    ``torch.compile(model, backend="torch_tensorrt")``. Eager execution runs
    the body as ordinary code, and so do other ``torch.compile`` backends and
    TorchScript tracing, which log a warning. The kernel is a Python object: an
    exported program must be compiled in the process that exported it, because
    saving the program does not save the kernel.

    To replace a submodule you cannot edit, pass
    ``regions={"path.to.module": FusedKernel(...)}`` to the compile call.

    Args:
        kernel: Function or ``functools.partial``, not a method, returning a
            ``@triton.jit`` kernel or CUDA ``KernelSpec``, such as
            ``lambda: my_kernel``. Strict export and ``torch.compile`` call it
            outside the trace, so a lambda written in ``forward`` must refer
            to globals, not ``self`` or locals; create other factories before
            compilation, for example in ``__init__``.
        inputs: Every tensor the body reads, including parameters and buffers,
            in kernel parameter order. Omit it only when the body reads one
            tensor.
        name: Diagnostic label.
        signature: Triton signature; pointer types and ``i32`` extras are
            inferred when it is omitted.
        constexprs: Triton ``tl.constexpr`` values.
        grid: Triton launch grid callback, required for a Triton kernel; the
            rules for ``kernel`` apply.
        extra_args_fn: Triton scalar launch argument callback; the rules for
            ``kernel`` apply.
        num_warps: Triton warps per block.
        num_stages: Triton software pipelining depth.
    """
    check_name(name, "fused_kernel")
    _check_inputs(inputs, "fused_kernel")
    check_kernel_options(kernel, grid, extra_args_fn)
    if not _recording():
        yield
        return
    _expect_plugin(name)
    key = _remember_plugin(
        kernel, signature, constexprs, grid, extra_args_fn, num_warps, num_stages
    )
    with _scope(name=name, route="plugin", plugin=key, inputs=inputs):
        yield


@torch._dynamo.assume_constant_result  # type: ignore[misc]
def _remember_operator(op: Any) -> str:
    return operator_name(check_operator(op))


@contextmanager
def fused_op(
    op: Any,
    *,
    inputs: Optional[Sequence[torch.Tensor]] = None,
    name: Optional[str] = None,
) -> Iterator[None]:
    """Replace a scope with a registered custom operator that stays in PyTorch.

    ``op`` is a ``torch.library.custom_op`` function or
    ``torch.ops.<namespace>.<name>``, optionally with an explicit overload.
    ``inputs`` lists every tensor the body reads, including parameters and
    buffers, in operator argument order; it may be omitted only when the body
    reads at most one tensor. Outputs follow production order. The op needs a
    functional Tensor-only schema and a matching fake implementation. Random
    operations are supported.

    ``torch.export`` records the scope in the exported program, and
    ``torch_tensorrt.compile(..., ir="dynamo")`` and
    ``torch_tensorrt.dynamo.compile`` honor it; so does
    ``torch.compile(model, backend="torch_tensorrt")``. Eager execution runs
    the body as ordinary code, and so do other ``torch.compile`` backends and
    TorchScript tracing, which log a warning. The exported program records the
    operator by name, so the process that compiles it must have registered the
    operator.

    To replace a submodule you cannot edit, pass
    ``regions={"path.to.module": FusedOp(op, inputs=...)}`` to the compile call.
    """
    check_name(name, "fused_op")
    check_operator_type(op)
    if isinstance(op, torch._library.custom_ops.CustomOpDef):
        op = op._opoverload
    _check_inputs(inputs, "fused_op")
    if not _recording():
        yield
        return
    key = _remember_operator(op)
    with _scope(name=name, operator=key, inputs=inputs):
        yield


@contextmanager
def _scope(
    *,
    name: Optional[str],
    backend: str = "eager",
    route: str = "torch",
    plugin: str = "",
    operator: str = "",
    inputs: Optional[Sequence[torch.Tensor]] = None,
) -> Iterator[None]:
    listed = None if inputs is None else list(inputs)
    sentinel = BEGIN(name or "", backend, route, plugin, operator, listed)
    # The token pairs the scope's tags with its own begin and end markers.
    annotations = {
        TOKEN_KEY: _next_token(),
        NAME_KEY: name or "",
        BACKEND_KEY: backend,
        ROUTE_KEY: route,
        PLUGIN_KEY: plugin,
        OPERATOR_KEY: operator,
    }
    with (
        torch.fx.traceback.annotate(annotations),
        torch._dynamo.error_on_graph_break(True),
    ):
        try:
            yield
        finally:
            END(sentinel)
