# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from contextlib import contextmanager
from functools import partial
from types import FunctionType
from typing import Any, Callable, Iterator, Optional, Sequence

import torch

from ._constants import (
    BACKEND_KEY,
    NAME_KEY,
    OPERATOR_KEY,
    PLUGIN_KEY,
    ROUTE_KEY,
    SUPPORTED_BACKENDS,
    TOKEN_KEY,
)
from ._errors import RegionCaptureError
from ._generation import KernelGenerator
from ._session import RegionCompilationSession, current_session

logger = logging.getLogger(__name__)

_warned_unowned_trace = False


@torch._dynamo.assume_constant_result  # type: ignore[misc]
def _warn_unowned_trace() -> None:
    # Dynamo runs this at trace time. A logger call in traced code is dropped
    # by strict torch.export and fails torch.compile(fullgraph=True).
    global _warned_unowned_trace
    if not _warned_unowned_trace:
        _warned_unowned_trace = True
        logger.warning(
            "torch_tensorrt.region annotations are ignored in this trace, so "
            "execute_in_torch bodies can be converted to TensorRT and fused_op "
            "and fused_kernel do not use the supplied operator or kernel. Only "
            "torch_tensorrt.compile(..., ir='dynamo', strict=True) captures "
            "regions; torch.compile, torch.export, torch_tensorrt.dynamo.trace "
            "and MutableTorchTensorRTModule do not."
        )


def _active_session() -> Optional[RegionCompilationSession]:
    """Validate capture before emitting markers or resolving a user kernel."""
    if not torch.compiler.is_compiling():
        return None
    session = current_session()
    if session is None:
        _warn_unowned_trace()
        return None
    if not session.strict:
        raise RegionCaptureError(
            "Region capture requires torch_tensorrt.compile(..., ir='dynamo', "
            "strict=True); non-strict region capture is not supported"
        )
    if session.capture_error is not None:
        raise RegionCaptureError(session.capture_error)
    return session


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

    Only ``torch_tensorrt.compile(..., ir="dynamo", strict=True)`` captures
    the scope; non-strict compilation raises ``RegionCaptureError``. Eager
    execution runs the body as ordinary code, and so do traces such as
    ``torch.compile`` and ``torch.export``, which log a one-time warning.
    """
    if name is not None and not isinstance(name, str):
        raise TypeError("execute_in_torch name must be a string or None")
    if not isinstance(backend, str):
        raise TypeError("execute_in_torch backend must be a string")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported execute_in_torch backend {backend!r}; "
            "expected 'eager' or 'inductor'"
        )
    with _scope(name=name, backend=backend):
        yield


@torch._dynamo.assume_constant_result  # type: ignore[misc]
def _expect_plugin(name: Optional[str]) -> None:
    session = current_session()
    assert session is not None
    # Dynamo converts _remember_plugin's arguments before calling it. If that
    # fails, this stays set and the session reports the scope's options.
    session.pending_plugin = name or ""


@torch._dynamo.assume_constant_result  # type: ignore[misc]
def _remember_plugin(
    kernel: Optional[Callable[[], Any]],
    generator: Optional[KernelGenerator],
    signature: Optional[dict[str, str]],
    constexprs: Optional[dict[str, Any]],
    grid: Optional[Callable[..., Any]],
    extra_args_fn: Optional[Callable[..., Any]],
    num_warps: Optional[int],
    num_stages: Optional[int],
) -> int:
    from ._plugin import prepare_plugin

    session = current_session()
    assert session is not None
    session.pending_plugin = None
    # Resolve the factory outside tracing: Dynamo cannot pass a raw JIT kernel.
    key = len(session.plugins)
    if generator is not None:
        session.plugins[key] = generator
    else:
        assert kernel is not None
        session.plugins[key] = prepare_plugin(
            kernel(),
            signature=signature,
            constexprs=constexprs,
            grid=grid,
            extra_args_fn=extra_args_fn,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return key


@contextmanager
def fused_kernel(
    kernel: Optional[Callable[[], Any]] = None,
    *,
    generator: Optional[KernelGenerator] = None,
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

    Alternatively, ``generator`` authors and evaluates a kernel during
    compilation and supplies its launch configuration. Generated regions
    require static, contiguous, nonempty float16, bfloat16 or float32 CUDA
    tensor boundaries. A dry run checks what generation requires and, instead
    of generating a kernel, executes the body where the kernel's plugin would
    run.

    Only ``torch_tensorrt.compile(..., ir="dynamo", strict=True)`` captures
    the scope; non-strict compilation raises ``RegionCaptureError``. Eager
    execution runs the body as ordinary code, and so do traces such as
    ``torch.compile`` and ``torch.export``, which log a one-time warning.

    Args:
        kernel: Function or ``functools.partial``, not a method, returning a
            ``@triton.jit`` kernel or CUDA ``KernelSpec``, such as
            ``lambda: my_kernel``. Compilation calls it outside tracing, so a
            lambda written in ``forward`` must refer to globals, not ``self``
            or locals; create other factories before compilation, for example
            in ``__init__``.
        generator: A :class:`KernelGenerator` to use instead of ``kernel``.
            Construct it before compilation, for example once at module level or
            in a closure, not inside the traced ``forward``.
        inputs: Every tensor the body reads, including parameters and buffers,
            in kernel parameter order. Omit it only when the body reads one
            tensor, or for a ``generator``, which writes its kernel for the
            order the region first reads them.
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
    if (kernel is None) == (generator is None):
        raise ValueError("fused_kernel requires exactly one of kernel or generator")
    if generator is not None:
        if not isinstance(generator, KernelGenerator):
            raise TypeError("fused_kernel generator must be a KernelGenerator")
        if any(
            value is not None
            for value in (
                signature,
                constexprs,
                grid,
                extra_args_fn,
                num_warps,
                num_stages,
            )
        ):
            raise ValueError(
                "KernelGenerator supplies its own kernel launch configuration"
            )
    if name is not None and not isinstance(name, str):
        raise TypeError("fused_kernel name must be a string or None")
    _check_inputs(inputs, "fused_kernel")
    # Compilation passes these to _remember_plugin as constants: Dynamo converts
    # functions and partials, but not kernel objects or bound methods.
    if generator is None and not isinstance(kernel, (FunctionType, partial)):
        raise TypeError(
            "fused_kernel kernel must be a function or functools.partial returning "
            f"the kernel, such as kernel=lambda: my_kernel; got {type(kernel).__name__}"
        )
    for option, callback in (("grid", grid), ("extra_args_fn", extra_args_fn)):
        if callback is not None and not isinstance(callback, (FunctionType, partial)):
            raise TypeError(
                f"fused_kernel {option} must be a function or functools.partial; "
                f"got {type(callback).__name__}"
            )
    if _active_session() is None:
        yield
        return
    _expect_plugin(name)
    key = _remember_plugin(
        kernel,
        generator,
        signature,
        constexprs,
        grid,
        extra_args_fn,
        num_warps,
        num_stages,
    )
    with _scope(name=name, route="plugin", plugin=key, inputs=inputs):
        yield


@torch._dynamo.assume_constant_result  # type: ignore[misc]
def _remember_operator(op: Any) -> int:
    if isinstance(op, torch._ops.OpOverloadPacket):
        try:
            op = op.default
        except AttributeError as error:
            raise TypeError(
                "fused_op requires an explicit custom-op overload"
            ) from error
    if not isinstance(op, torch._ops.OpOverload) or op.namespace in {
        "aten",
        "prim",
        "prims",
        "torch_tensorrt_region",
    }:
        raise TypeError("fused_op requires a registered torch custom operator")
    schema = op._schema
    if schema.is_mutable:
        raise RegionCaptureError("fused_op does not support a mutating operator schema")
    if any(
        argument.alias_info is not None
        for argument in (*schema.arguments, *schema.returns)
    ):
        raise RegionCaptureError(
            "fused_op does not support an aliasing operator schema"
        )
    output_types = [argument.type for argument in schema.returns]
    if len(output_types) == 1 and isinstance(output_types[0], torch.TupleType):
        output_types = output_types[0].elements()
    if (
        not output_types
        or any(argument.kwarg_only for argument in schema.arguments)
        or any(
            argument_type != torch._C.TensorType.get()
            for argument_type in [argument.type for argument in schema.arguments]
            + output_types
        )
    ):
        raise RegionCaptureError(
            "fused_op requires positional Tensor inputs and Tensor outputs"
        )
    session = current_session()
    assert session is not None
    key = len(session.operators)
    session.operators[key] = op
    return key


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

    Only ``torch_tensorrt.compile(..., ir="dynamo", strict=True)`` captures
    the scope; non-strict compilation raises ``RegionCaptureError``. Eager
    execution runs the body as ordinary code, and so do traces such as
    ``torch.compile`` and ``torch.export``, which log a one-time warning.
    """
    if name is not None and not isinstance(name, str):
        raise TypeError("fused_op name must be a string or None")
    if isinstance(op, torch._library.custom_ops.CustomOpDef):
        op = op._opoverload
    if not isinstance(op, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
        raise TypeError(
            "fused_op requires a torch.library.custom_op function or "
            "torch.ops.<namespace>.<name>"
        )
    _check_inputs(inputs, "fused_op")
    if _active_session() is None:
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
    plugin: int = -1,
    operator: int = -1,
    inputs: Optional[Sequence[torch.Tensor]] = None,
) -> Iterator[None]:
    if _active_session() is None:
        yield
        return

    from ._marker_ops import BEGIN, END

    listed = None if inputs is None else list(inputs)
    sentinel = BEGIN(name or "", backend, route, plugin, operator, listed)
    # Graph-local identity pairs metadata with the exact begin/end edge.
    annotations = {
        TOKEN_KEY: id(sentinel),
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
