# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

import torch

from ._constants import (
    BACKEND_KEY,
    NAME_KEY,
    PLUGIN_KEY,
    ROUTE_KEY,
    SUPPORTED_BACKENDS,
    TOKEN_KEY,
)
from ._errors import RegionCaptureError
from ._session import RegionCompilationSession, current_session


def _active_session() -> Optional[RegionCompilationSession]:
    """Validate capture before emitting markers or resolving a user kernel."""
    session = current_session()
    if session is None or not torch.compiler.is_compiling():
        return None
    if not session.strict:
        raise RegionCaptureError(
            "Region capture requires torch_tensorrt.compile(..., ir='dynamo', "
            "strict=True); non-strict region capture is not supported"
        )
    if session.capture_error is not None:
        raise RegionCaptureError(session.capture_error)
    return session


@contextmanager
def execute_in_torch(
    *, name: Optional[str] = None, backend: str = "eager"
) -> Iterator[None]:
    """Keep one captured scope together in PyTorch during Dynamo compilation.

    Outside a Torch-TensorRT-owned compilation this is an eager no-op. The
    initial capture path requires ``strict=True``. The body must still be
    exportable, functional inference code; the scope does not permit graph
    breaks or arbitrary Python execution. Nested regions are rejected by the
    capture normalizer. ``name`` is an optional diagnostic label, not an ID.
    ``backend="eager"`` preserves ordinary PyTorch execution. ``"inductor"``
    compiles the region on its first execution in the resulting module, while
    keeping it outside TensorRT. Other backends are currently unsupported.
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
def _remember_plugin(
    kernel: Callable[[], Any],
    signature: Optional[dict[str, str]],
    constexprs: Optional[dict[str, Any]],
    grid: Optional[Callable[..., Any]],
    extra_args_fn: Optional[Callable[..., Any]],
    num_warps: Optional[int],
    num_stages: Optional[int],
) -> int:
    # Run at capture time, outside the traced tensor program. The definition
    # belongs to this compilation session, never to a process-global registry.
    from ._plugin import PluginSpec

    session = current_session()
    assert session is not None
    # Dynamo represents a raw @triton.jit function as TritonKernelVariable,
    # which cannot cross an assume_constant_result boundary. A zero-argument
    # Python factory is a constant and resolves the actual kernel here, outside
    # tensor tracing. The same form carries a CUDA KernelSpec without teaching
    # Dynamo how to flatten the mutable launch descriptor.
    resolved_kernel = kernel()
    key = len(session.plugins)
    session.plugins[key] = PluginSpec(
        resolved_kernel,
        signature,
        constexprs,
        grid,
        extra_args_fn,
        num_warps,
        num_stages,
    )
    return key


@contextmanager
def fuse_in_plugin(
    kernel: Callable[[], Any],
    *,
    name: Optional[str] = None,
    signature: Optional[dict[str, str]] = None,
    constexprs: Optional[dict[str, Any]] = None,
    grid: Optional[Callable[..., Any]] = None,
    extra_args_fn: Optional[Callable[..., Any]] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> Iterator[None]:
    """Replace this functional region with a user kernel in one TensorRT plugin.

    Supply ``kernel=lambda: my_triton_kernel`` and its launch configuration,
    or a zero-argument factory returning a CUDA ``kernels.KernelSpec``. Dynamo
    cannot treat a raw Triton JIT object as a Python constant; the factory is
    resolved outside the traced tensor program. An inline factory must refer
    to a global kernel; a factory created before capture can close over local
    kernel objects. The captured body supplies the Torch reference, schema and
    fake outputs; plugin and converter registration are automatic.
    Tensor inputs follow first-use order in the captured body; output pointers
    follow the production order of tensors used after the scope (which may
    differ from the model's return order). Static contiguous CUDA boundaries
    are required.
    Outside owned compilation, the body executes normally.
    """
    if name is not None and not isinstance(name, str):
        raise TypeError("fuse_in_plugin name must be a string or None")
    if _active_session() is None:
        yield
        return
    key = _remember_plugin(
        kernel, signature, constexprs, grid, extra_args_fn, num_warps, num_stages
    )
    with _scope(name=name, route="plugin", plugin=key):
        yield


@contextmanager
def _scope(
    *,
    name: Optional[str],
    backend: str = "eager",
    route: str = "torch",
    plugin: int = -1,
) -> Iterator[None]:
    if _active_session() is None:
        yield
        return

    # The owning session imported this module before Dynamo started tracing.
    from ._marker_ops import BEGIN, END

    sentinel = BEGIN(name or "", backend, route, plugin)
    # This identity is graph-local capture metadata, never an operator argument,
    # occurrence ID, or persistent cache key. The parser verifies uniqueness and
    # the end's exact sentinel edge before accepting membership.
    annotations = {
        TOKEN_KEY: id(sentinel),
        NAME_KEY: name or "",
        BACKEND_KEY: backend,
        ROUTE_KEY: route,
        PLUGIN_KEY: plugin,
    }
    with (
        torch.fx.traceback.annotate(annotations),
        torch._dynamo.error_on_graph_break(True),
    ):
        try:
            yield
        finally:
            END(sentinel)
