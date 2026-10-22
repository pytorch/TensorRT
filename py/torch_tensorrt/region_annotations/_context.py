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
from ._generation import KernelGenerator
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
    """Keep a functional, exportable scope outside TensorRT.

    ``backend="eager"`` uses PyTorch; ``"inductor"`` compiles on first execution.
    Requires ``strict=True``; nesting and graph breaks are unsupported.
    ``name`` is a diagnostic label. Outside owned compilation this is a no-op.
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
def fuse_in_plugin(
    kernel: Optional[Callable[[], Any]] = None,
    *,
    generator: Optional[KernelGenerator] = None,
    name: Optional[str] = None,
    signature: Optional[dict[str, str]] = None,
    constexprs: Optional[dict[str, Any]] = None,
    grid: Optional[Callable[..., Any]] = None,
    extra_args_fn: Optional[Callable[..., Any]] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> Iterator[None]:
    """Replace a functional scope with one TensorRT kernel plugin.

    ``kernel`` is a factory returning a Triton JIT kernel or CUDA ``KernelSpec``.
    Inline factories reference global kernels; create factories before capture
    to use local kernels. Triton options follow ``kernels.triton_op``; pointer
    types and i32 extras are inferred unless ``signature`` is supplied.
    Inputs follow first-use order, outputs production order. Boundaries must
    be static, contiguous CUDA tensors. The body provides the Torch contract
    and runs normally outside owned compilation.

    Alternatively, ``generator=KernelGenerator(...)`` authors and evaluates a
    Triton or cuTile kernel during compilation using the selected agent CLI.
    Generated regions currently require floating-point tensor boundaries.
    """
    if (kernel is None) == (generator is None):
        raise ValueError("fuse_in_plugin requires exactly one of kernel or generator")
    if generator is not None:
        if not isinstance(generator, KernelGenerator):
            raise TypeError("fuse_in_plugin generator must be a KernelGenerator")
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
        raise TypeError("fuse_in_plugin name must be a string or None")
    if _active_session() is None:
        yield
        return
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

    from ._marker_ops import BEGIN, END

    sentinel = BEGIN(name or "", backend, route, plugin)
    # Graph-local identity pairs metadata with the exact begin/end edge.
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
