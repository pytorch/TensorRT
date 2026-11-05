# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from contextvars import ContextVar, Token
from functools import partial
from types import TracebackType
from typing import TYPE_CHECKING, Optional

from ._errors import RegionCaptureError

if TYPE_CHECKING:
    from torch._ops import OpOverload
    from torch_tensorrt.kernels import KernelSpec

_CURRENT_SESSION: ContextVar[Optional[RegionCompilationSession]] = ContextVar(
    "torch_tensorrt_region_session", default=None
)


class RegionCompilationSession:
    """Own marker registration and kernel definitions from export to compilation."""

    def __init__(self, *, strict: bool) -> None:
        self.strict = strict
        self.capture_error: Optional[str] = None
        # Name of the fused_kernel scope whose options Dynamo is converting
        # ('' if unnamed), else None.
        self.pending_plugin: Optional[str] = None
        self.plugins: dict[int, KernelSpec | partial[None]] = {}
        self.operators: dict[int, OpOverload] = {}
        self._context_token: Optional[Token[Optional[RegionCompilationSession]]] = None

    def __enter__(self) -> RegionCompilationSession:
        if self._context_token is not None:
            raise RegionCaptureError(
                "A region compilation session cannot re-enter itself"
            )
        if self.strict:
            from . import _marker_ops

            self.capture_error = _marker_ops.CAPTURE_ERROR
        self._context_token = _CURRENT_SESSION.set(self)
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        token = self._context_token
        if token is None:
            raise RegionCaptureError("A region compilation session was not entered")
        _CURRENT_SESSION.reset(token)
        self._context_token = None
        pending, self.pending_plugin = self.pending_plugin, None
        if pending is not None and isinstance(exc, Exception):
            # Dynamo's conversion error does not explain the fused_kernel contract.
            label = f" {pending!r}" if pending else ""
            raise RegionCaptureError(
                f"fused_kernel{label} options must be compile-time constants: "
                "they cannot hold tensors or symbolic sizes, and a kernel, grid "
                "or extra_args_fn lambda written in forward cannot refer to self "
                "or local variables. Refer to globals, or create the function "
                "before compilation, for example in __init__"
            ) from exc


def current_session() -> Optional[RegionCompilationSession]:
    return _CURRENT_SESSION.get()
