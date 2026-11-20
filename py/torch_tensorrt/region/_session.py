# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from contextvars import ContextVar, Token
from types import TracebackType
from typing import Optional

from ._errors import RegionCaptureError

_CURRENT_SESSION: ContextVar[Optional[RegionCompilationSession]] = ContextVar(
    "torch_tensorrt_region_session", default=None
)


class RegionCompilationSession:
    """Explain region capture failures while Torch-TensorRT traces one model.

    Regions are recorded by any ``torch.export`` trace and by the Torch-TensorRT
    ``torch.compile`` backend, with or without a session. The session only adds
    diagnostics that need the surrounding compile call. ``strict`` is kept for
    callers; strict and non-strict export both record regions.
    """

    def __init__(self, *, strict: bool = True) -> None:
        self.strict = strict
        self.capture_error: Optional[str] = None
        # Name of the fused_kernel scope whose options Dynamo is converting
        # ('' if unnamed), else None.
        self.pending_plugin: Optional[str] = None
        # Set by a compile call that checked the model it traced was in eval mode.
        self.model_in_eval_mode = False
        # Why tracing could not record how the modules that rules replace are
        # called, when it could not.
        self.call_record_failure: Optional[str] = None
        self._context_token: Optional[Token[Optional[RegionCompilationSession]]] = None

    def __enter__(self) -> RegionCompilationSession:
        if self._context_token is not None:
            raise RegionCaptureError(
                "A region compilation session cannot re-enter itself"
            )
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
