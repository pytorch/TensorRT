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
    """Enable region capture only for an owned trace/compilation attempt.

    The compiler enters this context before export and keeps it until compilation
    ends. Marker registration happens outside tracing; ordinary eager imports do
    not register operators. Tokens and occurrence IDs are never allocated here.
    """

    def __init__(self, *, strict: bool) -> None:
        self.strict = strict
        self.capture_error: Optional[str] = None
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

    @staticmethod
    def current() -> Optional[RegionCompilationSession]:
        return _CURRENT_SESSION.get()


def current_session() -> Optional[RegionCompilationSession]:
    return _CURRENT_SESSION.get()
