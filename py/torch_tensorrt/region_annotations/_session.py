# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from contextvars import ContextVar, Token
from functools import partial
from types import TracebackType
from typing import TYPE_CHECKING, Optional

from ._errors import RegionCaptureError

if TYPE_CHECKING:
    from torch_tensorrt.kernels import KernelSpec

    from ._generation import KernelGenerator

_CURRENT_SESSION: ContextVar[Optional[RegionCompilationSession]] = ContextVar(
    "torch_tensorrt_region_session", default=None
)


class RegionCompilationSession:
    """Own marker registration and kernel definitions from export to compilation."""

    def __init__(self, *, strict: bool) -> None:
        self.strict = strict
        self.capture_error: Optional[str] = None
        self.plugins: dict[int, KernelSpec | partial[None] | KernelGenerator] = {}
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


def current_session() -> Optional[RegionCompilationSession]:
    return _CURRENT_SESSION.get()
