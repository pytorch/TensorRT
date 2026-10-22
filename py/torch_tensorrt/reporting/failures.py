# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Conservative exception classification without importing compiler runtimes."""

from __future__ import annotations

import re
from concurrent.futures import CancelledError

from .models import Failure, FailureCategory, ReportStatus, Stage


class InvalidInputError(ValueError):
    """A request is invalid and must be corrected by the caller."""


def status_for_failure(failure: Failure) -> ReportStatus:
    """Return the terminal report status implied by a failure category."""
    return (
        ReportStatus.CANCELLED
        if failure.category == FailureCategory.CANCELLED
        else ReportStatus.FAILED
    )


def _safe_message(exc: BaseException) -> str:
    """Produce a bounded, single-line exception message with secrets redacted."""
    try:
        message = str(exc)
    except Exception:
        message = type(exc).__name__
    # Only retain the first line. Framework exceptions can contain a graph or traceback.
    message = message.splitlines()[0] if message else type(exc).__name__
    message = re.sub(r"[\x00-\x1f\x7f]", " ", message)
    # Remove local paths and common credential forms before returning diagnostics.
    message = re.sub(r"(?<!\w)(?:[A-Za-z]:[\\/]|/)[^\s,;]+", "<path>", message)
    message = re.sub(
        r"(?i)\b(token|password|secret|api[_-]?key)\s*[=:]\s*[^\s,;]+",
        r"\1=<redacted>",
        message,
    )
    return message[:2048]


def classify_exception(exc: BaseException, stage: Stage | str) -> Failure:
    """Classify known exception types; unknown exceptions remain internal errors.

    Native frameworks are inspected through their exception class hierarchy, avoiding
    an import of Torch, TensorRT, or their shared libraries merely to report a failure.
    Exception text is not used to guess OOM, unsupported-op, or retryability.
    """
    return _classify_exception(exc, Stage(stage), set())


def _classify_exception(exc: BaseException, stage: Stage, visited: set[int]) -> Failure:
    """Classify one exception, following meaningful causes without cycles."""
    visited.add(id(exc))
    hierarchy = {(base.__module__, base.__name__) for base in type(exc).__mro__}
    if isinstance(exc, TimeoutError):
        category = FailureCategory.TIMEOUT
        actions = ["Increase the timeout or reduce the workload."]
    elif isinstance(exc, (CancelledError, KeyboardInterrupt)):
        category = FailureCategory.CANCELLED
        actions = []
    elif isinstance(exc, MemoryError) or any(
        name in ("OutOfMemoryError", "CudaOutOfMemoryError")
        and module.startswith(("torch", "tensorrt"))
        for module, name in hierarchy
    ):
        category = FailureCategory.OUT_OF_MEMORY
        actions = [
            "Reduce input profiles or memory use, or run on a device with more available memory."
        ]
    elif any(
        name == "UnsupportedOperatorException" and module.startswith("torch_tensorrt")
        for module, name in hierarchy
    ):
        category = FailureCategory.UNSUPPORTED_OPERATOR
        actions = [
            "Inspect operator support and allow PyTorch fallback or change the unsupported operation."
        ]
    elif isinstance(exc, InvalidInputError):
        category = FailureCategory.INVALID_INPUT
        actions = ["Correct the inputs or settings identified by the diagnostic."]
    elif isinstance(exc, ImportError):
        category = FailureCategory.UNSUPPORTED_ENVIRONMENT
        actions = [
            "Install a compatible environment using the documented setup instructions."
        ]
    elif isinstance(exc, OSError) and stage in (
        Stage.SAVE,
        Stage.SERIALIZATION,
        Stage.FINALIZATION,
    ):
        category = FailureCategory.ARTIFACT_ERROR
        actions = ["Check output permissions, free space, and artifact integrity."]
    else:
        category = FailureCategory.INTERNAL_ERROR
        actions = [
            "Inspect retained diagnostics before changing or retrying the request."
        ]
        cause = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
        if cause is not None and id(cause) not in visited and len(visited) < 16:
            classified_cause = _classify_exception(cause, stage, visited)
            if classified_cause.category != FailureCategory.INTERNAL_ERROR:
                return classified_cause
    retry_unchanged = category == FailureCategory.CANCELLED
    return Failure(
        stage=stage,
        category=category,
        message=_safe_message(exc),
        retry_unchanged=retry_unchanged,
        retry_after_change=not retry_unchanged,
        suggested_actions=actions,
        exception_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
    )
