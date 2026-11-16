# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Job-scoped event emission with explicit persistence failure semantics."""

from __future__ import annotations

import contextlib
import threading
from collections import deque
from contextvars import ContextVar
from typing import Any, Callable, Iterator, Protocol, cast

from .models import Event, EventKind, Stage


class EventSink(Protocol):
    def __call__(self, event: Event) -> None:
        """Persist or forward one validated event."""
        ...


class EventPersistenceError(RuntimeError):
    """A sink explicitly marked critical could not persist an event."""


class EventEmitter:
    """An emitter owned by one job; sequence assignment and callbacks are serialized.

    Optional callbacks cannot fail the operation. Their exception types are recorded
    in ``callback_failures``. A durable sink may set ``critical = True`` to request
    that persistence errors raise ``EventPersistenceError`` rather than disappear.
    Native/thread callbacks should hold and call this emitter explicitly; Python
    ContextVars do not automatically propagate to new threads.
    """

    def __init__(
        self,
        job_id: str = "",
        sink: EventSink | Callable[[Event], None] | None = None,
        *,
        max_events: int = 10000,
    ) -> None:
        """Create a job emitter with optional bounded history and event sink."""
        if type(max_events) is not int or max_events < 0:
            raise ValueError("max_events must be a nonnegative integer")
        self.job_id = job_id
        self.sink = sink
        self._events: deque[Event] = deque(maxlen=max_events)
        self.dropped_events = 0
        self.callback_failures: list[str] = []
        self.callback_failure_count = 0
        self._sequence = 0
        self._lock = threading.RLock()

    @property
    def events(self) -> list[Event]:
        """A bounded snapshot; sinks receive all events including evicted history."""
        with self._lock:
            return list(self._events)

    def emit(
        self,
        stage: Stage | str,
        kind: EventKind | str,
        stage_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> Event:
        """Validate, snapshot, sequence, retain, and dispatch one event."""
        with self._lock:
            next_sequence = self._sequence + 1
            candidate = Event(
                sequence=next_sequence,
                job_id=self.job_id,
                stage=Stage(stage),
                kind=EventKind(kind),
                stage_id=stage_id,
                details=details if details is not None else {},
            )
            # Validate and snapshot nested JSON data before retaining or handing the
            # event to another component. Commit the sequence only after validation.
            event = cast(Event, Event.from_dict(candidate.to_dict()))
            self._sequence = next_sequence
            if len(self._events) == self._events.maxlen:
                self.dropped_events += 1
            self._events.append(event)
            if self.sink is not None:
                try:
                    self.sink(event)
                except Exception as exc:
                    self.callback_failure_count += 1
                    exception_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
                    if (
                        exception_type not in self.callback_failures
                        and len(self.callback_failures) < 32
                    ):
                        self.callback_failures.append(exception_type)
                    if getattr(self.sink, "critical", False):
                        raise EventPersistenceError(
                            "The critical event sink could not persist an event"
                        ) from exc
            return event


_EMITTER: ContextVar[EventEmitter | None] = ContextVar(
    "torch_tensorrt_event_emitter", default=None
)


@contextlib.contextmanager
def event_context(
    job_id: str = "", sink: EventSink | None = None
) -> Iterator[EventEmitter]:
    """Install a context-local emitter and restore the previous one on exit."""
    emitter = EventEmitter(job_id, sink)
    token = _EMITTER.set(emitter)
    try:
        yield emitter
    finally:
        _EMITTER.reset(token)


def current_emitter() -> EventEmitter | None:
    """Return the emitter installed in the current context, if any."""
    return _EMITTER.get()


def emit_event(
    stage: Stage | str, kind: EventKind | str, stage_id: str = "", **details: Any
) -> Event | None:
    """Emit through the current context, or do nothing when none is installed."""
    emitter = current_emitter()
    if emitter is None:
        return None
    return emitter.emit(stage, kind, stage_id, details)
