# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""A single TensorRT engine held as serialized bytes, a live ``ICudaEngine``, or both."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import tensorrt as trt


class EngineHandle:
    """One TensorRT engine, materialized lazily in whichever form is requested.

    The builder produces a live ``ICudaEngine``; a checkpoint produces bytes.
    Each form is derived from the other on first request and then cached, so a
    freshly compiled engine is never serialized unless something needs the
    bytes (save, export, the C++ runtime), and loaded bytes are deserialized
    only once.

    "The bytes" here are the plain ``ICudaEngine.serialize()`` form. Callers that
    need a different flavor (weight-stripped for the engine cache,
    ``INCLUDE_REFIT`` after refit) serialize explicitly and hand the result back
    via :meth:`refitted` or the constructor.
    """

    __slots__ = ("_serialized", "_cuda_engine", "_engine_checked")

    def __init__(
        self,
        *,
        serialized: Optional[bytes] = None,
        cuda_engine: Optional[trt.ICudaEngine] = None,
    ) -> None:
        if serialized is None and cuda_engine is None:
            raise ValueError("EngineHandle needs serialized bytes or a live engine")
        if serialized is not None and not isinstance(serialized, (bytes, bytearray)):
            raise TypeError("Expected serialized engine as bytes")
        self._serialized: Optional[bytes] = (
            bytes(serialized) if serialized is not None else None
        )
        self._cuda_engine: Optional[trt.ICudaEngine] = cuda_engine
        # Whether ``_cuda_engine`` is known to run in this process. A passed-in
        # engine is checked on first use; one this handle deserialized is.
        self._engine_checked = False

    @classmethod
    def coerce(cls, value: "EngineHandle | bytes | bytearray") -> "EngineHandle":
        """Accept either a handle or raw bytes (the wire format's engine slot)."""
        if isinstance(value, EngineHandle):
            return value
        return cls(serialized=value)

    @property
    def has_cuda_engine(self) -> bool:
        return self._cuda_engine is not None

    @property
    def has_serialized(self) -> bool:
        return self._serialized is not None

    def serialized(self) -> bytes:
        """Engine bytes, serializing the live engine on first request."""
        if self._serialized is None:
            assert self._cuda_engine is not None
            self._serialized = bytes(self._cuda_engine.serialize())
        return self._serialized

    def cuda_engine(self, runtime: trt.Runtime) -> trt.ICudaEngine:
        """Live engine, deserializing with ``runtime`` on first request.

        A passed-in engine is used only if it is this process's
        ``trt.ICudaEngine``. Anything else, such as a bytes-only proxy engine from
        a builder outside this process, cannot run here, so the engine is
        reloaded from its bytes. Callers must have the engine's target device
        current.
        """
        if self._cuda_engine is not None and not self._engine_checked:
            if not isinstance(self._cuda_engine, trt.ICudaEngine):
                self.release_cuda_engine()
        if self._cuda_engine is None:
            engine = runtime.deserialize_cuda_engine(self._serialized)
            if engine is None:
                raise RuntimeError("Unable to deserialize the TensorRT engine")
            self._cuda_engine = engine
        self._engine_checked = True
        return self._cuda_engine

    def release_cuda_engine(self) -> None:
        """Keep only the bytes, so the live engine's device memory can be freed.

        Used once another owner (the C++ runtime) has deserialized its own copy.
        """
        self.serialized()
        self._cuda_engine = None

    def refitted(self, serialized: Optional[bytes] = None) -> None:
        """Record that the live engine was refit in place.

        Cached bytes are stale; ``serialized`` replaces them when the caller
        already has the refit bytes, otherwise they are re-derived on demand.
        """
        if self._cuda_engine is None:
            raise RuntimeError("Cannot mark a bytes-only EngineHandle as refitted")
        self._serialized = bytes(serialized) if serialized is not None else None

    def __deepcopy__(self, memo: dict[int, Any]) -> "EngineHandle":
        # TRT engines are not copyable; copies share the one engine.
        return self

    def __reduce__(self) -> Tuple[Callable[..., "EngineHandle"], Tuple[Any, ...]]:
        return (_engine_handle_from_bytes, (self.serialized(),))

    def __repr__(self) -> str:
        forms = [
            name
            for name, present in (
                ("bytes", self.has_serialized),
                ("live", self.has_cuda_engine),
            )
            if present
        ]
        return f"EngineHandle({'+'.join(forms)})"


def _engine_handle_from_bytes(serialized: bytes) -> EngineHandle:
    return EngineHandle(serialized=serialized)
