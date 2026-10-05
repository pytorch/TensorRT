# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import contextmanager
from typing import Iterator, Optional

import torch

from ._errors import RegionCaptureError
from ._session import current_session


@contextmanager
def execute_in_torch(*, name: Optional[str] = None) -> Iterator[None]:
    """Keep one captured scope together in PyTorch during Dynamo compilation.

    Outside a Torch-TensorRT-owned compilation this is an eager no-op. The
    initial capture path requires ``strict=True``. The body must still be
    exportable, functional inference code; the scope does not permit graph
    breaks or arbitrary Python execution. Nested regions are rejected by the
    capture normalizer. ``name`` is an optional diagnostic label, not an ID.
    """
    if name is not None and not isinstance(name, str):
        raise TypeError("execute_in_torch name must be a string or None")
    session = current_session()
    if session is None or not torch.compiler.is_compiling():
        yield
        return
    if not session.strict:
        raise RegionCaptureError(
            "execute_in_torch requires torch_tensorrt.compile(..., ir='dynamo', "
            "strict=True); non-strict region capture is not supported"
        )
    if session.capture_error is not None:
        raise RegionCaptureError(session.capture_error)

    # The owning session imported this module before Dynamo started tracing.
    from ._marker_ops import BEGIN, END, NAME_KEY, TOKEN_KEY

    sentinel = BEGIN(name or "")
    # This identity is graph-local capture metadata, never an operator argument,
    # occurrence ID, or persistent cache key. The parser verifies uniqueness and
    # the end's exact sentinel edge before accepting membership.
    annotations = {TOKEN_KEY: id(sentinel), NAME_KEY: name or ""}
    with (
        torch.fx.traceback.annotate(annotations),
        torch._dynamo.error_on_graph_break(True),
    ):
        try:
            yield
        finally:
            END(sentinel)
