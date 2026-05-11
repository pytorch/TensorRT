"""
Stream management ops for the torch_tensorrt FX-graph stream-plan system.

Schemas and eager CUDA implementations for the four stream-control ops
(enter_compute_stream, exit_compute_stream, set_stream, sync_streams) plus
the call_trt_with_token engine-call op live in core/runtime/stream_ops.cpp
and register when libtorchtrt.so loads.

This module:
  - registers EffectType.ORDERED so AOT Autograd threads effect tokens through
    the four stream-control ops, preserving order across compiler passes
  - exposes call_trt_with_token under a stable Python name for FX rewrites

call_trt_with_token is a *real dispatcher op* (not a HOP).  Its first
argument is the int token produced by the preceding set_stream or
sync_streams node, creating a hard data-flow edge that any scheduler that
respects data flow — including Inductor and AOTI codegen — must honor:

    set_stream / sync_streams  →  call_trt_with_token

The engine argument is a torch.classes.tensorrt.Engine (TorchBind class)
passed via a get_attr node.  AOTI handles this op like any other custom op
with TorchBind-typed args: it emits a C++ dispatcher call into the loaded
.pt2's resolved torchbind constants.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch._higher_order_ops.effects import _EffectType, _register_effectful_op


def _maybe_get_op(op_path: str) -> Optional[Any]:
    """Return the dispatcher op at op_path, or None if not (yet) registered.

    The schemas live in core/runtime/stream_ops.cpp and register when
    libtorchtrt.so loads.  When this Python module is imported before the
    .so (or before a freshly-rebuilt .so picks up new ops), accessing the
    op through torch.ops raises AttributeError.  We tolerate that here so
    importing the package never crashes during a partial install, and let
    apply_stream_plan surface a clear error at use time.
    """
    parts = op_path.split(".")
    obj: Any = torch.ops
    try:
        for p in parts:
            obj = getattr(obj, p)
        return obj
    except (AttributeError, RuntimeError):
        return None


# Stable Python alias for the dispatcher op.  The C++ schema in
# core/runtime/stream_ops.cpp is:
#   call_trt_with_token(int token,
#                       __torch__.torch.classes.tensorrt.Engine engine,
#                       Tensor[] inputs) -> Tensor[]
call_trt_with_token = _maybe_get_op("tensorrt.call_trt_with_token.default")


# ── Effect registration ───────────────────────────────────────────────────────
# EffectType.ORDERED causes AOT Autograd to wrap each op with with_effects(),
# threading dummy token tensors between consecutive ORDERED ops so the
# compiler sees a real data-dependency chain and cannot reorder them.
# It also makes Node.is_impure() return True via _get_effect(), so FX DCE
# will not remove these nodes even when their outputs have no consumers.
#
# call_trt_with_token does NOT need ORDERED — the int token first-arg gives
# it a real data dependency on the preceding stream-control op, and the
# engine call's Tensor[] outputs flow into downstream ops.

for _op_path in [
    "tensorrt.enter_compute_stream.default",
    "tensorrt.exit_compute_stream.default",
    "tensorrt.set_stream.default",
    "tensorrt.sync_streams.default",
]:
    _op = _maybe_get_op(_op_path)
    if _op is not None:
        _register_effectful_op(_op, _EffectType.ORDERED)
