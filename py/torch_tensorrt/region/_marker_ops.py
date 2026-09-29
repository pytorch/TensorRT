# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Temporary scope markers, consumed before ordinary graph lowering.

This module is imported by the owned compilation session before tracing. The
private effect/annotation interfaces are required only when a region is used.
Their absence must not prevent compilation of an unannotated model.
"""

from typing import Optional

import torch

TOKEN_KEY = "torch_tensorrt_region_token"
NAME_KEY = "torch_tensorrt_region_name"

_LIBRARY = torch.library.Library("torch_tensorrt_region", "DEF")
_LIBRARY.define("begin(str name) -> Tensor")
_LIBRARY.define("end(Tensor sentinel) -> ()")


def _begin(name: str) -> torch.Tensor:
    # No tensor arguments means CPU-only dispatch is insufficient. A Composite
    # implementation also prevents dependence on the model's current CUDA device.
    return torch.empty((), dtype=torch.uint8, device="cpu")


def _end(sentinel: torch.Tensor) -> None:
    return None


_LIBRARY.impl("begin", _begin, "CompositeExplicitAutograd")
_LIBRARY.impl("end", _end, "CompositeExplicitAutograd")
torch.library.register_fake("torch_tensorrt_region::begin")(_begin)
torch.library.register_fake("torch_tensorrt_region::end")(_end)

BEGIN = torch.ops.torch_tensorrt_region.begin.default
END = torch.ops.torch_tensorrt_region.end.default

CAPTURE_ERROR: Optional[str] = None
try:
    from torch._library.effects import EffectType

    register_effect = torch.library._register_effectful_op
    annotate = torch.fx.traceback.annotate
    error_on_graph_break = torch._dynamo.error_on_graph_break
    register_effect(BEGIN, EffectType.ORDERED, lib=_LIBRARY)
    register_effect(END, EffectType.ORDERED, lib=_LIBRARY)
except (ImportError, AttributeError, TypeError) as error:
    CAPTURE_ERROR = (
        "execute_in_torch requires PyTorch ordered-effect markers, tracing "
        f"annotations, and graph-break guards; this PyTorch version lacks them: {error}"
    )
