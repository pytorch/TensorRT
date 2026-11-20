# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Scope markers that a trace records and compilation consumes.

They are registered when ``torch_tensorrt`` is imported, so that a saved program
holding them can be loaded.
"""

from typing import Optional, Sequence

import torch

_LIBRARY = torch.library.Library("torch_tensorrt_region", "DEF")
_LIBRARY.define(
    'begin(str name, str backend="eager", str route="torch", str plugin="", '
    'str operator="", Tensor[]? inputs=None) -> Tensor'
)
_LIBRARY.define("end(Tensor sentinel) -> ()")


def _begin(
    name: str,
    backend: str = "eager",
    route: str = "torch",
    plugin: str = "",
    operator: str = "",
    inputs: Optional[Sequence[torch.Tensor]] = None,
) -> torch.Tensor:
    # ``inputs`` only records the replacement's argument order for capture;
    # composite dispatch handles the call on any current device.
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
        "Regions (execute_in_torch, fused_op, fused_kernel and regions= rules) "
        "require PyTorch ordered-effect markers, tracing annotations, and "
        f"graph-break guards; this PyTorch version lacks them: {error}. To trace "
        "an annotated model without its regions, use "
        "torch_tensorrt.region.ignore_regions()"
    )
