# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TensorRT conversion for ``tensorrt::kv_cache_update`` (defined in
``lowering/_kv_cache_update.py``).

* **KV-cache fast path** -- when the cache is a direct network input, an
  ``IKVCacheUpdateLayer`` that takes the per-row positions as its
  ``writeIndices``, its output aliased to the cache input.
* **Fallback** -- otherwise, a scatter of each update row at its own positions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import tensorrt as trt
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_trt_tensor,
)
from torch_tensorrt.dynamo.conversion.impl.slice_scatter import (
    emit_kv_cache_update_layer,
)


def kv_cache_update(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    cache: TRTTensor,
    update: TRTTensor,
    write_indices: TRTTensor,
) -> TRTTensor:
    """``cache`` with ``update[b]`` written from position ``write_indices[b]`` of
    row ``b``. The capability validator has checked the shapes: static, 4-D cache
    and update agreeing on every dim but the sequence one, ``write_indices`` of
    shape ``[b]``."""
    if write_indices.dtype != trt.int32:
        write_indices = cast_trt_tensor(
            ctx,
            write_indices,
            trt.int32,
            name + "_write_indices_i32",
            target,
            source_ir,
        )

    # Not when lift_mutated_buffers filed this write copy-back: its value is then a
    # graph output as well, and a buffer that is both copy-back and engine-aliased
    # raises on every call.
    kv_forbidden = ctx.current_node is not None and ctx.current_node.meta.get(
        "_trt_no_kv_alias"
    )
    if not kv_forbidden:
        out = emit_kv_cache_update_layer(ctx, name, cache, update, write_indices)
        if out is not None:
            return out

    # Fallback: update[b, :, t, :] goes to position write_indices[b] + t.
    b, h, s, d = update.shape
    rows = impl.shuffle.reshape(
        ctx, target, source_ir, name + "_row_starts", write_indices, (b, 1, 1, 1)
    )
    offsets = get_trt_tensor(
        ctx, np.arange(s, dtype=np.int32).reshape(1, 1, s, 1), name + "_offsets"
    )
    positions = impl.elementwise.add(
        ctx, target, source_ir, name + "_positions", rows, offsets
    )
    positions = impl.slice.expand(
        ctx, target, source_ir, name + "_positions_expand", positions, (b, h, s, d)
    )
    return impl.select.scatter(
        ctx, target, source_ir, name + "_scatter", cache, 2, positions, update
    )
