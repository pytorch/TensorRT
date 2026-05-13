"""TensorRT conversion for ``aten.index_copy.default``.

Mirrors the structure of ``slice_scatter`` but the eligibility check is
gated by a validator (declared in ``aten_ops_converters.py``) and there
are two registered converters for the same op:

* ``aten_ops_index_copy_kv`` — HIGH priority, validator-gated. Fires only
  when the input is a 4-D static cache, ``dim=2``, and the source has
  ``shape[2] == 1`` (single-position write — the common
  streaming-decoder pattern). Emits ``IKVCacheUpdateLayer`` whose output
  is aliased to the cache input.

* ``aten_ops_index_copy_fallback`` — STANDARD priority, always fires.
  Implements the general semantics via scatter (equivalent to what the
  torch decomposition would produce).

Since both converters live in TRT, no graph break is introduced for
non-KV cases; they just take a less-efficient TRT path.

The two functions here implement the bodies. Registration with the
validator lives next to the other aten converters for discoverability.
"""

from __future__ import annotations

import logging
from typing import Optional

from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import cast_trt_tensor
from torch_tensorrt.dynamo.conversion.impl import select
from torch_tensorrt.dynamo.conversion.impl.slice_scatter import (
    emit_kv_cache_update_layer,
)

import tensorrt as trt
from tensorrt import ITensor as TRTTensor

logger = logging.getLogger(__name__)


def index_copy_kv(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
    index: TRTTensor,
    src: TRTTensor,
) -> TRTTensor:
    """KV-cache fast path. Caller (the validator) has already verified
    that this case is KV-eligible — 4-D static cache, dim=2, source
    rank=4 with seq dim of size 1.

    ``index`` is the position tensor: shape ``[s_update]``. For the
    eligible case (s_update == 1) this is a single value that we use
    directly as the write start. KVCacheUpdate's ``writeIndices`` arg
    expects shape ``[batch]``, so we broadcast/repeat the single index
    value across the batch dimension.
    """
    cache_shape = tuple(input.shape)
    batch = cache_shape[0]

    # KVCacheUpdate accepts int32 / int64 writeIndices; TRT auto-promotes
    # but be explicit to avoid surprises across version drift.
    if index.dtype != trt.int32:
        index = cast_trt_tensor(ctx, index, trt.int32, name + "_index_to_int32")

    # writeIndices shape must be [batch]. For batch=1 with index shape [1]
    # this is already correct. For batch > 1, broadcast: emit a constant
    # of shape [batch] filled with the single index value at runtime via
    # a gather/expand pattern. The validator's job is to allow only cases
    # where this is well-defined; for now we restrict to batch == 1 (see
    # the validator).
    write_indices = index

    # When batch > 1, we'd need to broadcast `index` (shape [1]) to
    # [batch]. The validator currently keeps us in the batch==1 case.
    if isinstance(batch, int) and batch > 1:
        # Defensive: shouldn't happen if the validator is correct, but
        # fall back rather than emit a wrong layer.
        logger.debug(
            "index_copy_kv: batch > 1 not yet supported for runtime indices; "
            "falling back to scatter"
        )
        return index_copy_fallback(ctx, target, source_ir, name, input, dim, index, src)

    out = emit_kv_cache_update_layer(ctx, name, input, src, write_indices)
    if out is None:
        # KV emission failed (e.g. input not a direct network input);
        # fall through to scatter so correctness is preserved.
        return index_copy_fallback(ctx, target, source_ir, name, input, dim, index, src)
    return out


def index_copy_fallback(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
    index: TRTTensor,
    src: TRTTensor,
) -> TRTTensor:
    """General-purpose ``index_copy``: scatter ``src`` into ``input`` at
    positions in ``index`` along ``dim``. Equivalent to the standard
    torch decomposition: build a broadcast index tensor of the same
    shape as ``src`` with ``index`` placed along ``dim`` and call
    ``scatter``.
    """
    rank = len(input.shape)
    src_shape = tuple(src.shape)

    # Reshape `index` (1-D, length matches src.shape[dim]) so it broadcasts
    # over the remaining dims of `src`.
    reshape_to = [1] * rank
    if isinstance(src_shape[dim], int):
        reshape_to[dim] = src_shape[dim]
    else:
        # Dynamic seq dim — defer to a shape-aware reshape. Build it via
        # a shape op so the reshape is dynamic-shape-safe.
        # For now require static; raise instead of silently producing wrong results.
        raise NotImplementedError(
            "index_copy fallback with dynamic source shape on dim %d is not "
            "yet supported." % dim
        )

    shuffle = ctx.net.add_shuffle(index)
    shuffle.reshape_dims = trt.Dims(reshape_to)
    reshaped_index = shuffle.get_output(0)

    # Broadcast/expand to src's shape. ``broadcast_to`` semantics: numpy
    # array first then to TRT.
    if all(isinstance(s, int) for s in src_shape):
        # Static case: just expand via a slice with broadcast strides.
        expand_layer = ctx.net.add_slice(
            reshaped_index,
            start=tuple(0 for _ in range(rank)),
            shape=src_shape,
            stride=tuple(0 if i != dim else 1 for i in range(rank)),
        )
        index_broadcast = expand_layer.get_output(0)
    else:
        raise NotImplementedError(
            "index_copy fallback with dynamic shapes is not yet supported."
        )

    return select.scatter(
        ctx,
        target,
        source_ir,
        name + "_fallback_scatter",
        input,
        dim,
        index_broadcast,
        src,
    )
