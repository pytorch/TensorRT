"""TensorRT conversion for ``aten.index_copy.default``.

Mirrors the structure of ``slice_scatter`` but the eligibility check is
gated by a validator (declared in ``aten_ops_converters.py``) and there
are two registered converters for the same op:

* ``aten_ops_index_copy_kv`` — HIGH priority, validator-gated. Fires only
  when the input is a 4-D static cache, ``dim=2``, and the source writes a
  contiguous run of positions — one per decode step, a whole prompt in
  prefill, or a dynamic number of them where one engine serves both. Emits
  ``IKVCacheUpdateLayer`` whose output is aliased to the cache input.

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

import tensorrt as trt
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl import select
from torch_tensorrt.dynamo.conversion.impl.shape import shape as get_shape
from torch_tensorrt.dynamo.conversion.impl.slice_scatter import (
    emit_kv_cache_update_layer,
)
from torch_tensorrt.dynamo.utils import DYNAMIC_DIM

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
    rank=4, and positions that ascend by one from the first.

    ``index`` is the position tensor: shape ``[s_update]``, which may be
    dynamic. KVCacheUpdate's ``writeIndices`` arg is shape ``[batch]`` and
    holds only where the block starts, the layer walking the rest from there,
    so the write start is element 0 whatever ``s_update`` turns out to be.

    ``kLINEAR`` requires ``writeIndices[0] + s_update <= s_max``, which cannot
    be checked here: the start is a runtime value. A single-position write could
    only ever overrun by one slot, but a prefill can overrun by a whole prompt,
    so a caller writing past the end of the cache gets no diagnostic. The
    static equivalent is checked in ``slice_scatter._kv_eligible``.
    """
    cache_shape = tuple(input.shape)
    batch = cache_shape[0]

    # KVCacheUpdate accepts int32 / int64 writeIndices; TRT auto-promotes
    # but be explicit to avoid surprises across version drift.
    if index.dtype != trt.int32:
        index = cast_trt_tensor(ctx, index, trt.int32, name + "_index_to_int32")

    # writeIndices shape must be [batch]. Taking the leading position gives that
    # for batch == 1 whether the write is one position or many -- and it is the
    # slice, not `index` itself, that keeps a multi-position write honest, since
    # index shape [s_update] is the wrong shape to hand the layer. For batch > 1
    # we'd broadcast this value to [batch]; the validator keeps us at 1.
    start_slice = ctx.net.add_slice(index, start=(0,), shape=(1,), stride=(1,))
    set_layer_name(start_slice, target, name + "_write_start", source_ir)
    write_indices = start_slice.get_output(0)

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

    def is_static(extent: int) -> bool:
        # A dynamic extent reaches here as DYNAMIC_DIM, which is the int -1, so
        # testing the type alone lets it through to build a nonsense reshape and
        # the failure then surfaces from inside scatter as "__len__() should
        # return >= 0" with nothing pointing back to this op.
        return bool(isinstance(extent, int) and extent != DYNAMIC_DIM)

    # Reshape `index` (1-D, length matches src.shape[dim]) so it broadcasts
    # over the remaining dims of `src`. A dynamic write length needs no shape
    # tensor here: -1 is also how a shuffle spells "infer this extent".
    reshape_to = [1] * rank
    reshape_to[dim] = src_shape[dim] if is_static(src_shape[dim]) else -1

    shuffle = ctx.net.add_shuffle(index)
    shuffle.reshape_dims = trt.Dims(reshape_to)
    set_layer_name(shuffle, target, name + "_fallback_index_reshape", source_ir)
    reshaped_index = shuffle.get_output(0)

    # Broadcast/expand to src's shape.
    if all(is_static(s) for s in src_shape):
        # Static case: just expand via a slice with broadcast strides.
        expand_layer = ctx.net.add_slice(
            reshaped_index,
            start=tuple(0 for _ in range(rank)),
            shape=src_shape,
            stride=tuple(0 if i != dim else 1 for i in range(rank)),
        )
        set_layer_name(expand_layer, target, name + "_fallback_index_expand", source_ir)
        index_broadcast = expand_layer.get_output(0)
    else:
        # Same shape-aware expand the slice_scatter fallback uses, reading each
        # dynamic extent off `src` at runtime. This path is what keeps a KV write
        # that the layer turned down (see emit_kv_cache_update_layer) falling
        # back rather than failing the build: the validator has already claimed
        # the node by then, so a raise here would abort the compile outright.
        index_broadcast = impl.slice.expand(
            ctx,
            target,
            source_ir,
            name + "_fallback_index_expand",
            reshaped_index,
            tuple(
                (
                    size
                    if is_static(size)
                    else get_shape(
                        ctx,
                        target,
                        source_ir,
                        name + f"_fallback_src_shape_{axis}",
                        src,
                        axis,
                    )
                )
                for axis, size in enumerate(src_shape)
            ),
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
