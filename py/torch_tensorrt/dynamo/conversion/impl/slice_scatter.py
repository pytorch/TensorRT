"""TensorRT conversion for ``aten.slice_scatter.default``.

Two paths:

* **KV-cache fast path** (``IKVCacheUpdateLayer``) — fires when the input is
  a direct network input, the layer's invariants hold (4-D static shape, write
  on dim 2, ``start + update_len <= s_max``), and the batch dim is static. The
  resulting output is recorded in ``ctx.aliased_io`` so the runtime can bind
  it to the input's device pointer.

* **Fallback** — equivalent to the previous Torch-TRT decomposition: build a
  broadcast index tensor and emit a regular scatter. Used whenever the KV
  constraints fail.

Slice_scatter is intentionally NOT decomposed in the Torch-TRT decomposition
table; this converter is the single place that handles it.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import (
    AliasedOutput,
    AliasKind,
    ConversionContext,
)
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_trt_tensor,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl import select

import tensorrt as trt
from tensorrt import ITensor as TRTTensor

logger = logging.getLogger(__name__)


def _kv_eligible(
    cache_shape: Tuple[int, ...], dim: int, start: int, update_len: int
) -> Tuple[bool, str]:
    """Apply IKVCacheUpdateLayer's invariants.

    Returns (eligible, reason). The reason is non-empty in both cases for logs.
    """
    if any(not isinstance(s, int) or s < 0 for s in cache_shape):
        return False, f"cache shape is dynamic ({cache_shape}); s_max must be static"
    if len(cache_shape) != 4:
        return (
            False,
            f"cache rank is {len(cache_shape)}; KVCacheUpdate requires 4-D [b,d,s_max,h]",
        )
    if dim != 2:
        return False, f"write dim is {dim}; KVCacheUpdate requires dim=2"
    s_max = cache_shape[2]
    if start + update_len > s_max:
        return (
            False,
            f"write_start({start})+update_len({update_len}) > s_max({s_max})",
        )
    return True, f"eligible (s_max={s_max}, write_start={start}, len={update_len})"


def input_binding_name(ctx: ConversionContext, tensor: TRTTensor) -> Optional[str]:
    """If ``tensor`` is a direct network input, return its binding name, else None."""
    for i in range(ctx.net.num_inputs):
        net_input = ctx.net.get_input(i)
        if net_input is tensor or net_input.name == tensor.name:
            return str(net_input.name)
    return None


def emit_kv_cache_update_layer(
    ctx: ConversionContext,
    name: str,
    cache: TRTTensor,
    src: TRTTensor,
    write_indices: TRTTensor,
) -> Optional[TRTTensor]:
    """Lower-level KVCacheUpdate emission given a write_indices ITensor.

    Performs the binding-name lookup, calls ``add_kv_cache_update``, and
    records the aliased output. Returns the layer output ITensor (which is
    a network output, aliased to ``cache``) or None if the cache isn't a
    direct network input or the layer can't be created.

    Validators upstream are expected to have already verified shape /
    dtype / dim invariants; this function trusts its inputs.
    """
    cache_input_name = input_binding_name(ctx, cache)
    if cache_input_name is None:
        logger.debug("KV cache update: skipped — input is not a direct network input")
        return None

    layer = ctx.net.add_kv_cache_update(
        cache, src, write_indices, trt.KVCacheMode.LINEAR
    )
    if layer is None:
        logger.debug("KV cache update: add_kv_cache_update returned None")
        return None
    set_layer_name(layer, "kv_cache_update", name + "_kv_cache_update", SourceIR.ATEN)
    out = layer.get_output(0)

    ctx.aliased_outputs.append(
        AliasedOutput(
            output_tensor=out,
            input_binding_name=cache_input_name,
            kind=AliasKind.KV_CACHE_UPDATE,
        )
    )
    logger.debug(
        "KV cache update: emitted; output %s aliased to input %s",
        out.name,
        cache_input_name,
    )
    return out


def try_emit_kv_cache_update(
    ctx: ConversionContext,
    name: str,
    cache: TRTTensor,
    src: TRTTensor,
    dim: int,
    start: int,
    update_len: int,
) -> Optional[TRTTensor]:
    """Emit IKVCacheUpdateLayer if all constraints are met. None otherwise.

    Shared by the slice_scatter and index_copy converters. ``start`` is the
    constant write position (e.g. ``slice_scatter``'s ``start`` arg or the
    single value from an ``index_copy`` index tensor). The resulting layer
    writes ``update_len`` slots starting at ``start`` per batch element and
    its output is recorded as aliased to the cache input.
    """
    cache_shape = tuple(cache.shape)
    eligible, reason = _kv_eligible(cache_shape, dim, start, update_len)
    if not eligible:
        logger.debug("slice_scatter: KV fast path skipped — %s", reason)
        return None

    batch = cache_shape[0]
    if not isinstance(batch, int) or batch < 0:
        logger.debug(
            "slice_scatter: KV fast path skipped — dynamic batch dim (%s); writeIndices "
            "must be statically sized for now",
            batch,
        )
        return None

    write_indices_np: np.ndarray = np.full((batch,), start, dtype=np.int32)
    write_indices = get_trt_tensor(ctx, write_indices_np, name + "_write_indices")

    return emit_kv_cache_update_layer(ctx, name, cache, src, write_indices)


def slice_scatter(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    src: TRTTensor,
    dim: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: Optional[int] = None,
) -> TRTTensor:
    """Emit either an IKVCacheUpdateLayer (with aliased I/O) or a scatter sequence."""
    rank = len(input.shape)
    dim_size = input.shape[dim]

    if start is None:
        start = 0
    if isinstance(start, int) and start < 0 and isinstance(dim_size, int):
        start = dim_size + start
    if end is None:
        end = dim_size
    if isinstance(end, int) and end < 0 and isinstance(dim_size, int):
        end = dim_size + end
    if step is None:
        step = 1

    # Trivial: full overwrite.
    if (
        isinstance(start, int)
        and isinstance(end, int)
        and isinstance(dim_size, int)
        and start == 0
        and end == dim_size
        and step == 1
    ):
        return src

    if not (isinstance(start, int) and isinstance(end, int) and isinstance(step, int)):
        raise NotImplementedError(
            "slice_scatter with dynamic start/end/step is not yet supported"
        )

    update_len = end - start

    # KV fast path.
    kv_out = try_emit_kv_cache_update(ctx, name, input, src, dim, start, update_len)
    if kv_out is not None:
        return kv_out

    # Fallback: build broadcast indices and scatter.
    indices_np: np.ndarray = np.arange(start, end, step, dtype=np.int64)
    target_shape = [1] * rank
    target_shape[dim] = len(indices_np)
    indices_np = indices_np.reshape(target_shape)
    src_shape = tuple(src.shape)
    indices_np = np.broadcast_to(indices_np, src_shape).astype(np.int64)
    indices_tensor = get_trt_tensor(ctx, indices_np, name + "_fallback_indices")

    return select.scatter(
        ctx,
        target,
        source_ir,
        name + "_fallback_scatter",
        input,
        dim,
        indices_tensor,
        src,
    )
