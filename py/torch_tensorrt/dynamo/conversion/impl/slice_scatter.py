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
import sys
from enum import Enum, auto
from typing import Any, Optional, Tuple

import numpy as np
import tensorrt as trt
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
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
from torch_tensorrt.dynamo.conversion.impl.shape import shape as get_shape
from torch_tensorrt.dynamo.utils import DYNAMIC_DIM

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


class KVWriteStatus(Enum):
    """How ``resolve_slice_scatter_write`` resolved a write's slice bounds.

    Anything but ``OK`` is one of the converter's early exits, reached before it
    can emit an IKVCacheUpdateLayer.
    """

    OK = auto()
    FULL_OVERWRITE = auto()
    DYNAMIC_BOUNDS = auto()
    DYNAMIC_DIM_SIZE = auto()
    BAD_DIM = auto()


# torch.export uses INT64_MAX as "open end". `sys.maxsize` is the same value and is
# what the aten.slice converter matches on, so both are accepted.
_OPEN_END = (sys.maxsize, 2**63 - 1)


def _needs_dim_size(bound: Any) -> bool:
    """Whether ``bound`` has to be resolved against the dim being written: ``None`` and
    an open end run to it, a negative index counts back from it. A non-int bound is
    reported as ``DYNAMIC_BOUNDS`` instead, so it passes here.
    """
    if bound is None:
        return True
    return isinstance(bound, int) and (bound < 0 or bound in _OPEN_END)


def resolve_slice_scatter_write(
    cache_shape: Tuple[Any, ...],
    dim: Any,
    start: Any,
    end: Any,
    step: Any,
) -> Tuple[Optional[int], Optional[int], Optional[int], KVWriteStatus]:
    """Resolve ``slice_scatter``'s slice bounds into the form ``_kv_eligible`` takes.

    Returns ``(start, end, step, status)``. Under ``OK`` and ``FULL_OVERWRITE`` the
    three bounds are Python ``int``s, with the op's defaults filled in, negative
    indices counted from the end and the whole slice clamped to the dim; under the
    other statuses all three are ``None``, since nothing resolved. ``status`` is one
    of:

    * ``OK`` — the bounds are concrete, and the caller goes on to
      ``_kv_eligible(cache_shape, dim, start, end - start)``.
    * ``FULL_OVERWRITE`` — the slice spans the whole dim with a step equal to 1, so
      the converter returns ``src`` and emits no KV layer. ``step`` comes back as the
      integer 1, which is all this branch establishes: the caller's own step may be a
      symbolic object that merely compares equal to it, and no caller reads the value
      under this status.
    * ``DYNAMIC_BOUNDS`` — a bound is not a Python ``int``, so the converter
      raises ``NotImplementedError``.
    * ``DYNAMIC_DIM_SIZE`` — a bound needs the size of the dim it writes (see
      :func:`_needs_dim_size`) and that dim is dynamic. ``aten_ops_slice_scatter``'s
      validator rejects these so the partitioner runs them in PyTorch; the converter
      raises if one reaches it anyway.
    * ``BAD_DIM`` — ``dim`` is either not a Python ``int`` or does not index
      ``cache_shape``; the converter raises ``IndexError`` for both. A
      ``numpy.int64`` is rejected on the type check even when its value is in range.

    Single source of truth for the arguments ``_kv_eligible`` is called with: the
    converter below derives them from the TRT tensors, and the pre-conversion
    predictor in ``lowering/_buffer_lifting.py`` derives them from the fx node. A
    write is classified before conversion as engine-aliased and has its ``copy_``
    dropped on the strength of that prediction, so if the two derivations disagree
    the aliasing never materializes and the write-back is lost.

    ``step`` is read here for the full-overwrite shortcut and, on the converter side,
    only by the scatter fallback; the KV fast path never reads it. See the note on the
    ``OK`` return for what that costs a strided write the KV path accepts.
    """
    if not isinstance(dim, int) or not -len(cache_shape) <= dim < len(cache_shape):
        return None, None, None, KVWriteStatus.BAD_DIM
    dim_size = cache_shape[dim]
    # A dynamic dim reaches the converter as DYNAMIC_DIM (-1) and the predictor as a
    # SymInt; neither is a size, and folding the -1 into an index is how an open-ended
    # write on a dynamic dim used to resolve to an empty index range.
    dim_is_static = isinstance(dim_size, int) and dim_size >= 0

    if start is None:
        start = 0
    if step is None:
        step = 1

    if not dim_is_static and (_needs_dim_size(start) or _needs_dim_size(end)):
        return None, None, None, KVWriteStatus.DYNAMIC_DIM_SIZE

    if end is None:
        end = dim_size
    if dim_is_static:
        if isinstance(start, int) and start < 0:
            start = dim_size + start
        if isinstance(end, int) and end < 0:
            end = dim_size + end
        # Aten clamps a slice to its dim, and so must this: unclamped, the open end
        # reaches the fallback's np.arange as an INT64_MAX-long index range.
        if isinstance(start, int):
            start = min(max(start, 0), dim_size)
        if isinstance(end, int):
            end = min(max(end, 0), dim_size)

    # A slice covering the whole dim is a plain copy of the source whatever `step` is
    # made of, so it is settled before the bounds are required to be concrete: `step`
    # only has to compare equal to 1, which a symbolic step can do.
    if (
        isinstance(start, int)
        and isinstance(end, int)
        and dim_is_static
        and start == 0
        and end == dim_size
        and step == 1
    ):
        return start, end, 1, KVWriteStatus.FULL_OVERWRITE

    if not (isinstance(start, int) and isinstance(end, int) and isinstance(step, int)):
        return None, None, None, KVWriteStatus.DYNAMIC_BOUNDS

    # `step` is passed through untouched, and `try_emit_kv_cache_update` never reads
    # it: a strided write the KV fast path accepts is classified and lowered as if it
    # were contiguous, so `cache[:, :, 0:8:2, :]` lands in slots 0, 1, 2, 3 rather than
    # 0, 2, 4, 6, silently. Known wrong and unfixed. The scatter fallback below does
    # honour `step` (`test_fallback_step_two`), so only the KV path miscompiles.
    return start, end, step, KVWriteStatus.OK


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

    start, end, step, status = resolve_slice_scatter_write(
        tuple(input.shape), dim, start, end, step
    )

    if status is KVWriteStatus.BAD_DIM:
        raise IndexError(
            f"slice_scatter: {dim} of type {type(dim).__name__} is not a valid dim "
            f"for a rank-{rank} input; dim must be a Python int in [-{rank}, {rank})"
        )

    # Trivial: full overwrite.
    if status is KVWriteStatus.FULL_OVERWRITE:
        return src

    if status is KVWriteStatus.DYNAMIC_BOUNDS:
        raise NotImplementedError(
            "slice_scatter with dynamic start/end/step is not yet supported"
        )

    if status is KVWriteStatus.DYNAMIC_DIM_SIZE:
        # The validator keeps these out of TensorRT, so this is only reachable for a
        # node it could not read a shape from.
        raise NotImplementedError(
            f"slice_scatter: dim {dim} of the input is dynamic, so this write's bounds "
            "cannot be resolved without its size"
        )
    # OK is the only status left, and it resolves all three bounds to Python ints.
    assert start is not None and end is not None and step is not None

    update_len = end - start

    # KV fast path, unless the write carries _trt_no_kv_alias: the classifier already
    # filed it copy-back and re-attached its value as a graph output, and a buffer
    # that is both copy-back and engine-aliased raises on every call.
    kv_forbidden = ctx.current_node is not None and ctx.current_node.meta.get(
        "_trt_no_kv_alias"
    )
    kv_out = (
        None
        if kv_forbidden
        else try_emit_kv_cache_update(ctx, name, input, src, dim, start, update_len)
    )
    if kv_out is not None:
        return kv_out

    # Fallback: build broadcast indices and scatter.
    indices_np: np.ndarray = np.arange(start, end, step, dtype=np.int64)
    target_shape = [1] * rank
    target_shape[dim] = len(indices_np)
    indices_np = indices_np.reshape(target_shape)
    src_shape = tuple(src.shape)
    if DYNAMIC_DIM in src_shape:
        indices_tensor = get_trt_tensor(ctx, indices_np, name + "_fallback_indices")
        runtime_src_shape = []
        for axis, size in enumerate(src_shape):
            if axis == dim % rank:
                runtime_src_shape.append(indices_np.shape[axis])
            elif size == DYNAMIC_DIM:
                runtime_src_shape.append(
                    get_shape(
                        ctx,
                        target,
                        source_ir,
                        name + f"_fallback_src_shape_{axis}",
                        src,
                        axis,
                    )
                )
            else:
                runtime_src_shape.append(size)
        indices_tensor = impl.slice.expand(
            ctx,
            target,
            source_ir,
            name + "_fallback_indices_expand",
            indices_tensor,
            tuple(runtime_src_shape),
        )
    else:
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
