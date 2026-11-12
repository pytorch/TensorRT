# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    flatten_dims,
    get_axes_for_reduce_op,
    get_positive_dim,
    get_trt_tensor,
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl.elementwise import convert_binary_elementwise
from torch_tensorrt.dynamo.conversion.impl.shape import shape as get_shape
from torch_tensorrt.dynamo.utils import DYNAMIC_DIM


def argmax_argmin(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    topk_option: trt.TopKOperation,
    dim: Optional[int],
    keep_dim: bool = False,
) -> TRTTensor:
    if input.dtype == trt.int32:
        input = cast_trt_tensor(ctx, input, trt.float32, name, target, source_ir)
    if len(input.shape) == 0:
        # TensorRT cannot run topk on a 0-d tensor. As shape (1,) along that dim the
        # result is 0-d, which PyTorch returns whatever keep_dim says.
        input = impl.shuffle.reshape(ctx, target, source_ir, f"{name}_1d", input, (1,))
        dim, keep_dim = 0, False

    # Three different cases here:
    # 1. dim == None, flatten input tensor first, keep_dim will be ignore and the output rank == input rank
    # 2. input rank == 1: TopK layer does not support 1 dimensional topk operation. Broadcast input to rank == 2
    # 3. normal cases, no additional handlings
    out = input
    is_dynamic_present = has_dynamic_shape(input.shape)

    if dim is None:
        if is_dynamic_present and len(input.shape) != 1:
            multiplier = get_trt_tensor(ctx, 1, name + "_shape")
            for i in range(0, len(input.shape)):
                if input.shape[i] != DYNAMIC_DIM:
                    multiplier = convert_binary_elementwise(
                        ctx,
                        target,
                        source_ir,
                        name + f"_shape_{i}",
                        trt.ElementWiseOperation.PROD,
                        multiplier,
                        input.shape[i],
                    )
                else:
                    multiplier = convert_binary_elementwise(
                        ctx,
                        target,
                        source_ir,
                        name + f"_shape_{i}",
                        trt.ElementWiseOperation.PROD,
                        multiplier,
                        get_shape(
                            ctx,
                            target,
                            source_ir,
                            name + f"_shape_dim_stop_{i}",
                            input,
                            i,
                        ),
                    )
            # form shape tensor
            new_shape_layer = ctx.net.add_concatenation(
                [multiplier, get_trt_tensor(ctx, 1, name + "_one_shape")]
            )
            set_layer_name(
                new_shape_layer, target, name + "_new_shape_concat", source_ir
            )
            concat_tensor = new_shape_layer.get_output(0)

            reshape_dynamic_layer = ctx.net.add_shuffle(input)
            reshape_dynamic_layer.set_input(1, concat_tensor)
            set_layer_name(
                reshape_dynamic_layer, target, name + "_reshape_layer", source_ir
            )
            out = reshape_dynamic_layer.get_output(0)

        else:
            new_shape = (*flatten_dims(input, 0, -1), 1)
            out = impl.shuffle.reshape(
                ctx, target, source_ir, f"{name}_flatten", input, new_shape
            )
    elif len(input.shape) == 1:
        new_shape = (*input.shape, 1)
        out = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_broadcast", input, new_shape
        )

    # Reduce over the flattened input if the dimension is None, otherwise the specified dimension
    reduce_mask = get_axes_for_reduce_op(
        get_positive_dim(dim if dim is not None else 0, len(out.shape))
    )

    topk_layer = ctx.net.add_topk(out, topk_option, 1, reduce_mask)
    set_layer_name(topk_layer, target, name, source_ir)

    out = topk_layer.get_output(1)

    if dim is None:
        new_shape = ((1,) * len(input.shape)) if keep_dim else ()
        out = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_unflatten", out, new_shape
        )
    elif len(input.shape) == 1:
        out = impl.squeeze.squeeze(
            ctx,
            target,
            source_ir,
            f"{name}_squeeze",
            out,
            1 if keep_dim else (0, 1),
        )
    elif not keep_dim:
        out = impl.squeeze.squeeze(ctx, target, source_ir, f"{name}_squeeze", out, dim)

    return out


def argmax(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Optional[int],
    keep_dim: bool = False,
) -> TRTTensor:
    return argmax_argmin(
        ctx, target, source_ir, name, input, trt.TopKOperation.MAX, dim, keep_dim
    )


def argmin(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Optional[int],
    keep_dim: bool = False,
) -> TRTTensor:
    return argmax_argmin(
        ctx, target, source_ir, name, input, trt.TopKOperation.MIN, dim, keep_dim
    )


def sort(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
    descending: bool,
    return_indices: bool = True,
) -> Union[TRTTensor, Tuple[TRTTensor, TRTTensor]]:
    dim = get_positive_dim(dim, len(input.shape))
    k = input.shape[dim]
    return topk(
        ctx,
        target,
        source_ir,
        name,
        input,
        k,
        dim,
        descending,
        sorted=None,
        return_indices=return_indices,
    )


def topk(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    k: int,
    dim: int,
    largest: bool,
    sorted: Optional[bool],
    return_indices: bool = True,
) -> Union[TRTTensor, Tuple[TRTTensor, TRTTensor]]:
    # ITopKLayer requires rank >= 2 (same broadcast as argmax_argmin above). A 0-d input
    # is one element, and its results are 0-d again.
    output_shape = (-1,) if len(input.shape) == 1 else ()
    low_rank = len(input.shape) <= 1
    if low_rank:
        input = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_broadcast", input, (-1, 1)
        )
        dim = 0

    topk_layer = ctx.net.add_topk(
        input,
        trt.TopKOperation.MAX if largest else trt.TopKOperation.MIN,
        k,
        get_axes_for_reduce_op(get_positive_dim(dim, len(input.shape))),
    )

    # TensorRT ITopKLayer does not have a sorted flag, it is always returning the sorted topk elements
    # so here no matter sorted is True or False the returned the topk Tensor object is always sorted
    set_layer_name(topk_layer, target, f"{name}_topk", source_ir)

    values = topk_layer.get_output(0)
    indices = topk_layer.get_output(1)

    if low_rank:
        values = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_values_1d", values, output_shape
        )
        indices = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_indices_1d", indices, output_shape
        )

    if return_indices:
        return values, indices
    else:
        return values


# Counts keys tile by tile with a float16 matrix multiply. Every count is at most the tile
# size, and float16 holds whole numbers exactly up to 2048, so the counts are exact.
_COUNTING_SORT_TILE = 128


def counting_sort(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
    descending: bool,
    num_keys: int,
) -> Tuple[TRTTensor, TRTTensor]:
    """Stable sort along ``dim`` of one row of integer keys that all lie in ``[0, num_keys)``.

    Every other dim must have size 1. ITopKLayer sorts at most 3840 elements and does not
    promise to keep equal keys in input order. A counting sort sorts any length and keeps equal
    keys in order: an element goes to the position given by the number of smaller keys plus the
    number of equal keys before it. It builds tensors of ``num_keys`` elements per input
    element, padded to whole tiles, so its cost grows with ``num_keys``.
    """
    rank = len(input.shape)
    dim = get_positive_dim(dim, rank)
    length = input.shape[dim]
    if length == DYNAMIC_DIM:
        length = get_shape(ctx, target, source_ir, f"{name}_length", input, dim)
    row = impl.shuffle.reshape(ctx, target, source_ir, f"{name}_row", input, [-1])
    keys = cast_trt_tensor(ctx, row, trt.int32, f"{name}_keys", target, source_ir)
    if descending:
        keys = impl.elementwise.sub(
            ctx, target, source_ir, f"{name}_descending", num_keys - 1, keys
        )

    # Pad the row to whole tiles with a key that matches no bucket.
    tile = _COUNTING_SORT_TILE
    if isinstance(length, int):
        pad = -length % tile
    else:
        num_tiles = impl.elementwise.floor_divide(
            ctx,
            target,
            source_ir,
            f"{name}_num_tiles",
            impl.elementwise.add(
                ctx, target, source_ir, f"{name}_round_up", length, tile - 1
            ),
            tile,
        )
        pad = impl.elementwise.sub(
            ctx,
            target,
            source_ir,
            f"{name}_pad_length",
            impl.elementwise.mul(
                ctx, target, source_ir, f"{name}_padded_length", num_tiles, tile
            ),
            length,
        )
    padded = impl.pad.constant_padNd(
        ctx, target, source_ir, f"{name}_pad", keys, [0, pad], num_keys
    )
    tiles = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_tiles", padded, [-1, tile]
    )

    # one_hot[g, t, k] is 1 where element t of tile g has key k.
    buckets = get_trt_tensor(
        ctx, np.arange(num_keys, dtype=np.int32), f"{name}_buckets"
    )
    one_hot = impl.elementwise.eq(
        ctx,
        target,
        source_ir,
        f"{name}_one_hot",
        impl.unsqueeze.unsqueeze(
            ctx, target, source_ir, f"{name}_tiles_unsqueeze", tiles, 2
        ),
        buckets,
    )
    one_hot = cast_trt_tensor(
        ctx, one_hot, trt.float16, f"{name}_one_hot_half", target, source_ir
    )
    # A running count down each tile is a product with a lower triangle of ones. A cumulative
    # layer along the whole dim gives the same counts but runs several times slower.
    lower = get_trt_tensor(
        ctx, np.tril(np.ones((tile, tile), dtype=np.float16)), f"{name}_lower"
    )
    in_tile = impl.matmul.matrix_multiply(
        ctx, target, source_ir, f"{name}_in_tile", lower, one_hot
    )
    in_tile = cast_trt_tensor(
        ctx, in_tile, trt.int32, f"{name}_in_tile_int", target, source_ir
    )

    # Add the keys counted in earlier tiles and the smaller keys of the whole row.
    tile_counts = impl.select.select(
        ctx, target, source_ir, f"{name}_tile_counts", in_tile, 1, tile - 1
    )
    earlier_tiles = impl.elementwise.sub(
        ctx,
        target,
        source_ir,
        f"{name}_earlier_tiles",
        impl.slice.cumsum(
            ctx, target, source_ir, f"{name}_tile_cumsum", tile_counts, 0, torch.int32
        ),
        tile_counts,
    )
    totals = impl.reduce.sum(
        ctx, target, source_ir, f"{name}_totals", tile_counts, 0, True
    )
    smaller_keys = impl.elementwise.sub(
        ctx,
        target,
        source_ir,
        f"{name}_smaller_keys",
        impl.slice.cumsum(
            ctx, target, source_ir, f"{name}_key_cumsum", totals, 1, torch.int32
        ),
        totals,
    )
    offsets = impl.elementwise.add(
        ctx, target, source_ir, f"{name}_offsets", earlier_tiles, smaller_keys
    )
    counts = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_counts",
        in_tile,
        impl.unsqueeze.unsqueeze(
            ctx, target, source_ir, f"{name}_offsets_unsqueeze", offsets, 1
        ),
    )

    # Element i reads its count at its own key, at i * num_keys + key of the flat counts. The
    # padding comes after every element, so it is never read.
    iota = impl.arange.arange(
        ctx, target, source_ir, f"{name}_iota", 0, length, 1, torch.int32
    )
    own_index = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_own_index",
        impl.elementwise.mul(
            ctx, target, source_ir, f"{name}_row_start", iota, num_keys
        ),
        keys,
    )
    flat_counts = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_flat_counts", counts, [-1]
    )
    positions = impl.elementwise.sub(
        ctx,
        target,
        source_ir,
        f"{name}_positions",
        impl.select.gather(
            ctx, target, source_ir, f"{name}_own_count", flat_counts, 0, own_index
        ),
        1,
    )

    # Element i goes to positions[i]. The positions are a permutation, so the scatter
    # writes every element and its data input only provides the shape.
    order = impl.select.scatter(
        ctx, target, source_ir, f"{name}_order", iota, 0, positions, iota
    )
    values = impl.select.gather(ctx, target, source_ir, f"{name}_values", row, 0, order)
    indices = cast_trt_tensor(
        ctx, order, trt.int64, f"{name}_indices", target, source_ir
    )
    shape = [1] * rank
    shape[dim] = length
    return (
        impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_values_shape", values, shape
        ),
        impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_indices_shape", indices, shape
        ),
    )
