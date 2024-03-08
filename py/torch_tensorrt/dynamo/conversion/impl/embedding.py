import functools
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch_tensorrt.dynamo.conversion.impl as impl
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    append,
    get_trt_tensor,
    set_item,
    to_numpy,
)
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTTensor


def embedding(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: TRTTensor,
    padding_idx: int,
    scale_grad_by_freq: bool,
    sparse: bool,
) -> TRTTensor:
    indices_tensor = input
    embedding_tensor = weight
    if isinstance(indices_tensor, torch.Tensor) and indices_tensor.dtype == torch.int64:
        raise RuntimeError(
            "The `embedding` op has indices_tensor dtype=int64. This is incorrect since it has to be int32 to run on TRT."
        )
    indices_tensor = get_trt_tensor(ctx, indices_tensor, f"{name}_indices_tensor")
    embedding_tensor = get_trt_tensor(ctx, embedding_tensor, f"{name}_embedding_tensor")
    # unsupported parameters
    # ignore padding_idx, scale_grad_by_freq, and sparse
    # since they are meaningful for training only

    # useful for training only
    # if scale_grad_by_freq:
    #     raise RuntimeError(
    #         "Currently we don't support scale gradient by word frequency."
    #     )

    # if sparse:
    #     raise RuntimeError("Currently we don't support sparse gradient.")

    # Implement embedding lookup with gather layer
    gather_layer = ctx.net.add_gather(embedding_tensor, indices_tensor, axis=0)
    set_layer_name(gather_layer, target, f"{name}_gather", source_ir)
    return gather_layer.get_output(0)


def embedding_bag_with_traversable_offsets(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    embed: TRTTensor,
    offsets_list: Union[torch.Tensor, np.ndarray, Sequence[int]],
    mode: int,
    include_last_offset: bool,
) -> Tuple[TRTTensor, TRTTensor, TRTTensor, TRTTensor]:
    reduce_name = ""
    if mode == 0:  # sum
        reduce_op = functools.partial(
            impl.reduce.sum, ctx=ctx, target=target, source_ir=source_ir
        )
        reduce_name = "sum"
    elif mode == 1:  # mean
        reduce_op = functools.partial(
            impl.reduce.mean, ctx=ctx, target=target, source_ir=source_ir
        )
        reduce_name = "mean"
    elif mode == 2:  # max
        reduce_op = functools.partial(
            impl.reduce.max,
            ctx=ctx,
            target=target,
            source_ir=source_ir,
            return_indices=False,
        )
        reduce_name = "max"

    offsets: np.ndarray = to_numpy(offsets_list)
    len_embed = embed.shape[0]

    if include_last_offset:
        # modify the last index of offsets to the end index
        # however, pytorch doc says if `include_last_offset` is True, the size of offsets
        # is equal to the number of bags + 1. The last element is the size of the input,
        # or the ending index position of the last bag (sequence).
        offsets.itemset(-1, len_embed)
    else:
        # add the end index to offsets
        offsets = np.append(offsets, len_embed)

    zero_tensor = get_trt_tensor(
        ctx, np.zeros((1, embed.shape[1]), dtype=np.float32), f"{name}_zero_tensor"
    )

    # separately reduce embeddings for different bags
    reduced_embed_bags = []
    len_offsets = offsets.shape[0]
    for i in range(len_offsets - 1):
        if offsets[i] < offsets[i + 1]:
            sliced_embed = impl.slice.slice_op(
                ctx,
                target,
                source_ir,
                f"{name}_slice_embed_{i}",
                embed,
                0,
                int(offsets[i]),
                int(offsets[i + 1]),
                1,
            )
            reduced_one_bag = reduce_op(
                name=f"{name}_{reduce_name}_{i}",
                input_val=sliced_embed,
                dim=0,
                keepdim=True,
            )
            reduced_embed_bags.append(reduced_one_bag)
        else:  # offsets[i] == offsets[i + 1]
            reduced_embed_bags.append(zero_tensor)

    out = impl.cat.cat(ctx, target, source_ir, f"{name}_cat", reduced_embed_bags, 0)
    return out, None, None, None


def embedding_bag_with_ITensor_offsets(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    embed: TRTTensor,
    offsets: TRTTensor,
    mode: int,
    include_last_offset: bool,
) -> Tuple[TRTTensor, TRTTensor, TRTTensor, TRTTensor]:
    len_embed = embed.shape[0]

    if include_last_offset:
        # modify the last index of offsets to the end index
        # however, pytorch doc says if `include_last_offset` is True, the size of offsets
        # is equal to the number of bags + 1. The last element is the size of the input,
        # or the ending index position of the last bag (sequence).
        offsets = set_item(
            ctx, target, source_ir, f"{name}_set_item", offsets, -1, len_embed
        )
    else:
        # add the end index to `offsets`
        offsets = append(ctx, target, source_ir, f"{name}_append", offsets, len_embed)

    reduced_embed_bags = []
    # get the first item in offsets
    start = ctx.net.add_gather(
        offsets, get_trt_tensor(ctx, 0, f"{name}_tensor_0"), 0
    ).get_output(0)

    # create a placeholder tensor, whose shape is the same as an embedding
    # if mode is 0 (sum) or 1 (mean), the placeholder tensor is filled with zeros
    # if mode is 2 (max), the placeholder tensor is filled with negative infinity
    zero_tensor = get_trt_tensor(
        ctx, np.zeros((1, embed.shape[1]), dtype=np.float32), f"{name}_zero_tensor"
    )
    placeholder_tensor = (
        get_trt_tensor(
            ctx,
            np.full((1, embed.shape[1]), -np.inf, dtype=np.float32),
            f"{name}_negative_inf_tensor",
        )
        if mode == 2
        else zero_tensor
    )

    # create a list of constant ITensor for reuse
    incremental_tensor_list = []
    for i in range(0, len_embed):
        incremental_tensor_list.append(
            get_trt_tensor(ctx, i, f"incremental_tensor_{i}")
        )

    # traverse offsets to calculate the embedding of each bag
    for i in range(1, offsets.shape[0]):
        end = ctx.net.add_gather(offsets, incremental_tensor_list[i], 0).get_output(0)

        one_bag_list = []
        # traverse the constant list to see if the index is in the range of the current bag
        for j in range(0, len_embed):
            j_tensor = incremental_tensor_list[j]

            # create a TRT conditional layer
            conditional_layer = ctx.net.add_if_conditional()
            # two conditions
            cond1 = impl.elementwise.ge(
                ctx, target, source_ir, f"{name}_ge_{i}_{j}", j_tensor, start
            )
            cond2 = impl.elementwise.lt(
                ctx, target, source_ir, f"{name}_lt_{i}_{j}", j_tensor, end
            )
            condition = impl.elementwise.logical_and(
                ctx, target, source_ir, f"{name}_and_{i}_{j}", cond1, cond2
            )
            condition = impl.shuffle.reshape(
                ctx,
                target,
                source_ir,
                f"{name}_reshape_condition_{i}_{j}",
                condition,
                [],
            )
            # set the combined condition to the conditional layer
            conditional_layer.set_condition(condition)
            # if true, run this subgraph
            one_piece_embed = impl.select.index(
                ctx, target, source_ir, f"{name}_index_{i}_{j}", embed, [j_tensor]
            )
            true_sg = conditional_layer.add_input(one_piece_embed)
            # if false, run this subgraph
            false_sg = conditional_layer.add_input(placeholder_tensor)

            cond_output_layer = conditional_layer.add_output(
                true_sg.get_output(0), false_sg.get_output(0)
            )
            one_bag_list.append(cond_output_layer.get_output(0))

        # concat the one_bag_list along the first dimension
        one_bag = impl.cat.cat(
            ctx,
            target,
            source_ir,
            f"{name}_concat_bag{i}",
            one_bag_list,
            dim=0,
        )

        # reduce the one_bag along the first dimension, the result of which is an embedding of each bag
        if mode == 0:  # sum
            reduced_one_bag = impl.reduce.sum(
                ctx,
                target,
                source_ir,
                name=f"{name}_sum_bag{i}",
                input_val=one_bag,
                dim=0,
                keepdim=True,
            )

        # Since one_bag includes many zeros, directly calculating mean will cause results incorrect
        elif mode == 1:  # mean
            reduced_one_bag = impl.reduce.sum(
                ctx,
                target,
                source_ir,
                name=f"{name}_sum_bag{i}",
                input_val=one_bag,
                dim=0,
                keepdim=True,
            )
            diff = impl.elementwise.sub(
                ctx, target, source_ir, f"{name}_diff_bag{i}", end, start
            )
            reduced_one_bag = impl.elementwise.div(
                ctx, target, source_ir, f"{name}_div_bag{i}", reduced_one_bag, diff
            )

        elif mode == 2:  # max
            reduced_one_bag = impl.reduce.max(
                ctx,
                target,
                source_ir,
                name=f"{name}_max_bag{i}",
                input_val=one_bag,
                dim=0,
                keepdim=True,
                return_indices=False,
            )

        # create a TRT conditional layer
        conditional_layer = ctx.net.add_if_conditional()
        # two conditions
        condition = impl.elementwise.eq(
            ctx, target, source_ir, f"{name}_eq_{i}", start, end
        )
        condition = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_reshape_condition_eq_{i}", condition, []
        )
        # set the combined condition to the conditional layer
        conditional_layer.set_condition(condition)
        # if true, run this subgraph
        true_sg = conditional_layer.add_input(zero_tensor)
        # if false, run this subgraph
        false_sg = conditional_layer.add_input(reduced_one_bag)

        reduced_one_bag_layer = conditional_layer.add_output(
            true_sg.get_output(0), false_sg.get_output(0)
        )

        reduced_embed_bags.append(reduced_one_bag_layer.get_output(0))
        start = end

    # concat the reduced_embed_bags along the first dimension
    out = impl.cat.cat(ctx, target, source_ir, f"{name}_cat", reduced_embed_bags, 0)
    return out, None, None, None


def embedding_bag(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    weight: TRTTensor,
    indices: TRTTensor,
    offsets: TRTTensor,
    scale_grad_by_freq: bool,
    mode: int,
    sparse: bool,
    per_sample_weights: Optional[TRTTensor],  # for sum mode only
    include_last_offset: bool,
    padding_idx: int,
) -> Tuple[TRTTensor, TRTTensor, TRTTensor, TRTTensor]:
    """
    This function is for calculating embedding bags.

    In PyTorch, `offsets` is only used when input is 1D. If input is 2D of shape (B, N),
    it will be treated as B bags (sequences) each of fixed length N, and this will return
    B values aggregated in a way depending on the mode. `offsets` is ignored and required
    to be None in this case.

    However, according to the schema, `offsets` is required for input with any dimensions.
    Accordingly, this function flattens N-D input to 1D and then to calculate embedding bags.
    """

    # TODO: support 2D inputs
    # indices = impl.shuffle.reshape(ctx, target, source_ir, f"{name}_reshape_indices", indices, (-1,))

    # calculate embedding
    embed = embedding(
        ctx,
        target,
        source_ir,
        f"{name}_embedding",
        indices,
        weight,
        padding_idx,
        scale_grad_by_freq,
        sparse,
    )

    # give weights to embedding
    if per_sample_weights is not None:
        assert (
            per_sample_weights.shape == indices.shape
        ), f"`per_sample_weights` (shape: {per_sample_weights.shape}) must have exactly the same shape as indices/input (shape: {indices.shape})!"
        per_sample_weights = get_trt_tensor(
            ctx, per_sample_weights, f"{name}_per_sample_weights", np.float32
        )
        per_sample_weights = impl.shuffle.reshape(
            ctx,
            target,
            source_ir,
            f"{name}_reshape_per_sample_weights",
            per_sample_weights,
            (-1, 1),
        )
        embed = impl.elementwise.mul(
            ctx,
            target,
            source_ir,
            f"{name}_mul_per_sample_weights",
            embed,
            per_sample_weights,
        )

    if isinstance(offsets, TRTTensor):
        return embedding_bag_with_ITensor_offsets(
            ctx, target, source_ir, name, embed, offsets, mode, include_last_offset
        )
    else:
        # this branch has less time complexity
        return embedding_bag_with_traversable_offsets(
            ctx, target, source_ir, name, embed, offsets, mode, include_last_offset
        )
