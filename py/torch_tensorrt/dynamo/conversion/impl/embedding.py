import functools
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch_tensorrt.dynamo.conversion.impl as impl
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor, to_numpy
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTTensor


def embedding(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: TRTTensor,
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
    # ignore padding_idx since it is meaningful for training only

    if scale_grad_by_freq:
        raise RuntimeError(
            "Currently we don't support scale gradient by word frequency."
        )

    if sparse:
        raise RuntimeError("Currently we don't support sparse gradient.")

    # Implement embedding lookup with gather layer
    gather_layer = ctx.net.add_gather(embedding_tensor, indices_tensor, axis=0)
    set_layer_name(gather_layer, target, f"{name}_gather", source_ir)
    return gather_layer.get_output(0)


def embedding_bag(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    weight: TRTTensor,
    indices: TRTTensor,
    offsets: Union[torch.Tensor, np.ndarray, Sequence[int]],
    scale_grad_by_freq: bool,
    mode: int,
    sparse: bool,
    per_sample_weights: Optional[TRTTensor],
    include_last_offset: bool,
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

    # calculate embedding
    embed = embedding(
        ctx,
        target,
        source_ir,
        f"{name}_embedding",
        indices,
        weight,
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

    offsets = to_numpy(offsets)

    if include_last_offset is False:
        # add the end index to offsets
        offsets = np.append(offsets, indices.shape[0])
    else:
        # modify the last index of offsets to the end index
        # however, pytorch doc says if `include_last_offset` is True, the size of offsets
        # is equal to the number of bags + 1. The last element is the size of the input,
        # or the ending index position of the last bag (sequence).
        offsets[-1] = indices.shape[0]

    # separately reduce embeddings for different bags
    reduced_embed = []
    len_offsets = len(offsets)
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
            reduced_sliced_embed = reduce_op(
                name=f"{name}_{reduce_name}_{i}",
                input_val=sliced_embed,
                dim=0,
                keepdim=True,
            )
            reduced_embed.append(reduced_sliced_embed)

    out = impl.cat.cat(ctx, target, source_ir, f"{name}_cat", reduced_embed, 0)
    # out = reduce_op(input_val=embed, dim=1, keepdim=False)  # Note: This implementation doesn't work for N-dim

    return out, None, None, None
