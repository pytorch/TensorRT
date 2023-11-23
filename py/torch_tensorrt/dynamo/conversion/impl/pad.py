import copy
from typing import Optional, Sequence, Union

import torch
import torch_tensorrt.dynamo.conversion.impl as impl
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.fx.converters.converter_utils import has_dynamic_shape
from torch_tensorrt.fx.types import TRTTensor


def constant_padNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    pad: Sequence[int],
    value: Union[int, float] = 0,
) -> TRTTensor:
    """
    Note: IPaddingLayer is deprecated in TensorRT 8.2 and will be removed in TensorRT 10.0.
    Use ISliceLayer to pad the tensor, which supports new non-constant, reflects padding
    mode and clamp, and supports padding output with dynamic shape.
    """
    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for padding."

    # Implement constant padding via concat
    curr_dim = len(input.shape) - 1

    for i in range(0, len(pad), 2):
        input_shape = list(input.shape)

        pre_pad = pad[i]
        post_pad = pad[i + 1]
        pre_pad_shape = copy.deepcopy(input_shape)
        pre_pad_shape[curr_dim] = pre_pad
        pre_pad_tensor = torch.full(pre_pad_shape, float(value))
        if pre_pad == post_pad:
            post_pad_tensor = pre_pad_tensor
        else:
            post_pad_shape = copy.deepcopy(input_shape)
            post_pad_shape[curr_dim] = post_pad
            post_pad_tensor = torch.full(post_pad_shape, float(value))
        output = impl.cat.cat(
            ctx,
            target,
            source_ir,
            f"{name}_concat{curr_dim}",
            input=(pre_pad_tensor, input, post_pad_tensor),
            dim=curr_dim,
        )
        curr_dim -= 1
        input = output

    return output


def reflection_padNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    padding: Sequence[int],
) -> TRTTensor:
    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for padding."

    padding_dims = len(padding) // 2

    if padding_dims == 1 or padding_dims == 2 or padding_dims == 3:
        for i in range(padding_dims):
            dim = -1 - i
            pre_pad, post_pad = padding[2 * i], padding[2 * i + 1]
            pre_pad_tensor = impl.slice.slice_op(
                ctx,
                target,
                source_ir,
                f"{name}_slice_pre{i}",
                input,
                dim=dim,
                start=pre_pad,
                stop=0,
                step=-1,
            )

            post_pad_tensor = impl.slice.slice_op(
                ctx,
                target,
                source_ir,
                f"{name}_slice_post{i}",
                input,
                dim=dim,
                start=input.shape[dim] - 2,
                stop=input.shape[dim] - post_pad - 2,
                step=-1,
            )

            output = impl.cat.cat(
                ctx,
                target,
                source_ir,
                f"{name}_concat_dim{dim}",
                input=(pre_pad_tensor, input, post_pad_tensor),
                dim=dim,
            )
            input = output

        return output

    else:
        raise RuntimeError(
            f"We currently only support for padding 1D, 2D, and 3D, but got {padding_dims}D"
        )


def replication_padNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    padding: Sequence[int],
) -> TRTTensor:
    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for padding."

    padding_dims = len(padding) // 2

    if padding_dims == 1 or padding_dims == 2 or padding_dims == 3:
        for i in range(padding_dims):
            dim = -1 - i
            pre_pad, post_pad = padding[2 * i], padding[2 * i + 1]
            pre_pad_tensor = impl.slice.slice_op(
                ctx,
                target,
                source_ir,
                f"{name}_slice_pre{i}",
                input,
                dim=dim,
                start=0,
                stop=1,
                step=1,
            )
            new_shape = input.shape
            new_shape[dim] = pre_pad
            pre_pad_tensor = impl.slice.expand(
                ctx,
                target,
                source_ir,
                f"{name}_expand_pre{i}",
                pre_pad_tensor,
                new_shape,
            )

            post_pad_tensor = impl.slice.slice_op(
                ctx,
                target,
                source_ir,
                f"{name}_slice_post{i}",
                input,
                dim=dim,
                start=input.shape[dim] - 1,
                stop=input.shape[dim],
                step=1,
            )
            new_shape[dim] = post_pad
            post_pad_tensor = impl.slice.expand(
                ctx,
                target,
                source_ir,
                f"{name}_expand_post{i}",
                post_pad_tensor,
                new_shape,
            )
            output = impl.cat.cat(
                ctx,
                target,
                source_ir,
                f"{name}_concat_dim{dim}",
                input=(pre_pad_tensor, input, post_pad_tensor),
                dim=dim,
            )
            input = output

        return output

    else:
        raise RuntimeError(
            f"We currently only support for padding 1D, 2D, and 3D, but got {padding_dims}D"
        )


def circular_padNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    pad: Sequence[int],
) -> TRTTensor:
    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for padding."

    padding_dims = len(pad) // 2

    if padding_dims == 1 or padding_dims == 2 or padding_dims == 3:
        for i in range(padding_dims):
            dim = -1 - i
            pre_pad, post_pad = pad[2 * i], pad[2 * i + 1]
            pre_pad_tensor = impl.slice.slice_op(
                ctx,
                target,
                source_ir,
                f"{name}_slice_pre{i}",
                input,
                dim=dim,
                start=input.shape[dim] - pre_pad,
                stop=input.shape[dim],
                step=1,
            )

            post_pad_tensor = impl.slice.slice_op(
                ctx,
                target,
                source_ir,
                f"{name}_slice_post{i}",
                input,
                dim=dim,
                start=0,
                stop=post_pad,
                step=1,
            )

            output = impl.cat.cat(
                ctx,
                target,
                source_ir,
                f"{name}_concat_dim{dim}",
                input=(pre_pad_tensor, input, post_pad_tensor),
                dim=dim,
            )
            input = output

        return output

    else:
        raise RuntimeError(
            f"We currently only support for padding 1D, 2D, and 3D, but got {padding_dims}D"
        )


def pad(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    pad: Sequence[int],
    mode: str = "constant",
    value: Optional[float] = None,
) -> TRTTensor:
    if mode == "constant":
        return constant_padNd(
            ctx,
            target,
            source_ir,
            f"{name}_{mode}",
            input,
            pad,
            value if value is not None else 0,
        )
    elif mode == "reflect":
        return reflection_padNd(ctx, target, source_ir, f"{name}_{mode}", input, pad)
    elif mode == "replicate":
        return replication_padNd(ctx, target, source_ir, f"{name}_{mode}", input, pad)
    elif mode == "circular":
        return circular_padNd(ctx, target, source_ir, f"{name}_{mode}", input, pad)
    else:
        raise RuntimeError(
            f'We currently only support for `mode` in ["constant", "reflect", "replicate", "circular"], but got {mode}'
        )
