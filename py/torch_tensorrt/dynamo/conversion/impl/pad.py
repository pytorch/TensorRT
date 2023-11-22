import copy
from typing import Optional, Sequence, Union

import torch
import torch_tensorrt.dynamo.conversion.impl as impl
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.fx.converters.converter_utils import (
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.fx.types import TRTTensor


def constant_padNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    pad: Sequence[int],
    value: int = 0,
) -> TRTTensor:
    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for padding."

    pad_len = len(pad)

    if pad_len == 4 and value == 0:
        pre_padding = (pad[2], pad[0])
        post_padding = (pad[3], pad[1])

        # add padding layer
        pad_layer = ctx.net.add_padding_nd(
            input=input,
            pre_padding=pre_padding,
            post_padding=post_padding,
        )

        pad_layer.pre_padding_nd = pre_padding
        pad_layer.post_padding_nd = post_padding

        set_layer_name(pad_layer, target, name, source_ir)
        return pad_layer.get_output(0)

    else:
        # Implement constant padding via concat
        curr_dim = len(input.shape) - 1

        for i in range(0, pad_len, 2):
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
