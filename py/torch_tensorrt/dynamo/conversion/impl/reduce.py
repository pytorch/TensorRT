from typing import Optional, Sequence, Tuple, Union

import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_axes_for_reduce_op,
    get_positive_dim,
)
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTTensor


def amax(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    dim: Sequence[int] = [],
    keepdim: bool = False,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    if isinstance(dim, (tuple, list)) and len(dim) == 0:
        dim = tuple(range(len(input_val.shape)))

    layer = ctx.net.add_reduce(
        input_val,
        trt.ReduceOperation.MAX,
        axes=get_axes_for_reduce_op(get_positive_dim(dim, len(input_val.shape))),
        keep_dims=keepdim,
    )
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def amin(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    dim: Sequence[int] = [],
    keepdim: bool = False,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    if isinstance(dim, (tuple, list)) and len(dim) == 0:
        dim = tuple(range(len(input_val.shape)))

    layer = ctx.net.add_reduce(
        input_val,
        trt.ReduceOperation.MIN,
        axes=get_axes_for_reduce_op(get_positive_dim(dim, len(input_val.shape))),
        keep_dims=keepdim,
    )
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def sum(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    dim: Optional[Union[int, Sequence[int]]],
    keepdim: bool,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (input_val.dtype == trt.bool):
        input_val = cast_trt_tensor(ctx, input_val, trt.int32, name)

    if dim is None or (isinstance(dim, (tuple, list)) and len(dim) == 0):
        dim = tuple(range(len(input_val.shape)))

    layer = ctx.net.add_reduce(
        input_val,
        trt.ReduceOperation.SUM,
        axes=get_axes_for_reduce_op(get_positive_dim(dim, len(input_val.shape))),
        keep_dims=keepdim,
    )
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def prod(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    dim: Optional[Union[int, Sequence[int]]],
    keepdim: bool,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    if dim is None:
        dim = tuple(range(len(input_val.shape)))

    layer = ctx.net.add_reduce(
        input_val,
        trt.ReduceOperation.PROD,
        axes=get_axes_for_reduce_op(get_positive_dim(dim, len(input_val.shape))),
        keep_dims=keepdim,
    )
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def max(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    dim: Optional[Union[int, Sequence[int]]],
    keepdim: bool,
    return_indices: bool,
) -> Union[TRTTensor, Tuple[TRTTensor, TRTTensor]]:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    if dim is None:
        dim = tuple(range(len(input_val.shape)))

    layer = ctx.net.add_reduce(
        input_val,
        trt.ReduceOperation.MAX,
        axes=get_axes_for_reduce_op(get_positive_dim(dim, len(input_val.shape))),
        keep_dims=keepdim,
    )
    set_layer_name(layer, target, name, source_ir)

    if return_indices:
        return layer.get_output(0), None

    return layer.get_output(0)


def min(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    dim: Optional[Union[int, Sequence[int]]],
    keepdim: bool,
    return_indices: bool,
) -> Union[TRTTensor, Tuple[TRTTensor, TRTTensor]]:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    if dim is None:
        dim = tuple(range(len(input_val.shape)))

    layer = ctx.net.add_reduce(
        input_val,
        trt.ReduceOperation.MIN,
        axes=get_axes_for_reduce_op(get_positive_dim(dim, len(input_val.shape))),
        keep_dims=keepdim,
    )
    set_layer_name(layer, target, name, source_ir)

    if return_indices:
        return layer.get_output(0), None

    return layer.get_output(0)


def mean(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    dim: Optional[Union[int, Sequence[int]]],
    keepdim: bool,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(ctx, input_val, trt.float32, name)

    if dim is None or (isinstance(dim, (tuple, list)) and len(dim) == 0):
        dim = tuple(range(len(input_val.shape)))

    layer = ctx.net.add_reduce(
        input_val,
        trt.ReduceOperation.AVG,
        axes=get_axes_for_reduce_op(get_positive_dim(dim, len(input_val.shape))),
        keep_dims=keepdim,
    )
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)
