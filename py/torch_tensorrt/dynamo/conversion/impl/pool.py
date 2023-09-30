from typing import Optional, Sequence, Union

import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import extend_attr_to_tuple
from torch_tensorrt.fx.converters.converter_utils import (
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.fx.types import TRTTensor


def avg_poolNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    kernel_size: Sequence[int],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> TRTTensor:
    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for pooling."

    if ceil_mode is not False:
        raise RuntimeError("ceil_mode is not yet supported!")

    if divisor_override is not None:
        raise RuntimeError("divisor_override is not yet supported!")

    dim = len(kernel_size)

    kernel_size = extend_attr_to_tuple(kernel_size, dim)

    if stride == []:
        stride = kernel_size
    else:
        stride = extend_attr_to_tuple(stride, dim)

    padding = extend_attr_to_tuple(padding, dim)

    # add average pooling layer
    pool_layer = ctx.net.add_pooling_nd(
        input=input,
        type=trt.PoolingType.AVERAGE,
        window_size=kernel_size,
    )

    pool_layer.stride_nd = stride
    pool_layer.padding_nd = padding
    pool_layer.average_count_excludes_padding = not count_include_pad

    set_layer_name(pool_layer, target, name, source_ir)
    return pool_layer.get_output(0)


def max_poolNd(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    kernel_size: Sequence[int],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    ceil_mode: bool = False,
) -> TRTTensor:
    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for pooling."

    if dilation != 1:
        raise RuntimeError("dilation is not yet supported!")

    if ceil_mode is not False:
        raise RuntimeError("ceil_mode is not yet supported!")

    dim = len(kernel_size)

    kernel_size = extend_attr_to_tuple(kernel_size, dim)

    if stride == []:
        stride = kernel_size
    else:
        stride = extend_attr_to_tuple(stride, dim)

    padding = extend_attr_to_tuple(padding, dim)

    # add max pooling layer
    pool_layer = ctx.net.add_pooling_nd(
        input=input,
        type=trt.PoolingType.MAX,
        window_size=kernel_size,
    )

    pool_layer.stride_nd = stride
    pool_layer.padding_nd = padding

    set_layer_name(pool_layer, target, name, source_ir)
    return pool_layer.get_output(0)
