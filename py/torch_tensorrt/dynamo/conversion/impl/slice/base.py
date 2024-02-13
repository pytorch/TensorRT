from typing import Optional

from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.impl.shape import get_shape_with_dynamic_shape
from torch_tensorrt.fx.converters.converter_utils import (
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor
from torch_tensorrt.fx.types import Shape, TRTTensor

def get_dynamic_shape(ctx, target, source_ir, name, shape, input):
    trt_shape = []
    shape = input.shape
    for i, s in enumerate(shape):
        if isinstance(s, TRTTensor):
            trt_shape.append(s)
        else:
            a = get_trt_tensor(ctx, s, f"{name}_{i}")
            trt_shape.append(a)
    shape_layer = ctx.net.add_concatenation(inputs=trt_shape)
    shape_layer.axis = 0
    shape_layer.name = f"{name}_output_shape"
    
    return shape_layer.get_output(0)

def slice(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    start: Shape,
    shape: Shape,
    stride: Shape,
) -> TRTTensor:
    dynamic_shape = has_dynamic_shape(input.shape)
    if dynamic_shape:
        shape = get_dynamic_shape(ctx, target, source_ir, name, shape, input)
    layer = ctx.net.add_slice(
        input,
        start=start,
        shape=[] if dynamic_shape else shape,
        stride=stride,
    )
    if dynamic_shape:
        layer.set_input(2, shape)
    set_layer_name(layer, target, name)
    return layer.get_output(0)
