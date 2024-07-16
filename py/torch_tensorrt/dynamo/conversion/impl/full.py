from typing import List, Optional, Union

import numpy as np
import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    SourceIR,
    cast_trt_tensor,
    set_layer_name,
)
from torch_tensorrt.fx.types import TRTTensor


def full(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    shape: Union[List[int], TRTTensor],
    fill_value: Union[int, float, bool],
) -> TRTTensor:
    # in static shape scenario, shape is a list of int
    if isinstance(shape, List):
        return np.full(shape, fill_value)

    # in dynamic shape scenario, shape is a shape tensor
    # TODO: investigate how to use IFillLayer to directly fill a shape tensor
    # currently we use the IFillLayer to fill a 1 dimensional tensor and then reshape to the shape tensor
    end = impl.reduce.prod(
        ctx, target, source_ir, name + "_prod", shape, dim=None, keepdim=False
    )
    tensor = impl.arange.arange(ctx, target, source_ir, name + "_arange", 0, end, 1)
    tensor = impl.elementwise.mul(ctx, target, source_ir, name + "_mul", tensor, 0)
    if isinstance(fill_value, (int, float)):
        if isinstance(fill_value, float):
            tensor = cast_trt_tensor(
                ctx, tensor, trt.float32, name + "_casted", target, source_ir
            )
        tensor = impl.elementwise.add(
            ctx, target, source_ir, name + "_add", tensor, fill_value
        )

    if isinstance(fill_value, bool):
        tensor = cast_trt_tensor(
            ctx, tensor, trt.bool, name + "_casted", target, source_ir
        )
        tensor = impl.elementwise.logical_or(
            ctx, target, source_ir, name + "_add", tensor, fill_value
        )
    layer = ctx.net.add_shuffle(tensor)
    layer.set_input(1, shape)
    set_layer_name(layer, target, name + "_reshape", source_ir)
    return layer.get_output(0)
