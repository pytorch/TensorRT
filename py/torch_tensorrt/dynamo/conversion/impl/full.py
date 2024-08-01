from typing import List, Optional, Union

import numpy as np
import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    SourceIR,
    cast_trt_tensor,
    get_trt_tensor,
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
        # in static shape scenario, shape is a list of int
        if all(isinstance(dim, int) for dim in shape):
            return np.full(shape, fill_value)
        else:
            shape = impl.cat.cat(
                ctx, target, source_ir, name + "_concat_shape", shape, 0
            )

    # in dynamic shape scenario, shape is a shap tensor
    # use IFillLayer to fill the shape tensor with LINSPACE value
    layer = ctx.net.add_fill(shape.shape, trt.FillOperation.LINSPACE, shape.dtype)
    layer.set_input(0, shape)
    layer.set_input(1, get_trt_tensor(ctx, 0, name + "_start", min_rank=0))
    delta = get_trt_tensor(ctx, 1, name + "_delta")
    input = []
    for _ in range(shape.shape[0]):
        input.append(delta)
    delta = impl.cat.cat(ctx, target, source_ir, name + "_cat", input, dim=0)
    layer.set_input(2, delta)
    output = layer.get_output(0)

    # fill the output tensor with the actual fill_value
    output = impl.elementwise.mul(ctx, target, source_ir, name + "_mul", output, 0)
    if isinstance(fill_value, (int, float)):
        if isinstance(fill_value, float):
            output = cast_trt_tensor(
                ctx, output, trt.float32, name + "_casted", target, source_ir
            )
        output = impl.elementwise.add(
            ctx, target, source_ir, name + "_add", output, fill_value
        )

    if isinstance(fill_value, bool):
        output = cast_trt_tensor(
            ctx, output, trt.bool, name + "_casted", target, source_ir
        )
        output = impl.elementwise.logical_or(
            ctx, target, source_ir, name + "_add", output, fill_value
        )

    return output
