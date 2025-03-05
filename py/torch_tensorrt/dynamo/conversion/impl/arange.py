from typing import Optional, Union

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


def arange(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    start: Union[int, TRTTensor],
    end: Union[int, TRTTensor],
    step: Union[int, TRTTensor],
) -> TRTTensor:
    if any(isinstance(tensor, TRTTensor) for tensor in (start, end, step)):
        start_rank_0 = get_trt_tensor(ctx, start, name + "_start_rank_0", min_rank=0)
        start_rank_1 = get_trt_tensor(ctx, start, name + "_start_rank_1", min_rank=1)
        end = get_trt_tensor(ctx, end, name + "_end", min_rank=1)
        step = get_trt_tensor(ctx, step, name + "_step", min_rank=1)
        # Calculate shape = (end-start) / step
        shape = impl.elementwise.sub(
            ctx,
            target,
            source_ir,
            name + "_sub",
            end,
            start_rank_1,
        )
        shape = impl.elementwise.trunc_div(
            ctx,
            target,
            source_ir,
            name + "_shape",
            shape,
            step,
        )
        shape = cast_trt_tensor(ctx, shape, end.dtype, name + "_shape_casted")
        fill_layer = ctx.net.add_fill(
            shape.shape, trt.FillOperation.LINSPACE, shape.dtype
        )
        fill_layer.set_input(0, shape)
        # Set start index
        fill_layer.set_input(1, start_rank_0)
        # Set delta/step
        fill_layer.set_input(2, step)
        return fill_layer.get_output(0)
    return np.arange(start, end, step)
