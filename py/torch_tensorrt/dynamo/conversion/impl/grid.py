from typing import Optional, Sequence

import tensorrt as trt
import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    GridSamplerInterpolation,
    GridSamplerSampling,
    cast_trt_tensor,
)
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor


def grid(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    grid: TRTTensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    output_mask: Optional[Sequence[bool]] = None,
) -> TRTTensor:
    grid_layer = ctx.net.add_grid_sample(input, grid)
    interpolation_mode_trt = GridSamplerInterpolation()
    grid_layer.interpolation_mode = interpolation_mode_trt(interpolation_mode)
    sample_mode_trt = GridSamplerSampling()
    grid_layer.sample_mode = sample_mode_trt(padding_mode)
    grid_layer.align_corners = align_corners
    set_layer_name(grid_layer, target, name + "_grid_layer", source_ir)
    if output_mask is None:
        return grid_layer.get_output(0)
    else:
        if output_mask[0] and output_mask[1]:
            return (grid_layer.get_output(0), None)
        elif output_mask[0]:
            return grid_layer.get_output(0)
        else:
            return None
