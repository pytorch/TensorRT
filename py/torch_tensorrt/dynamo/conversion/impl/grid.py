from typing import Optional

import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion.converter_utils import GridSamplerInterpolation, GridSamplerPadding
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor

def grid(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    grid: TRTTensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> TRTTensor:
    grid_layer = network.add_grid_sample(input, grid)
    grid_layer.interpolation_mode = GridSamplerInterpolation(interpolation_mode)
    grid_layer.padding_mode = GridSamplerPadding(padding_mode)
    grid_layer.align_corners = align_corners
    set_layer_name(grid_layer, target, name + "_grid_layer", source_ir)
    return grid_layer.get_output(0)