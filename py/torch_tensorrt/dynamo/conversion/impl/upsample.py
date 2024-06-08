from typing import Optional, Sequence

import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTTensor


def upsample(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    size: Optional[Sequence[int]],
    scale_factor: Optional[Sequence[float]],
    mode: str,
    align_corners: bool,
) -> TRTTensor:
    layer = ctx.net.add_resize(input)

    if size is not None:
        layer.shape = list(input.shape)[:2] + list(size)
    else:
        layer.scales = [1.0, 1.0] + list(scale_factor)

    if mode == "nearest":
        layer.resize_mode = trt.InterpolationMode.NEAREST
        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC
    elif mode in ("linear", "bilinear", "trilinear"):
        layer.resize_mode = trt.InterpolationMode.LINEAR
        layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.ALIGN_CORNERS
            if align_corners
            else trt.ResizeCoordinateTransformation.HALF_PIXEL
        )
    elif mode == "bicubic":
        layer.resize_mode = trt.InterpolationMode.CUBIC
        layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.ALIGN_CORNERS
            if align_corners
            else trt.ResizeCoordinateTransformation.HALF_PIXEL
        )

    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)
