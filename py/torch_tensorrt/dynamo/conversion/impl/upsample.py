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
    out_shape: Optional[Sequence[int]],
    scale_factors: Optional[Sequence[float]],
    resize_mode: str,
    align_corners: bool,
) -> TRTTensor:
    resize_layer = ctx.net.add_resize(input)
    # output size calculation
    # Pytorch assumes that one of out_shape/scale_factor is None
    # Pytorch assumes that dimensions match for out_shape/scale factor
    if out_shape is not None:
        resize_layer.shape = list(input.shape)[:2] + list(out_shape)
    elif scale_factors is not None:
        resize_layer.scales = [1.0, 1.0] + list(scale_factors)
    else:
        raise RuntimeError(
            "At least one of out_shape and scale_factors should be specified."
        )

    # interpolate mode
    if resize_mode == "nearest" or None:
        resize_layer.resize_mode = trt.ResizeMode.NEAREST
    elif resize_mode == "bilinear":
        resize_layer.resize_mode = trt.ResizeMode.LINEAR
        if align_corners is None or not align_corners:
            raise RuntimeError(
                f"Interpolation works differently is align_corners is False for {resize_mode} mode in PyTorch and TensorRT."
            )
    else:
        raise RuntimeError(
            f"Interpolation mode is {resize_mode} which is not supported by TensorRT."
        )

    if resize_mode == "nearest":
        resize_layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.ASYMMETRIC
        )
    elif resize_mode == "bilinear":
        # align corners
        if align_corners is not None and align_corners:
            resize_layer.coordinate_transformation = (
                trt.ResizeCoordinateTransformation.ALIGN_CORNERS
            )
        else:
            resize_layer.coordinate_transformation = (
                trt.ResizeCoordinateTransformation.ASYMMETRIC
            )

    set_layer_name(resize_layer, target, name, source_ir)

    out = resize_layer.get_output(0)
    return out
