# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import tensorrt as trt
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_trt_tensor,
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl.cat import (
    unify_and_concat_trt_tensors as unify_trt_shape_tensors,
)
from torch_tensorrt.dynamo.conversion.impl.shape import (
    get_shape_with_dynamic_shape,
)


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

    if scale_factor is not None:
        layer.scales = [1.0, 1.0] + list(scale_factor)
    else:
        shape = list(input.shape)[:2]
        if size is not None:
            shape += list(size)
        if has_dynamic_shape(shape):
            shape = get_shape_with_dynamic_shape(
                ctx, target, source_ir, name, shape, input
            )
            layer.set_input(1, shape)
        else:
            trt_shape = unify_trt_shape_tensors(
                ctx, target, name, shape, concat_axis=0, force_trt_output=False
            )
            if isinstance(trt_shape, list):
                layer.shape = trt_shape
            else:
                layer.set_input(1, trt_shape)

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


def _antialias_taps(
    input_size: int,
    output_size: int,
    align_corners: bool,
    scale: Optional[float],
    mode: Literal["bilinear", "bicubic"],
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Returns ATen's antialiased resize on one axis as (output_size, taps) arrays: the
    input index of each tap, its weight, and whether it is inside the output's window.
    """
    if align_corners:
        scale = (input_size - 1) / (output_size - 1) if output_size > 1 else 0.0
    elif scale is not None and scale > 0:
        scale = 1.0 / scale
    else:
        scale = input_size / output_size
    # A downscale stretches the filter over every input pixel it covers.
    stretch = max(scale, 1.0)
    support = (1.0 if mode == "bilinear" else 2.0) * stretch
    center = scale * (np.arange(output_size) + 0.5)
    start = np.maximum((center - support + 0.5).astype(np.int64), 0)
    count = np.minimum((center + support + 0.5).astype(np.int64), input_size) - start
    taps = np.arange(count.max())
    in_window = taps < count[:, None]
    indices = np.minimum(start[:, None] + taps, input_size - 1)
    x = np.abs(indices + 0.5 - center[:, None]) / stretch
    if mode == "bilinear":
        weights = np.maximum(1.0 - x, 0.0)
    else:
        a = -0.5
        weights = np.where(
            x < 1.0,
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0,
            np.where(x < 2.0, (((x - 5.0) * x + 8.0) * x - 4.0) * a, 0.0),
        )
    weights *= in_window
    return indices, weights / weights.sum(axis=1, keepdims=True), in_window


def upsample_antialias(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    size: Sequence[int],
    align_corners: bool,
    scales: Sequence[Optional[float]],
    mode: Literal["bilinear", "bicubic"],
) -> TRTTensor:
    # TensorRT's resize has no antialiasing. As in ATen, each output pixel is a weighted
    # sum of the input pixels in its window, along the width and then the height. The
    # sums run in float32 and are rounded to the input dtype once, at the end.
    output = cast_trt_tensor(
        ctx, input, trt.float32, f"{name}_float32", target, source_ir
    )
    for axis in (3, 2):
        indices, weights, in_window = _antialias_taps(
            input.shape[axis],
            size[axis - 2],
            align_corners,
            scales[axis - 2],
            mode,
        )
        # All taps in one gather and one reduce: per-tap gathers and adds get fused
        # across chained resizes, and the engine build can then take minutes.
        gather = ctx.net.add_gather(
            output,
            get_trt_tensor(ctx, indices.astype(np.int32), f"{name}_indices_{axis}"),
            axis,
        )
        set_layer_name(gather, target, f"{name}_gather_{axis}", source_ir)
        # The gather swaps the resized axis for the (output, taps) axes of the indices.
        shape = (1,) * axis + weights.shape + (1,) * (3 - axis)
        weighted = impl.elementwise.mul(
            ctx,
            target,
            source_ir,
            f"{name}_weighted_{axis}",
            gather.get_output(0),
            weights.reshape(shape).astype(np.float32),
        )
        if not in_window.all():
            # A tap past a window's end must add nothing, even for an infinite pixel.
            weighted = impl.condition.where(
                ctx,
                target,
                source_ir,
                f"{name}_in_window_{axis}",
                weighted,
                np.zeros((1,) * 5, dtype=np.float32),
                in_window.reshape(shape),
            )
        output = impl.reduce.sum(
            ctx, target, source_ir, f"{name}_sum_{axis}", weighted, axis + 1, False
        )
    return cast_trt_tensor(ctx, output, input.dtype, f"{name}_cast", target, source_ir)
