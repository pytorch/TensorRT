# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Union

import numpy as np
import tensorrt as trt
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt import _enums
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    SourceIR,
    cast_trt_tensor,
    get_trt_tensor,
    set_layer_name,
)


def _sequence_dtype(
    dtype: Optional[torch.dtype],
    *operands: Union[int, float, torch.Tensor, TRTTensor],
) -> trt.DataType:
    """
    Resolve the dtype of the generated sequence.
    """
    if dtype is not None:
        return _enums.dtype._from(dtype).to(trt.DataType)

    for x in operands:
        if isinstance(x, TRTTensor):
            if _enums.dtype._from(x.dtype).to(torch.dtype).is_floating_point:
                return trt.DataType.FLOAT

        if isinstance(x, torch.Tensor):
            if x.dtype.is_floating_point:
                return trt.DataType.FLOAT

        if isinstance(x, float):
            return trt.DataType.FLOAT

    return trt.DataType.INT64


def arange(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    start: Union[int, float, TRTTensor],
    end: Union[int, float, TRTTensor],
    step: Union[int, float, TRTTensor],
    dtype: Optional[torch.dtype] = None,
) -> TRTTensor:
    """
    Create a sequence with a TensorRT Fill layer or a constant.

    If any of (start, end, step) is a TRT tensor, the Fill output length is
    computed dynamically. Static integer ranges use Fill to preserve their
    sequence provenance for attention-pattern recognition, without allocating
    the sequence on the host. Other static ranges retain the NumPy constant
    path for dtype, rounding, and truncate_double compatibility.
    """
    is_dynamic = any(isinstance(x, TRTTensor) for x in (start, end, step))
    if is_dynamic:
        value_dtype = _sequence_dtype(dtype, start, end, step)
    elif (
        dtype in (None, torch.int32, torch.int64)
        and isinstance(start, int)
        and isinstance(end, int)
        and isinstance(step, int)
    ):
        fill_shape = (len(range(start, end, step)),)
        # Match the existing static converter's INT64-to-INT32 normalization.
        value_dtype = trt.DataType.INT32
    else:
        # LINSPACE cannot output FP16/BF16. Keep the established constant path
        # for floating-point ranges, including their rounding semantics and
        # get_trt_tensor's optional FP64-to-FP32 truncation.
        resolved_dtype = dtype
        if resolved_dtype is None and any(
            isinstance(value, float) for value in (start, end, step)
        ):
            resolved_dtype = torch.get_default_dtype()
        fallback_dtype = None
        if resolved_dtype is not None:
            try:
                np_dtype = _enums.dtype._from(resolved_dtype).to(np.dtype)
            except TypeError:
                # BF16 has no NumPy representation; cast when creating the
                # constant instead of asking LINSPACE for an unsupported dtype.
                np_dtype = None
                fallback_dtype = resolved_dtype
        else:
            np_dtype = None
        values = np.arange(start, end, step, dtype=np_dtype)
        if values.dtype == np.int64:
            values = values.astype(np.int32)
        return get_trt_tensor(ctx, values, f"{name}_arange_const", dtype=fallback_dtype)

    start_name = name + ("_start_rank_0" if is_dynamic else "_start")
    start_tensor = get_trt_tensor(ctx, start, start_name, value_dtype, min_rank=0)
    if is_dynamic:
        start_tensor = cast_trt_tensor(
            ctx, start_tensor, value_dtype, name + "_start_rank_0_casted"
        )
    # LINSPACE's start input requires rank 0; if the upstream ITensor came in
    # as rank-1 (e.g. a SymInt materialized by a sym_size op), reshape it.
    if len(start_tensor.shape) > 0:
        squeeze_layer = ctx.net.add_shuffle(start_tensor)
        squeeze_layer.reshape_dims = trt.Dims()
        set_layer_name(squeeze_layer, target, name + "_start_rank_0_squeeze", source_ir)
        start_tensor = squeeze_layer.get_output(0)

    step_tensor = get_trt_tensor(
        ctx, step, name + "_step", dtype=value_dtype, min_rank=1
    )

    if is_dynamic:
        start_rank_1 = get_trt_tensor(
            ctx, start, name + "_start_rank_1", value_dtype, min_rank=1
        )
        end_tensor = get_trt_tensor(ctx, end, name + "_end", value_dtype, min_rank=1)
        start_rank_1 = cast_trt_tensor(
            ctx, start_rank_1, value_dtype, name + "_start_rank_1_casted"
        )
        end_tensor = cast_trt_tensor(ctx, end_tensor, value_dtype, name + "_end_casted")
        step_tensor = cast_trt_tensor(
            ctx, step_tensor, value_dtype, name + "_step_casted"
        )

        # The number of elements is ceil((end - start) / step), computed as
        # -floor((start - end) / step) so that the whole expression stays in the
        # operand dtype and remains a valid TRT shape tensor. Plain truncating
        # division would drop the last element whenever the span is not an exact
        # multiple of the step.
        length = impl.elementwise.sub(
            ctx, target, source_ir, name + "_sub", start_rank_1, end_tensor
        )
        length = impl.elementwise.floor_divide(
            ctx, target, source_ir, name + "_floor_div", length, step_tensor
        )
        length = impl.elementwise.mul(
            ctx, target, source_ir, name + "_negate", length, -1
        )
        length = cast_trt_tensor(
            ctx, length, trt.DataType.INT32, name + "_length_casted"
        )
        fill_shape = length.shape

    fill_layer = ctx.net.add_fill(fill_shape, trt.FillOperation.LINSPACE, value_dtype)
    set_layer_name(fill_layer, target, name + "_arange_fill", source_ir)
    if is_dynamic:
        fill_layer.set_input(0, length)  # output length
    fill_layer.set_input(1, start_tensor)  # start value
    fill_layer.set_input(2, step_tensor)  # step size

    return fill_layer.get_output(0)
