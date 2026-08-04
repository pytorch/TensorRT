from typing import Optional, Union

import numpy as np
import tensorrt as trt
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    SourceIR,
    cast_trt_tensor,
    get_trt_tensor,
    set_layer_name,
)


def arange(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    start: Union[int, TRTTensor],
    end: Union[int, TRTTensor],
    step: Union[int, TRTTensor],
) -> TRTTensor:
    """
    Creates a sequence of values (arange) with a TensorRT Fill layer.

    If any of (start, end, step) is a TRT tensor, the Fill output length is
    computed dynamically. Otherwise, NumPy is used only to determine the static
    output length. Keeping static ranges as Fill layers preserves their sequence
    provenance for downstream TensorRT graph-pattern recognition.
    """
    # If any argument is a TRT tensor, use dynamic arange with a Fill layer
    if any(isinstance(x, TRTTensor) for x in (start, end, step)):
        # Convert start, end, step into TRT tensors with appropriate rank
        start_rank_0 = get_trt_tensor(ctx, start, name + "_start_rank_0", min_rank=0)
        # LINSPACE's start input requires rank 0; if the upstream ITensor came in
        # as rank-1 (e.g. a SymInt materialized by a sym_size op), reshape it.
        if isinstance(start_rank_0, TRTTensor) and len(start_rank_0.shape) > 0:
            squeeze_layer = ctx.net.add_shuffle(start_rank_0)
            squeeze_layer.reshape_dims = trt.Dims()
            set_layer_name(
                squeeze_layer, target, name + "_start_rank_0_squeeze", source_ir
            )
            start_rank_0 = squeeze_layer.get_output(0)
        start_rank_1 = get_trt_tensor(ctx, start, name + "_start_rank_1", min_rank=1)
        end = get_trt_tensor(ctx, end, name + "_end", min_rank=1)
        step = get_trt_tensor(ctx, step, name + "_step", min_rank=1)

        # Compute (end - start) / step to determine the output length
        shape = impl.elementwise.sub(
            ctx, target, source_ir, name + "_sub", end, start_rank_1
        )
        shape = impl.elementwise.trunc_div(
            ctx, target, source_ir, name + "_shape", shape, step
        )
        shape = cast_trt_tensor(ctx, shape, end.dtype, name + "_shape_casted")

        # Build a Fill layer in LINSPACE mode
        fill_layer = ctx.net.add_fill(
            shape.shape, trt.FillOperation.LINSPACE, shape.dtype
        )
        fill_layer.set_input(0, shape)  # output length
        fill_layer.set_input(1, start_rank_0)  # start value
        fill_layer.set_input(2, step)  # step size

        return fill_layer.get_output(0)

    else:
        # Keep a static arange as LINSPACE rather than materializing its values in
        # a Constant. TensorRT uses this producer provenance when recognizing
        # compact causal attention masks.
        output_shape = np.arange(start, end, step, dtype=np.int32).shape
        start_tensor = get_trt_tensor(
            ctx, start, name + "_start", dtype=trt.int32, min_rank=0
        )
        step_tensor = get_trt_tensor(
            ctx, step, name + "_step", dtype=trt.int32, min_rank=1
        )
        fill_layer = ctx.net.add_fill(
            output_shape, trt.FillOperation.LINSPACE, trt.int32
        )
        fill_layer.set_input(1, start_tensor)
        fill_layer.set_input(2, step_tensor)
        set_layer_name(fill_layer, target, f"{name}_arange_fill", source_ir)
        return fill_layer.get_output(0)
