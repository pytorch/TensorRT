from typing import Optional, Union

import tensorrt as trt
import torch
from tensorrt import ITensor as TRTTensor
from torch._subclasses.fake_tensor import unset_fake_temporarily
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
    Creates a sequence of values (arange) either dynamically or statically,
    then outputs a TensorRT tensor.

    If any of (start, end, step) is a TRT tensor, it sets up a dynamic arange
    using a Fill layer. Otherwise, the sequence is computed at build time and
    frozen into a TensorRT constant tensor.
    """
    # If any argument is a TRT tensor, use dynamic arange with a Fill layer
    if any(isinstance(x, TRTTensor) for x in (start, end, step)):
        value_dtype = _sequence_dtype(dtype, start, end, step)
        start_rank_0 = get_trt_tensor(
            ctx, start, name + "_start_rank_0", value_dtype, min_rank=0
        )
        # LINSPACE's start input requires rank 0; if the upstream ITensor came in
        # as rank-1 (e.g. a SymInt materialized by a sym_size op), reshape it.
        if len(start_rank_0.shape) > 0:
            squeeze_layer = ctx.net.add_shuffle(start_rank_0)
            squeeze_layer.reshape_dims = trt.Dims()
            set_layer_name(
                squeeze_layer, target, name + "_start_rank_0_squeeze", source_ir
            )
            start_rank_0 = squeeze_layer.get_output(0)

        start_rank_1 = get_trt_tensor(
            ctx, start, name + "_start_rank_1", value_dtype, min_rank=1
        )
        end = get_trt_tensor(ctx, end, name + "_end", value_dtype, min_rank=1)
        step = get_trt_tensor(ctx, step, name + "_step", value_dtype, min_rank=1)

        # The number of elements is ceil((end - start) / step), computed as
        # -floor((start - end) / step) so that the whole expression stays in the
        # operand dtype and remains a valid TRT shape tensor. Plain truncating
        # division would drop the last element whenever the span is not an exact
        # multiple of the step.
        length = impl.elementwise.sub(
            ctx, target, source_ir, name + "_sub", start_rank_1, end
        )
        length = impl.elementwise.floor_divide(
            ctx, target, source_ir, name + "_floor_div", length, step
        )
        length = impl.elementwise.mul(
            ctx, target, source_ir, name + "_negate", length, -1
        )
        length = cast_trt_tensor(
            ctx, length, trt.DataType.INT32, name + "_length_casted"
        )

        # Build a Fill layer in LINSPACE mode
        fill_layer = ctx.net.add_fill(
            length.shape, trt.FillOperation.LINSPACE, value_dtype
        )
        set_layer_name(fill_layer, target, name + "_arange_fill", source_ir)
        fill_layer.set_input(0, length)  # output length
        fill_layer.set_input(1, start_rank_0)  # start value
        fill_layer.set_input(2, step)  # step size

        return fill_layer.get_output(0)

    else:
        # All arguments are static, so evaluate the sequence eagerly and freeze it
        # into the engine as a constant. Letting torch pick the dtype preserves
        # PyTorch's promotion rules (float result if any argument is a float).
        with unset_fake_temporarily():
            values = torch.arange(start, end, step, dtype=dtype)
        if values.dtype == torch.int64:
            values = values.to(torch.int32)
        return get_trt_tensor(ctx, values, f"{name}_arange_const")
