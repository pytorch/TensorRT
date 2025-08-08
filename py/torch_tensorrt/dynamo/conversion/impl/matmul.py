from typing import Optional

import tensorrt as trt
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt import _enums
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    broadcast,
    cast_trt_tensor,
    get_trt_tensor,
    set_layer_name,
)


def matrix_multiply(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    other: TRTTensor,
    input_matrix_op: trt.MatrixOperation = trt.MatrixOperation.NONE,
    other_matrix_op: trt.MatrixOperation = trt.MatrixOperation.NONE,
) -> TRTTensor:
    if not isinstance(input, trt.ITensor):
        input = get_trt_tensor(ctx, input, f"{name}_input")
    if not isinstance(other, trt.ITensor):
        other = get_trt_tensor(
            ctx,
            other,
            f"{name}_other",
            dtype=_enums.dtype._from(input.dtype).to(torch.dtype),
        )

    preset_diff = 0

    if len(input.shape) == 1:
        preset_diff -= 1
        input_matrix_op = trt.MatrixOperation.VECTOR

    if len(other.shape) == 1:
        preset_diff += 1
        other_matrix_op = trt.MatrixOperation.VECTOR

    input, other = broadcast(
        ctx, input, other, f"{name}_input", f"{name}_other", preset_diff
    )
    if ctx.net.get_flag(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED):
        promoted_type = _enums.dtype._from(
            torch.promote_types(
                _enums.dtype._from(input.dtype).to(torch.dtype),
                _enums.dtype._from(other.dtype).to(torch.dtype),
            )
        )
        trt_promoted_type = promoted_type.to(trt.DataType)
        input = cast_trt_tensor(ctx, input, trt_promoted_type, f"{name}_input_casted")
        other = cast_trt_tensor(ctx, other, trt_promoted_type, f"{name}_other_casted")

    layer = ctx.net.add_matrix_multiply(input, input_matrix_op, other, other_matrix_op)
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)
