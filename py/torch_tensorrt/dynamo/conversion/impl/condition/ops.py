from typing import Any, Optional, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    broadcastable,
    cast_trt_tensor,
    get_trt_tensor,
    prepend_ones,
    promote_trt_tensors_to_same_dtype,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl.elementwise import ne
from torch_tensorrt.dynamo.conversion.impl.shuffle import reshape as reshape_tensor


def where(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: Union[TRTTensor, np.ndarray, torch.Tensor],
    other: Union[TRTTensor, np.ndarray, torch.Tensor],
    condition: Union[TRTTensor, np.ndarray, torch.Tensor],
) -> TRTTensor:
    if not (broadcastable(input, other)):
        assert "The two torch tensors should be broadcastable"

    x_shape = list(input.shape)
    y_shape = list(other.shape)
    condition_shape = list(condition.shape)
    max_shape_len = max(len(x_shape), len(y_shape), len(condition_shape))

    if not isinstance(condition, TRTTensor):
        condition = get_trt_tensor(ctx, condition, f"{name}_condition")

    if condition.dtype != trt.bool:
        condition = cast_trt_tensor(ctx, condition, trt.float32, f"{name}_cast")
        condition = ne(ctx, target, source_ir, f"{name}_cond_zero", condition, 0)

    diff = max_shape_len - len(condition_shape)
    if diff > 0:
        condition = prepend_ones(ctx, condition, f"{name}_condition_broadcast", diff)

    if not isinstance(input, TRTTensor):
        input = get_trt_tensor(ctx, input, f"{name}_x")
    diff = max_shape_len - len(x_shape)
    if diff > 0:
        input = prepend_ones(ctx, input, f"{name}_input_broadcast", diff)

    if not isinstance(other, TRTTensor):
        other = get_trt_tensor(ctx, other, f"{name}_y")
    diff = max_shape_len - len(y_shape)
    if diff > 0:
        other = prepend_ones(ctx, other, f"{name}_other_broadcast", diff)

    # Ensure that input and other have the same TRT dtype
    input, other = promote_trt_tensors_to_same_dtype(ctx, input, other, name)

    return select(ctx, target, source_ir, name, input, other, condition)


def select(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    other: TRTTensor,
    condition: TRTTensor,
) -> TRTTensor:
    select_layer = ctx.net.add_select(condition, input, other)
    set_layer_name(select_layer, target, name + "_select", source_ir)
    return select_layer.get_output(0)


def _as_sequence(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def cond(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    pred: Union[TRTTensor, torch.Tensor, bool],
    true_fn: torch.fx.GraphModule,
    false_fn: torch.fx.GraphModule,
    operands: Any,
) -> Tuple[TRTTensor, ...]:
    """Convert torch.ops.higher_order.cond to a TensorRT IIfConditional.

    Both branches consume the same ``IIfConditionalInputLayer`` tensors. Layers
    created while converting ``true_fn`` / ``false_fn`` are associated with the
    corresponding branch by TensorRT via the path from those inputs to
    ``add_output``.
    """
    if not isinstance(true_fn, torch.fx.GraphModule) or not isinstance(
        false_fn, torch.fx.GraphModule
    ):
        raise RuntimeError(
            f"{name}: torch.cond branches must be GraphModules, got "
            f"{type(true_fn)} and {type(false_fn)}"
        )

    if not isinstance(pred, TRTTensor):
        pred = get_trt_tensor(ctx, pred, f"{name}_pred", dtype=torch.bool, min_rank=0)
    if pred.dtype != trt.bool:
        pred = cast_trt_tensor(
            ctx, pred, torch.bool, f"{name}_pred_bool", target, source_ir
        )
    # TensorRT requires a 0-D boolean predicate.
    if len(pred.shape) != 0:
        pred = reshape_tensor(ctx, target, source_ir, f"{name}_pred_scalar", pred, [])

    conditional = ctx.net.add_if_conditional()
    conditional.name = name
    conditional.set_condition(pred)

    wrapped_operands = []
    for i, operand in enumerate(_as_sequence(operands)):
        if not isinstance(operand, TRTTensor):
            operand = get_trt_tensor(ctx, operand, f"{name}_operand_{i}")
        wrapped_operands.append(conditional.add_input(operand).get_output(0))

    from torch_tensorrt.dynamo.conversion._SubgraphInterpreter import convert_subgraph

    true_outs = convert_subgraph(ctx, true_fn, wrapped_operands, f"{name}_true")
    false_outs = convert_subgraph(ctx, false_fn, wrapped_operands, f"{name}_false")
    if len(true_outs) != len(false_outs):
        raise RuntimeError(
            f"{name}: cond branches return different numbers of tensors "
            f"({len(true_outs)} vs {len(false_outs)})"
        )

    outputs: list[TRTTensor] = []
    for i, (t_out, f_out) in enumerate(zip(true_outs, false_outs)):
        layer = conditional.add_output(t_out, f_out)
        set_layer_name(layer, target, f"{name}_out_{i}", source_ir)
        outputs.append(layer.get_output(0))
    return tuple(outputs)
