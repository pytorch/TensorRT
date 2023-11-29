import operator
import warnings
from typing import Any, Callable, Optional, Union

import numpy as np
import tensorrt as trt
import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_trt_tensor,
)
from torch_tensorrt.fx.converters.converter_utils import broadcast, set_layer_name
from torch_tensorrt.fx.types import TRTElementWiseOp, TRTTensor
from torch_tensorrt.fx.utils import Frameworks, unified_dtype_converter


def get_python_op_from_trt_elementwise_op(
    trt_op: TRTElementWiseOp,
) -> Callable[[Any, Any], Any]:
    if trt_op == trt.ElementWiseOperation.SUM:
        return operator.add
    elif trt_op == trt.ElementWiseOperation.PROD:
        return operator.mul
    elif trt_op == trt.ElementWiseOperation.MAX:
        return lambda a, b: max(a, b)
    elif trt_op == trt.ElementWiseOperation.MIN:
        return lambda a, b: min(a, b)
    elif trt_op == trt.ElementWiseOperation.SUB:
        return operator.sub
    elif trt_op == trt.ElementWiseOperation.DIV:
        return operator.truediv
    elif trt_op == trt.ElementWiseOperation.POW:
        return operator.pow
    elif trt_op == trt.ElementWiseOperation.FLOOR_DIV:
        return operator.floordiv
    elif trt_op == trt.ElementWiseOperation.AND:
        return lambda a, b: a and b
    elif trt_op == trt.ElementWiseOperation.OR:
        return lambda a, b: a or b
    elif trt_op == trt.ElementWiseOperation.XOR:
        return lambda a, b: (a or b) and not (a and b)
    elif trt_op == trt.ElementWiseOperation.EQUAL:
        return operator.eq
    elif trt_op == trt.ElementWiseOperation.GREATER:
        return operator.gt
    elif trt_op == trt.ElementWiseOperation.LESS:
        return operator.lt
    else:
        raise RuntimeError(f"{trt_op} is not supported yet!")


def convert_binary_elementwise(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    op_type: trt.ElementWiseOperation,
    lhs_val: Union[int, float, bool, TRTTensor, torch.Tensor],
    rhs_val: Union[int, float, bool, TRTTensor, torch.Tensor],
) -> TRTTensor:
    """
    This function adds a TensorRT elementwise layer. We allow both operands to be
    constant (not a trt tensor) because in implicit batch dimension mode, we could
    introduce constant via .size() op. Other scenario should be const folded first.
    If any operand is not a trt tensor, we make it a trt constant layer while preserve
    its dtype. Then we broadcast these two inputs to have the same number of dimensions.
    We also promote the types of the two tensors to avoid dtype errors in TRT.

    Limitation:
        If we are using implicit batch dim mode, the operand that is not a trt
    tensor are not allowed to have larger ranks than the trt tensor operand.

    Args:
        ctx (ConversionContext): TensorRT ConversionContext object.
        target (Target): Target of fx node.
        source_ir (SourceIR): The IR that is calling the function.
        name (str): The name we want to assign to the created TensorRT layer.
        lhs_val (TRTTensor): Left operand of the binary operation. Could
            be a TensorRT tensor, a PyTorch tensor or a simple value.
        rhs_val (TRTTensor): Right operand of the binary operation. Similar
            to lhs_val.
        op_type (trt.ElementWiseOperation): Type of the TensorRT elementwise binary operation.

    Returns:
        The output of TensorRT Elementwise layer.
    """
    lhs_dtype = None
    rhs_dtype = None
    is_lhs_trt_tensor = False
    is_rhs_trt_tensor = False

    if isinstance(lhs_val, TRTTensor):
        lhs_dtype = lhs_val.dtype
        is_lhs_trt_tensor = True
    if isinstance(rhs_val, TRTTensor):
        rhs_dtype = rhs_val.dtype
        is_rhs_trt_tensor = True

    if not is_lhs_trt_tensor and not is_rhs_trt_tensor:
        warnings.warn(
            f"Both operands of the binary elementwise op {name} "
            "are constant. In this case, please consider constant fold the model first."
        )
        return get_python_op_from_trt_elementwise_op(op_type)(lhs_val, rhs_val)

    # If the following conditions are true:
    #  1. the network has implicit batch dimension,
    #  2. one operand has shape [] (real shape is [batch_size]),
    #  3. another operand is a scalar,
    # then the result should also have shape [] (real shape is [batch_size]).
    #
    # In such case, we need to convert the scalar operand to tensor, because
    # this way the shape will become [1], and then will be properly squeezed
    # into [], meaning that the result will have shape [], which is what we
    # expect.
    #
    # Note that the dtype here is supposed to be the same as the scalar
    # dtype but we don't have a way to detect whether it makes sense for the
    # scalar to be float or half. Hence we go with the lhs dtype.
    if is_lhs_trt_tensor and isinstance(rhs_val, (float, int, bool)):
        rhs_val = np.array(
            [rhs_val], dtype=unified_dtype_converter(lhs_dtype, Frameworks.NUMPY)
        )
    if is_rhs_trt_tensor and isinstance(lhs_val, (float, int, bool)):
        lhs_val = np.array(
            [lhs_val], dtype=unified_dtype_converter(rhs_dtype, Frameworks.NUMPY)
        )

    lhs_val = get_trt_tensor(ctx, lhs_val, f"{name}_lhs", lhs_dtype)
    rhs_val = get_trt_tensor(ctx, rhs_val, f"{name}_rhs", rhs_dtype)

    promoted_type = torch.promote_types(
        unified_dtype_converter(lhs_val.dtype, Frameworks.TORCH),
        unified_dtype_converter(rhs_val.dtype, Frameworks.TORCH),
    )
    trt_promoted_type = unified_dtype_converter(promoted_type, Frameworks.TRT)

    if trt_promoted_type != lhs_val.dtype:
        lhs_val = cast_trt_tensor(
            ctx, lhs_val, trt_promoted_type, name, target, source_ir
        )
    if trt_promoted_type != rhs_val.dtype:
        rhs_val = cast_trt_tensor(
            ctx, rhs_val, trt_promoted_type, name, target, source_ir
        )

    # Check the limitation in the doc string.
    if ctx.net.has_implicit_batch_dimension:
        if is_lhs_trt_tensor and not is_rhs_trt_tensor:
            assert len(lhs_val.shape) >= len(
                rhs_val.shape
            ), f"{lhs_val.shape} >= {rhs_val.shape}"
        elif not is_lhs_trt_tensor and is_rhs_trt_tensor:
            assert len(rhs_val.shape) >= len(
                lhs_val.shape
            ), f"{rhs_val.shape} >= {lhs_val.shape}"

    lhs_val, rhs_val = broadcast(
        ctx.net, lhs_val, rhs_val, f"{name}_lhs", f"{name}_rhs"
    )
    layer = ctx.net.add_elementwise(lhs_val, rhs_val, op_type)
    set_layer_name(layer, target, name, source_ir)
    output = layer.get_output(0)
    kind: str = str(target.__name__) if callable(target) else target
    output.name = output.name + "_" + kind
    return output
