import operator
import warnings
from typing import Union, Callable, Any, Optional

import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
from torch.fx.node import Target

from torch_tensorrt.fx.types import TRTNetwork, TRTTensor, TRTElementWiseOp
from torch_tensorrt.fx.utils import torch_dtype_from_trt
from torch_tensorrt.fx.converters.converter_utils import (
    SourceIR,
    set_layer_name,
    broadcast,
    squeeze_left,
    get_trt_tensor,
)


def get_python_op_from_trt_elementwise_op(
    trt_op: TRTElementWiseOp,
) -> Callable[[Any, Any], Any]:
    if trt_op == trt.ElementWiseOperation.SUM:
        return operator.add
    elif trt_op == trt.ElementWiseOperation.PROD:
        return operator.mul
    elif trt_op == trt.ElementWiseOperation.SUB:
        return operator.sub
    elif trt_op == trt.ElementWiseOperation.DIV:
        return operator.truediv
    elif trt_op == trt.ElementWiseOperation.FLOOR_DIV:
        return operator.floordiv
    else:
        raise RuntimeError(f"{trt_op} is not supported yet!")


def convert_binary_elementwise(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    op_type: trt.ElementWiseOperation,
    lhs_val: Union[int, float, TRTTensor, torch.Tensor],
    rhs_val: Union[int, float, TRTTensor, torch.Tensor],
) -> TRTTensor:
    """
    This function adds a TensorRT elementwise layer. We allow both operands to be
    constant (not a trt tensor) because in implicit batch dimension mode, we could
    introduce constant via .size() op. Other scenario should be const folded first.
    If any operand is not a trt tensor, we make it a trt constant layer while preserve
    its dtype. Then we broadcast these two inputs to have the same number of dimensions.

    Limitation:
        If we are using implicit batch dim mode, the operand that is not a trt
    tensor are not allowed to have larger ranks than the trt tensor operand.

    Args:
        network (TRTNetwork): TensorRT network object.
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
        lhs_dtype = torch_dtype_from_trt(lhs_val.dtype)
        is_lhs_trt_tensor = True
    if isinstance(rhs_val, TRTTensor):
        rhs_dtype = torch_dtype_from_trt(rhs_val.dtype)
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
    if is_lhs_trt_tensor and isinstance(rhs_val, (float, int)):
        rhs_val = torch.tensor([rhs_val], dtype=lhs_dtype)
    if is_rhs_trt_tensor and isinstance(lhs_val, (float, int)):
        lhs_val = torch.tensor([lhs_val], dtype=rhs_dtype)

    # When lhs is scalar, and rhs has shape [1,], then currently the assert
    # will fail because lhs shape has fewer dimensions than rhs shape.  This
    # happens when using implicit batch dimension, when we removed the 1st
    # dimension from input tensor, causing it to have shape [] - a scalar.  We
    # fix it by reducing the rhs constant with a squeeze_left, so it becomes a
    # scalar too. More generally, we squeeze_left on input if it's a constant
    # tensor. This is safe because broadcast will pad dimensions on the left
    # (prepend) to make lhs and rhs shape compatible.
    if network.has_implicit_batch_dimension:
        if isinstance(lhs_val, torch.Tensor):
            lhs_val = squeeze_left(lhs_val)
        if isinstance(rhs_val, torch.Tensor):
            rhs_val = squeeze_left(rhs_val)

    lhs_val = get_trt_tensor(network, lhs_val, f"{name}_lhs", lhs_dtype)
    rhs_val = get_trt_tensor(network, rhs_val, f"{name}_rhs", rhs_dtype)

    # Check the limitation in the doc string.
    if network.has_implicit_batch_dimension:
        if is_lhs_trt_tensor and not is_rhs_trt_tensor:
            assert len(lhs_val.shape) >= len(
                rhs_val.shape
            ), f"{lhs_val.shape} >= {rhs_val.shape}"
        elif not is_lhs_trt_tensor and is_rhs_trt_tensor:
            assert len(rhs_val.shape) >= len(
                lhs_val.shape
            ), f"{rhs_val.shape} >= {lhs_val.shape}"

    lhs_val, rhs_val = broadcast(
        network, lhs_val, rhs_val, f"{name}_lhs", f"{name}_rhs"
    )
    layer = network.add_elementwise(lhs_val, rhs_val, op_type)
    set_layer_name(layer, target, name, source_ir)
    output = layer.get_output(0)
    output.name = output.name + "_" + target.__name__
    return output
