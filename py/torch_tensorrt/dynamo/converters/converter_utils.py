import operator
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
from torch.fx.node import Argument, Target

from ..utils import torch_dtype_from_trt

from ..types import (
    Shape,
    TRTDataType,
    TRTElementWiseOp,
    TRTLayer,
    TRTNetwork,
    TRTPlugin,
    TRTPluginFieldCollection,
    TRTTensor,
)


def add_binary_elementwise_layer(
    network: TRTNetwork,
    lhs_val: Union[int, float, TRTTensor, torch.Tensor],
    rhs_val: Union[int, float, TRTTensor, torch.Tensor],
    op_type: trt.ElementWiseOperation,
    target: Target,
    name: str,
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
        lhs_val (TRTTensor): Left operand of the binary operation. Could
            be a TensorRT tensor, a PyTorch tensor or a simple value.
        rhs_val (TRTTensor): Right operand of the binary operation. Similar
            to lhs_val.
        op_type (trt.ElementWiseOperation): Type of the TensorRT elementwise binary operation.
        target (Target): Target of fx node.
        name (str): The name we want to assign to the created TensorRT layer.

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
    set_layer_name(layer, target, name)
    output = layer.get_output(0)
    output.name = output.name + "_" + target.__name__
    return output


def broadcast(
    network: TRTNetwork,
    a: TRTTensor,
    b: TRTTensor,
    a_name: str,
    b_name: str,
    preset_diff: int = 0,
) -> Tuple[TRTTensor, TRTTensor]:
    """
    Broadcast two TensorRT tensors to the same number of dimensions by
    prepending 1s to the tensor with less number of dimensions.

    Args:
        network (TRTNetwork): TensorRT network object.
        a (TRTTensor): A TensorRT ITensor.
        b (TRTTensor): A TensorRT ITensor.
        a_name (str): Name of tensor a.
        b_name (str): Name of tensor b.
        preset_diff (int): The difference of number of dimensions after broadcast.
            A positive number means after broadcast, tensor `a` would have `preset_diff`
            more dimensions than `b`. This is used in matmul, since we need to broadcast
            tensors but not always to the same number of dimension. The reason is that
            matmul supports Matrix x Vector and in this case broadcasted vector should
            have 1 less number of dimensions than the matrix tensor.

    Returns:
        Two TensorRT ITensors that are broadcasted to the same number of dimensions.
    """
    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)

    diff = len(a_shape) - len(b_shape) - preset_diff
    if diff > 0:
        b = prepend_ones(network, b, f"{b_name}_broadcast", diff)
    elif diff < 0:
        a = prepend_ones(network, a, f"{a_name}_broadcast", -diff)

    return a, b


def broadcastable(
    a: TRTTensor,
    b: TRTTensor,
) -> bool:
    "Check if two tensors are broadcastable according to torch rules"
    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)
    # check from the trailing
    diff = len(a_shape) - len(b_shape)
    if diff == 0:
        return True
    if diff > 0:
        max = len(a_shape)
        min = len(b_shape)
        greater_tensor = a_shape
        lesser_tensor = b_shape
    elif diff < 0:
        max = len(b_shape)
        min = len(a_shape)
        greater_tensor = b_shape
        lesser_tensor = a_shape
    j = min - 1
    for i in range(max - 1, diff - 1, -1):
        if not (
            greater_tensor[i] != lesser_tensor[j]
            and (greater_tensor[i] == 1 or lesser_tensor[i] == 1)
        ):
            return False
    return True


def create_constant(
    network: TRTNetwork,
    value: Union[int, float, torch.Tensor],
    name: str,
    dtype: Optional[torch.dtype],
) -> TRTTensor:
    """
    Add a TensorRT constant layer whose value is `value` to `network`.

    Args:
        network (TRTNetwork): A TensorRT network to which we want to add
            a constant layer.
        value (Union[int, float, torch.Tensor]): A literal value or a PyTorch tensor
            that will be used as value of the added TensorRT Constant layer.
        name (str): Name of the added TensorRT Constant layer.
        dtype (Optional[torch.dtype]): If a dtype is given, we will convert the type
            of the given `value` to this dtype.

    Returns:
        A TensorRT ITensor that represents the given value.
    """
    if isinstance(value, int):
        value = torch.IntTensor([value])

    if isinstance(value, float):
        value = torch.Tensor([value])

    if dtype:
        value = value.to(dtype)
    constant = network.add_constant(value.shape, to_numpy(value))
    constant.name = name
    return constant.get_output(0)


def get_positive_dim(dim: int, dim_size: int) -> int:
    """
    Given an integer number that represents a dimension in the array,
    transform it to a positive integer dim if it's negative. Otherwise, do
    nothing.

    Args:
        dim (int): A integer number that represents a dimension in an array.
        dim_size (int): The size of the dimension in the array.

    Returns:
        A positive integer that represent the same dimension as the given dim.
    """
    if dim < 0:
        return dim % dim_size
    return dim


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


def get_shape_with_dynamic_shape(
    network: TRTNetwork,
    shape: Union[list, tuple, torch.Tensor],
    input_val: TRTTensor,
    target: Target,
    name: str,
) -> TRTTensor:
    """
    Prepare the real output tensor shape for dynamic shape mode tensor input.
    How this functions works:
    Assuming the input_val has actual shape [2048, 256, 512], expected reduce operation
    output shape is [-1, 128, 256], this function should return [2048, 128, 256] as the actual
    reduce operation output shape. Steps of calculations are:
        1. get the actual tensor shape of input_val via add_shape layer;
        2. create a all 0 tensor [0, 0, 0];
        3. run elementwise comparision the [0, 0, 0] and [-1, 128, 256] tensor, get a condition tensor [True, False, False];
        4. use the condition tensor [True, False, False] to do selection between [2048, 256, 512] and [-1, 128, 256], replace
           all -1 dynamic shape dimensions with actual batch_size value;
        5. output shape with actual batch_size as [2048, 128, 256]

    Args:
        network (TRTNetwork): TensorRT network object.
        shape: calculated shape of the expected output tensor
        input_val (TRTTensor): A TensorRT ITensor.
        target (Target): Target of fx node.
        name (str): The name we want to assign to the created TensorRT layer.
    Returns:
        TensorRT ITensors that represents the actual shape of the input_val
    """
    # Ger real shape info for input_val
    input_shape = network.add_shape(input_val).get_output(0)

    scale_layer = network.add_constant(
        input_shape.shape, np.ascontiguousarray(shape, dtype=np.int32)
    )
    set_layer_name(scale_layer, target, f"{name}_scale")
    scale_res = scale_layer.get_output(0)

    length = input_shape.shape[0]
    zero_layer = network.add_constant(
        input_shape.shape, to_numpy(torch.zeros((length), dtype=torch.int32))
    )
    set_layer_name(zero_layer, target, f"{name}_zeros")

    condition_val = add_binary_elementwise_layer(
        network,
        scale_res,
        zero_layer.get_output(0),
        trt.ElementWiseOperation.LESS,
        target,
        f"{name}_shape",
    )
    select_layer = network.add_select(condition_val, input_shape, scale_res)
    set_layer_name(select_layer, target, f"{name}_select")
    return select_layer.get_output(0)


def get_trt_tensor(
    network: TRTNetwork, input_val: Any, name: str, dtype: Optional[torch.dtype] = None
) -> TRTTensor:
    """
    Given a value of random type, we try to convert it to a TensorRT ITensor.
    An runtime error is raised if we're not able to do that.

    Args:
        network (TRTNetwork): A TensorRT network. If we want to
            add a TensorRT Constant layer, we will add it to this network.
        input_val (Any): An value that we want to convert to a TensorRT ITensor.
        name (str): The name of the created TensorRT Constant layer if there's
            one.
        dtype (Optional[torch.dtype]): If dtype is provided, the given value
            will be converted to this dtype.

    Returns:
        A TensorRT ITensor that represents the given value.
    """
    # TRT can not add constant for bool type. We do a work around to 1) cast it to int and 2)cast to bool later
    # This is useful for logical operations which require input to be bool type
    if isinstance(input_val, np.ndarray):
        input_val = torch.from_numpy(input_val)
    if isinstance(input_val, bool):
        input_val = int(input_val)
    if isinstance(input_val, torch.Tensor) and input_val.dtype == torch.bool:
        input_val = input_val.to(torch.int32)
    if isinstance(input_val, torch.Tensor) and input_val.dtype == torch.int64:
        input_val = input_val.to(torch.int32)

    if isinstance(input_val, (torch.Tensor, int, float)):
        return create_constant(network, input_val, name, dtype)
    elif not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"Received input {input_val} of name {name} that "
            "is not part of the TensorRT region!"
        )
    else:
        return input_val


def has_dynamic_shape(shape: Shape) -> bool:
    """
    Determine if the given shape has dynamic dim. i.e. if there're -1 in shape.

    Args:
        shape (Shape): Shape of a tensor. Essentially is a sequence of integers.

    Returns:
        A boolean value indicates whether there's dynamic dim in the shape.
    """
    count = 0
    for s in shape:
        count += 1 if s == -1 else 0
    return count


def prepend_ones(
    network: TRTNetwork,
    tensor: TRTTensor,
    name: str,
    num_prepend_ones: int,
) -> TRTTensor:
    """
    Prepend 1s to the shape of TensorRT ITensor `tensor`.

    Args:
        network (TRTNetwork): The TensorRT network that `tensor`
            belongs to.
        tensor (TRTTensor): A TensorRT tensor.
        name (str): Name of the TensorRT Shuffle layer which is used to prepend
            1s.
        num_prepend_ones (int): Number of 1s that will be prepend.

    Returns:
        A Tensorrt ITensor which contains the same value as `tensor` but with
        more 1s prepended to the beginning of `tensor` shape.
    """
    layer = network.add_shuffle(tensor)

    # If there're dynamic dim in tensor's shape, we need to use shape layer to
    # compute the final shape.
    if has_dynamic_shape(tensor.shape):
        tensor_shape_layer = network.add_shape(tensor)
        tensor_shape_layer.name = f"{name}_broadcast_orig_shape"
        prepend_shape_layer = network.add_constant(
            (num_prepend_ones,), np.ones((num_prepend_ones,), dtype=np.int32)
        )
        prepend_shape_layer.name = f"{name}_broadcast_prepend_ones"
        reshape_dim_layer = network.add_concatenation(
            [prepend_shape_layer.get_output(0), tensor_shape_layer.get_output(0)]
        )
        reshape_dim_layer.axis = 0
        reshape_dim_layer.name = f"{name}_broadcast_final_shape"
        layer.set_input(1, reshape_dim_layer.get_output(0))
    else:
        layer.reshape_dims = (1,) * num_prepend_ones + tuple(tensor.shape)

    layer.name = name
    return layer.get_output(0)


def set_layer_name(layer: TRTLayer, target: Target, name: str, is_acc=True) -> None:
    """
    Set the TensorRT layer name to "[TensorRT Layer Type]_[Original Op Name]_[FX Node Name with Suffix]"

    Args:
        layer (TRTLayer): A TensorRT layer of which we want to set the name.
        target (Target): A fx node.target. For call_function node, it's the function that
            the node represents.
        name (str): Consists of fx node.name with optional suffix.
    """
    target_name = (
        target
        if isinstance(target, str)
        else f"acc_ops.{target.__name__}"
        if is_acc
        else f"aten_ops.{target.__name__}"
    )
    layer.name = f"[{layer.type.name}]-[{target_name}]-[{name}]"


def squeeze_left(const: torch.Tensor):
    """
    Squeeze the size-1 dimensions on the left side of the shape tuple.
    PyTorch's `squeeze()` doesn't support passing multiple `dim`s at once, so
    we do it iteratively.
    """
    while len(const.shape) > 0 and const.shape[0] == 1:
        const = const.squeeze(dim=0)
    return const


def to_numpy(tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    """
    Convert a PyTorch Tensor to a Numpy Array. If the tensor is
    quantized it will be dequantized first.

    Args:
        tensor (Optional[torch.Tensor]): A PyTorch tensor or None.

    Returns:
        A Numpy array.
    """

    if tensor is None:
        return tensor

    assert isinstance(
        tensor, torch.Tensor
    ), f"to_numpy can only be called on None or a torch.Tensor, got: {tensor}"
    if tensor.is_quantized:
        tensor = tensor.dequantize()

    return tensor.cpu().detach().contiguous().numpy()
