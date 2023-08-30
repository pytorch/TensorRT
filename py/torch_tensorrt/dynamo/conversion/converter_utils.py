import functools
import logging
import re
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
from torch.fx.node import Target
from torch_tensorrt.fx.converters.converter_utils import (
    Frameworks,
    get_axes_for_reduce_op,
    to_numpy,
    unified_dtype_converter,
)
from torch_tensorrt.fx.types import TRTDataType, TRTNetwork, TRTTensor

from .._SourceIR import SourceIR
from .converter_registry import ConverterRegistry

_LOGGER: logging.Logger = logging.getLogger(__name__)


def get_node_name(node: torch.fx.Node) -> str:
    # nn_module_stack preserves the call stack of pytorch nn.modules
    # The call stack contains a detailed name of the module
    # which shows exactly where the module is located in the
    # network architecture.
    stack_item = node.meta.get("nn_module_stack", None)
    # The current node is the last item in the stack
    mod_stack = stack_item.popitem() if stack_item else ""
    node_name = str(node)
    if mod_stack:
        mod_name = str(mod_stack[0]).replace("___", "/")
        # Clean up the module name
        mod_name = re.sub("^.*__self", "", mod_name)
        mod_name = re.sub(r"_(\d+)$", r"/\g<1>", mod_name)
        node_name = mod_name + "/" + node_name
    else:
        # Try an alternative way to get the module info
        # like the node.meta['source_fn'] attr
        pass

    _LOGGER.debug(f"Node meta name {node_name}")
    return node_name


def dynamic_unsupported(node: torch.fx.Node) -> bool:
    # Validate that none of the inputs to the node have Dynamic shapes
    assert isinstance(
        node, torch.fx.Node
    ), "Inputs to validator functions must be FX Nodes"

    # Check node value itself
    if getattr(node.meta["val"], "_has_symbolic_sizes_strides", False):
        return False

    # Check node arguments individually
    if any(
        getattr(arg.meta["val"], "_has_symbolic_sizes_strides", False)
        for arg in node.args
        if isinstance(arg, torch.fx.Node)
    ):
        return False

    # Check node keyword arguments individually
    if any(
        getattr(kwarg.meta["val"], "_has_symbolic_sizes_strides", False)
        for kwarg in node.kwargs.values()
        if isinstance(kwarg, torch.fx.Node)
    ):
        return False

    return True


def cast_trt_tensor(
    network: TRTNetwork,
    input_val: TRTTensor,
    dtype: TRTDataType,
    name: str,
    target: Target = "",
    source_ir: Optional[SourceIR] = None,
) -> TRTTensor:
    """
    Given a TRT Tensor, convert that Tensor to the specified dtype
    Adds an Identity layer to the network which performs the conversion
    Args:
        network (TRTNetwork): A TensorRT network
        input_val (TRTTensor): A TRT Tensor to cast to a new data type
        dtype (TRTDataType, torch.dtype, np.dtype): The data type to cast the input Tensor to
        name (str): Name of the calling layer
        target (Target): Target of calling node
        source_ir (SourceIR): SourceIR of calling converter
    Returns:
        A TensorRT ITensor which has been casted to the specified dtype
    """
    trt_dtype = unified_dtype_converter(dtype, Frameworks.TRT)

    if input_val.dtype != trt_dtype:
        source_ir = source_ir if source_ir is not None else SourceIR.UNKNOWN
        target_str = ConverterRegistry.qualified_name_or_str(target)
        target_name = f"{source_ir}_ops{('.' + target_str) if target_str else ''}"

        identity_layer = network.add_identity(input_val)
        identity_layer.set_output_type(0, trt_dtype)
        identity_layer.name = f"Cast ITensor {input_val.name} from {input_val.dtype} to {trt_dtype} - [{target_name}]-[{name}]"
        return identity_layer.get_output(0)
    else:
        return input_val


def cast_int_int_div_trt_tensor(
    network: TRTNetwork,
    lhs_val: TRTTensor,
    rhs_val: TRTTensor,
    name: str,
) -> List[TRTTensor]:
    """
    Given two `int` data type TRT Tensor to div operation, cast the TRT Tensor to float type
    Args:
        network (TRTNetwork): A TensorRT network
        lhs_val (TRTTensor): A TRT Tensor numerator
        rhs_val (TRTTensor): A TRT Tensor numerator
        name (str): Name of calling layer
    Returns:
        A list of lhs_val and rhs_val casted to the approriate datatype
    """
    if (lhs_val.dtype == trt.int8 or lhs_val.dtype == trt.int32) and (
        rhs_val.dtype == trt.int8 or rhs_val.dtype == trt.int32
    ):
        lhs_val = cast_trt_tensor(network, lhs_val, trt.float32, name)
        rhs_val = cast_trt_tensor(network, rhs_val, trt.float32, name)
    return [lhs_val, rhs_val]


def broadcastable(
    a: TRTTensor,
    b: TRTTensor,
) -> bool:
    "Check if two tensors are broadcastable according to torch rules"
    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)

    # check from the trailing
    diff = len(a_shape) - len(b_shape)

    # Validate tensors have same rank and shape
    if diff == 0 and all(a_shape[i] == b_shape[i] for i in range(len(a_shape))):
        return True

    # Left-pad the shorter dimension with ones
    if diff > 0:
        b_shape = (1,) * abs(diff) + b_shape
    else:
        a_shape = (1,) * abs(diff) + a_shape

    # Validate one of the following conditions for broadcastability per-dimension
    # 1. Equal number of dimensions or 2. Dimension has shape 1
    for i in range(len(a_shape)):
        if not (a_shape[i] == b_shape[i] or a_shape[i] == 1 or b_shape[i] == 1):
            return False
    return True


get_axes_for_reduce_op = functools.partial(
    get_axes_for_reduce_op, has_implicit_batch_dimension=False
)


def extend_attr_to_tuple(
    val: Any,
    num_elem: int,
) -> Tuple[Any, ...]:
    """
    If `val` is not a tuple or a list, then we make a tuple of size `num_elem` by
    replicating `val` `num_elem` times.

    Args:
        val (Any): Value that we want to process.

    Returns:
        A tuple.
    """
    if not isinstance(val, (tuple, list)):
        val = (val,) * num_elem
    elif len(val) == 1:
        val = (val[0],) * num_elem

    if isinstance(val, list):
        val = tuple(val)

    if isinstance(val, tuple):
        return val
    else:
        raise AssertionError(f"Could not extend attribute {val}")


def create_constant(
    network: TRTNetwork,
    value: Union[int, float, np.ndarray, torch.Tensor],
    name: str,
    dtype: Optional[Union[torch.dtype, np.dtype, TRTDataType]],
) -> TRTTensor:
    """
    Add a TensorRT constant layer whose value is `value` to `network`.
    Args:
        network (TRTNetwork): A TensorRT network to which we want to add
            a constant layer.
        value (Union[int, float, np.ndarray, torch.Tensor]): A literal value, Numpy array,
            or a PyTorch tensor that will be used as value of the added TensorRT Constant layer.
        name (str): Name of the added TensorRT Constant layer.
        dtype (Optional[Union[torch.dtype, np.dtype, TRTDataType]]):
            If a dtype is given, we will convert the type of the given `value` to this dtype.
    Returns:
        A TensorRT ITensor that represents the given value.
    """
    constant = network.add_constant(
        (1,) if isinstance(value, (int, float)) else value.shape,
        to_numpy(value, dtype).copy(),
    )
    constant.name = name
    return constant.get_output(0)


def get_trt_tensor(
    network: TRTNetwork,
    input_val: Any,
    name: str,
    dtype: Optional[Union[torch.dtype, np.dtype, TRTDataType]] = None,
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
        dtype (Optional[Union[torch.dtype, np.dtype, TRTDataType]]):
            If dtype is provided, the given value will be converted to this dtype.
    Returns:
        A TensorRT ITensor that represents the given value.
    """
    # TRT can not add constant for bool type. We do a work around to 1) cast it to int and 2)cast to bool later
    # This is useful for logical operations which require input to be bool type
    if isinstance(input_val, bool):
        input_val = int(input_val)
    elif isinstance(input_val, torch.Tensor) and (
        input_val.dtype == torch.bool or input_val.dtype == torch.int64
    ):
        input_val = input_val.to(torch.int32)
    elif isinstance(input_val, np.ndarray) and (
        input_val.dtype == np.bool_ or input_val.dtype == np.int64
    ):
        input_val = input_val.astype(np.int32)

    if isinstance(input_val, (torch.Tensor, np.ndarray, int, float)):
        return create_constant(network, input_val, name, dtype)
    elif isinstance(input_val, TRTTensor):
        return input_val
    else:
        raise AssertionError(f"Cannot convert {input_val} to TRT constant")
