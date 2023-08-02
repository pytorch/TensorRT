import torch

from torch_tensorrt.fx.types import (
    TRTDataType,
    TRTNetwork,
    TRTTensor,
)

from torch_tensorrt.fx.converters.converter_utils import (
    unified_dtype_converter,
    Frameworks,
)
import logging
import re
import tensorrt as trt
from typing import List

_LOGGER: logging.Logger = logging.getLogger(__name__)


def get_node_name(node):
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
        mod_name = re.sub("_(\d+)$", "/\g<1>", mod_name)
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
) -> TRTTensor:
    """
    Given a TRT Tensor, convert that Tensor to the specified dtype
    Adds an Identity layer to the network which performs the conversion
    Args:
        network (TRTNetwork): A TensorRT network
        input_val (TRTTensor): A TRT Tensor to cast to a new data type
        dtype (TRTDataType): The TRTDataType to cast the input Tensor to
        name (str): Name of the calling layer
    Returns:
        A TensorRT ITensor which has been casted to the specified dtype
    """
    trt_dtype = unified_dtype_converter(dtype, Frameworks.TRT)

    if input_val.dtype != trt_dtype:
        identity_layer = network.add_identity(input_val)
        identity_layer.set_output_type(0, trt_dtype)
        identity_layer.name = f"Cast ITensor {input_val.name} from {input_val.dtype} to {trt_dtype} - {name}"
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
    return list((lhs_val, rhs_val))


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
