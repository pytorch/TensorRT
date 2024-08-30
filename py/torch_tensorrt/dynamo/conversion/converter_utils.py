import collections
import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, overload

import numpy as np
import tensorrt as trt
import torch
import torch_tensorrt.dynamo.conversion.impl as impl
from torch.fx.node import Argument, Target
from torch.fx.passes.shape_prop import TensorMetadata
from torch_tensorrt import _enums
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    ConverterRegistry,
    DynamoConverterImplSignature,
)

from ..types import Shape, TRTDataType, TRTLayer, TRTTensor

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
        mod_name = mod_stack[1][0]
        node_name = mod_name + "/" + node_name
    else:
        # Try an alternative way to get the module info
        # like the node.meta['source_fn'] attr
        pass

    return node_name


def get_node_io(
    node: torch.fx.Node, constant_mapping: Dict[str, Tuple[Sequence[int], str]]
) -> str:
    """Gets a string representing the node inputs and outputs including tensor shapes and dtypes"""

    def format_tensor_metadata(metadata: Union[Any, Sequence[Any]]) -> str:
        """Formats the metadata for a single node"""
        # If the provided data is a simple TensorMetadata object, parse it
        if isinstance(metadata, TensorMetadata) or issubclass(
            type(metadata), torch.Tensor
        ):
            return f"{tuple(metadata.shape)}@{metadata.dtype}"  # type: ignore
        # If the provided data is a scalar, return it as is
        elif isinstance(metadata, (int, float, bool)):
            return f"{metadata}@Python-{type(metadata)}"
        # If the provided data is a sequence, recursively parse it
        elif isinstance(metadata, collections.abc.Sequence):
            formatted_str = "("
            for meta in metadata:
                formatted_str += format_tensor_metadata(meta) + ", "

            return formatted_str[:-2] + ")"
        else:
            _LOGGER.warning(
                f"Detected unparsable type in node formatting: {type(metadata)}"
            )
            return ""

    # Format input tensors
    metadata_string = "Inputs: ("

    # For each input argument, format it accordingly
    for arg in node.args:
        if isinstance(arg, torch.fx.Node):
            if arg.op == "get_attr":
                shape, dtype = constant_mapping[str(arg)]
                arg_repr = f"{shape}@{dtype}"
            elif arg.meta.get("tensor_meta") is not None:
                arg_repr = format_tensor_metadata(arg.meta["tensor_meta"])
            elif arg.meta.get("val") is not None:
                arg_repr = format_tensor_metadata(arg.meta["val"])
            else:
                arg_repr = ""

            metadata_string += f"{arg}: {arg_repr}, "
        else:
            metadata_string += f"{arg}, "

    metadata_string = (
        metadata_string[:-2] if metadata_string[-1] != "(" else metadata_string
    ) + ")"

    # Format output tensors and arguments
    metadata_string += " | Outputs: ("
    if node.op == "get_attr":
        shape, dtype = constant_mapping[str(node)]
        node_repr = f"{shape}@{dtype}"
    elif node.meta.get("tensor_meta") is not None:
        node_repr = format_tensor_metadata(node.meta["tensor_meta"])
    elif node.meta.get("val") is not None:
        node_repr = format_tensor_metadata(node.meta["val"])
    else:
        node_repr = ""
    metadata_string += f"{node}: {node_repr}, "
    metadata_string = metadata_string[:-2] + ")"

    return metadata_string


def is_only_operator_on_placeholder(node: torch.fx.Node) -> bool:
    """Detects whether a call_function node is the only operator on a placeholder"""
    # Returns true if the node operates on a placeholder and is a direct output
    return (
        node.op == "call_function"
        and any(
            arg.op == "placeholder"
            for arg in node.args
            if isinstance(arg, torch.fx.Node)
        )
        and any(user.op == "output" for user in list(node.users.keys()))
    )


def cast_trt_tensor(
    ctx: ConversionContext,
    input_val: TRTTensor,
    dtype: Union[TRTDataType, torch.dtype, np.dtype, _enums.dtype],
    name: str,
    target: Target = "",
    source_ir: Optional[SourceIR] = None,
) -> TRTTensor:
    """Given a TRT Tensor, convert that Tensor to the specified dtype

    Adds an Identity layer to the network which performs the conversion
    if the input's dtype is different from the cast type. Otherwise returns
    input unchanged

    Args:
        ctx (ConversionContext): A ConversionContext containing the TensorRT network
        input_val (TRTTensor): A TRT Tensor to cast to a new data type
        dtype (TRTDataType, torch.dtype, np.dtype): The data type to cast the input Tensor to
        name (str): Name of the calling layer
        target (Target): Target of calling node
        source_ir (SourceIR): SourceIR of calling converter
    Returns:
        A TensorRT ITensor which has been casted to the specified dtype
    """
    trt_dtype = _enums.dtype._from(dtype).to(trt.DataType)

    if input_val.dtype != trt_dtype:
        source_ir = source_ir if source_ir is not None else SourceIR.UNKNOWN
        target_str = ConverterRegistry.qualified_name_or_str(target)
        target_name = f"{source_ir}_ops{('.' + target_str) if target_str else ''}"

        cast_layer = ctx.net.add_cast(input_val, trt_dtype)
        cast_layer.name = f"Cast ITensor {input_val.name} from {input_val.dtype} to {trt_dtype} - [{target_name}]-[{name}]"
        return cast_layer.get_output(0)
    else:
        return input_val


def cast_int_int_div_trt_tensor(
    ctx: ConversionContext,
    lhs_val: TRTTensor,
    rhs_val: TRTTensor,
    name: str,
) -> List[TRTTensor]:
    """
    Given two `int` data type TRT Tensor to div operation, cast the TRT Tensor to float type
    Args:
        ctx (ConversionContext): A ConversionContext object
        lhs_val (TRTTensor): A TRT Tensor numerator
        rhs_val (TRTTensor): A TRT Tensor numerator
        name (str): Name of calling layer
    Returns:
        A list of lhs_val and rhs_val casted to the appropriate datatype
    """
    if lhs_val.dtype == trt.int32 and rhs_val.dtype == trt.int32:
        lhs_val = cast_trt_tensor(ctx, lhs_val, trt.float32, name)
        rhs_val = cast_trt_tensor(ctx, rhs_val, trt.float32, name)
    return [lhs_val, rhs_val]


def broadcastable(
    a: Union[TRTTensor, np.ndarray], b: Union[TRTTensor, np.ndarray]
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


def broadcast_to_same_shape(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    lhs_val: TRTTensor,
    rhs_val: TRTTensor,
) -> Tuple[TRTTensor, TRTTensor]:
    """Broadcast ITensors `lhs_val` and `rhs_val` to the same shape. If the shapes are already the same, return the
    original tensors. If the shapes are different, broadcast the tensors to the same shape.

    This helper function is different from fx/converter_utils.broadcast.
    fx/converter_utils.broadcast only broadcasts two ITensors to the same number of dimensions (ranks)
    by prepending 1s, while this function broadcasts two ITensors to the same shape.

    For example, we have original ITensors: lhs_val.shape: (2, 3) rhs_val.shape: (2, 2, 1, 3)
    If calling fx/converter_utils.broadcast, lhs_val.shape: (1, 1, 2, 3) lhs_val.shape: (2, 2, 1, 3).
    If calling this function broadcast_to_same_shape, lhs_val.shape: (2, 2, 2, 3) lhs_val.shape: (2, 2, 2, 3).

    Args:
        lhs_val (TRTTensor): A TensorRT ITensor.
        rhs_val (TRTTensor): A TensorRT ITensor.

    Returns:
        Tuple[TRTTensor, TRTTensor]: Two TensorRT ITensors that are broadcasted to the same shape

    """
    lhs_val, rhs_val = broadcast(ctx, lhs_val, rhs_val, f"{name}_lhs", f"{name}_rhs")

    lhs_val_shape = lhs_val.shape
    rhs_val_shape = rhs_val.shape

    if tuple(lhs_val_shape) != tuple(rhs_val_shape):
        rank = len(lhs_val_shape)
        expanded_dims = [-1] * len(lhs_val_shape)

        for dim in range(rank):
            expanded_dims[dim] = max(lhs_val_shape[dim], rhs_val_shape[dim])

        expanded_shape = tuple(expanded_dims)

        if lhs_val_shape != expanded_shape:
            lhs_val = impl.slice.expand(
                ctx,
                target,
                source_ir,
                f"{name}_expand_lhs_val",
                lhs_val,
                expanded_shape,
            )

        if rhs_val_shape != expanded_shape:
            rhs_val = impl.slice.expand(
                ctx,
                target,
                source_ir,
                f"{name}_expand_rhs_val",
                rhs_val,
                expanded_shape,
            )

    return lhs_val, rhs_val


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
        raise AssertionError(f"Object {val} could not be extended to tuple")


def cast_int_or_float_to_bool(
    ctx: ConversionContext, name: str, tensor: TRTTensor
) -> TRTTensor:
    if tensor.dtype != trt.bool:
        return cast_trt_tensor(ctx, tensor, trt.bool, name)

    return tensor


def create_constant(
    ctx: ConversionContext,
    value: Union[int, float, bool, np.ndarray, torch.Tensor],
    name: str,
    dtype: Optional[Union[torch.dtype, np.dtype, TRTDataType, _enums.dtype]],
    min_rank: Optional[int] = 1,
) -> TRTTensor:
    """
    Add a TensorRT constant layer whose value is `value` to `ctx.net`.
    Args:
        ctx (ConversionContext): A TensorRT ConversionContext to which we want to add
            a constant layer.
        value (Union[int, float, bool, np.ndarray, torch.Tensor]): A literal value, Numpy array,
            or a PyTorch tensor that will be used as value of the added TensorRT Constant layer.
        name (str): Name of the added TensorRT Constant layer.
        dtype (Optional[Union[torch.dtype, np.dtype, TRTDataType]]):
            If a dtype is given, we will convert the type of the given `value` to this dtype.
        min_rank (int): minimum rank of the constant tensor.
    Returns:
        A TensorRT ITensor that represents the given value.
    """
    shape = (1,)
    # Rank 0 constant is required in IFillLayer inputs.
    if min_rank == 0:
        shape = trt.Dims()
    numpy_value = to_numpy(value, dtype)
    constant = ctx.net.add_constant(
        shape if isinstance(value, (int, float, bool)) else value.shape,
        numpy_value.copy() if isinstance(numpy_value, np.ndarray) else numpy_value,
    )
    constant.name = name
    return constant.get_output(0)


def get_trt_tensor(
    ctx: ConversionContext,
    input_val: Any,
    name: str,
    dtype: Optional[Union[torch.dtype, np.dtype, TRTDataType, _enums.dtype]] = None,
    min_rank: int = 1,
) -> TRTTensor:
    """
    Given a value of random type, we try to convert it to a TensorRT ITensor.
    An runtime error is raised if we're not able to do that.
    Args:
        ctx (ConversionContext): A TensorRT ConversionContext. If we want to
            add a TensorRT Constant layer, we will add it to this network.
        input_val (Any): An value that we want to convert to a TensorRT ITensor.
        name (str): The name of the created TensorRT Constant layer if there's
            one.
        dtype (Optional[Union[torch.dtype, np.dtype, TRTDataType]]):
            If dtype is provided, the given value will be converted to this dtype.
        min_rank (int): minimum rank of the constant tensor.
    Returns:
        A TensorRT ITensor that represents the given value.
    """
    # If the input is 64-bit, cast it to 32-bit for TRT freezing
    if isinstance(input_val, torch.Tensor) and ctx.compilation_settings.truncate_double:
        if input_val.dtype == torch.float64:
            input_val = input_val.to(torch.float32)
    elif isinstance(input_val, np.ndarray) and ctx.compilation_settings.truncate_double:
        if input_val.dtype == np.float64:
            input_val = input_val.astype(np.float32)

    if isinstance(input_val, (torch.Tensor, np.ndarray, int, float, bool)):
        return create_constant(ctx, input_val, name, dtype, min_rank)
    elif isinstance(input_val, TRTTensor):
        return input_val
    else:
        raise AssertionError(f"Cannot convert {input_val} to TRT constant")


@overload
def get_positive_dim(dim: int, dim_size: int) -> int: ...


@overload
def get_positive_dim(dim: Sequence[int], dim_size: int) -> Tuple[int, ...]: ...


def get_positive_dim(
    dim: Union[int, Sequence[int]], dim_size: int
) -> Union[int, Tuple[int, ...]]:
    """
    Given an integer number or tuple that represents dimension(s) in the array,
    transform it to a positive integer dim if it's negative.
    Otherwise, truncate it to the dimension size

    Args:
        dim (Union[int, Sequence[int]]): A integer or Sequence of integers that represent dimension(s) in an array.
        dim_size (int): The size of the dimension in the array.

    Returns:
        A positive integer or tuple of integers that represent the same dimension as the given dim.
    """

    def positive_dim(d: int) -> int:
        if d < 0:
            return d % dim_size
        else:
            return min(d, dim_size)

    return (
        positive_dim(dim)
        if isinstance(dim, int)
        else tuple(positive_dim(d) for d in dim)
    )


def enforce_tensor_types(
    type_dictionary: Dict[Union[int, str], Tuple[Union[TRTTensor, np.ndarray], ...]],
    promote: bool = True,
) -> Callable[[DynamoConverterImplSignature], DynamoConverterImplSignature]:
    """Decorator to enforce tensor types for input arguments to converters

    Keys in the type dictionary must be integers if they refer to a positional
    argument in args, or strings if they refer to a keyword argument in kwargs

    Values must be tuples of data types denoting the approved data types for a given position
    The approved types are TRTTensor, np.ndarray, and torch.Tensor.

    Note: torch.Tensor cannot be present without np.ndarray

    The promote argument controls whether tensors will be promoted if they are of the
    incorrect format
    """
    assert all(
        isinstance(key, (int, str)) for key in type_dictionary
    ), "Invalid key for type enforcement"
    assert all(
        (
            isinstance(val, tuple)
            and not (torch.Tensor in val and np.ndarray not in val)
            and all((dtype in (TRTTensor, np.ndarray, torch.Tensor)) for dtype in val)
        )
        for val in type_dictionary.values()
    ), (
        "Invalid value(s) specified in type enforcement."
        "Note that torch.Tensor cannot be present as a type without np.ndarray."
    )

    def wrapper(func: DynamoConverterImplSignature) -> DynamoConverterImplSignature:
        @functools.wraps(func)
        def convert_with_type_enforcement(
            ctx: ConversionContext,
            target: Target,
            args: Tuple[Argument, ...],
            kwargs: Dict[str, Argument],
            name: str,
        ) -> Union[TRTTensor, Sequence[TRTTensor]]:
            new_args = args
            new_kwargs = {**kwargs}
            new_value = None

            # Go through type dictionary and promote types accordingly
            for index, approved_dtypes in type_dictionary.items():
                # Referencing an arg
                if isinstance(index, int):
                    candidate = args[index]
                # Referencing a kwarg
                elif isinstance(index, str):
                    candidate = kwargs[index]

                # If the candidate Tensor is already an approved type, do nothing
                if isinstance(candidate, approved_dtypes):
                    continue
                # If the candidate Tensor is not an approved type, but promotion is disabled, error
                elif not promote:
                    raise AssertionError(
                        f"Detected argument at index {index} had type {type(candidate)} "
                        f"which is not one of the approved types {approved_dtypes}"
                    )

                promoted = False

                # Type-promotion preference order depends on tuple order
                for dtype in approved_dtypes:
                    # Currently, we do not cast to Torch tensor, due to issues with such casts
                    # in FakeTensor contexts
                    if dtype == np.ndarray and not isinstance(candidate, TRTTensor):
                        new_value = to_numpy(candidate)
                        promoted = True
                        break
                    # As a fallback, freeze tensors to IConstantLayers if they cannot be handled as Numpy arrays
                    elif dtype == TRTTensor:
                        _LOGGER.debug(
                            f"Freezing tensor {name}_constant_{index} to TRT IConstantLayer"
                        )
                        new_value = get_trt_tensor(
                            ctx, candidate, name + f"_constant_{index}"
                        )
                        promoted = True
                        break

                if not promoted:
                    raise AssertionError(
                        f"Argument {candidate} at index {index} was not able to be "
                        f"converted to one of the following types: {approved_dtypes}"
                    )

                # Reassemble args or kwargs if the value was modified
                if isinstance(index, int):
                    new_args = new_args[:index] + (new_value,) + new_args[index + 1 :]
                elif isinstance(index, str):
                    new_kwargs[index] = new_value

            return func(ctx, target, new_args, new_kwargs, name)

        return convert_with_type_enforcement

    return wrapper


def to_numpy(
    value: Optional[Union[torch.Tensor, np.ndarray, int, float, bool]],
    dtype: Optional[Union[torch.dtype, np.dtype, TRTDataType, _enums.dtype]] = None,
) -> Optional[np.ndarray]:
    """
    Convert a PyTorch Tensor, Numpy array, or scalar to a Numpy Array. If the tensor is
    quantized it will be dequantized first.
    Args:
        value (Optional[Union[torch.Tensor, np.ndarray, int, float, bool]]):
            A PyTorch tensor, Numpy array, int, float, or bool
        dtype (Optional[Union[torch.dtype, np.dtype, TRTDataType]]):
            If a dtype is given, we will convert the type of the given `value` to this dtype.
    Returns:
        A Numpy array or None, if the input was None.
    """
    output = None

    if value is None or isinstance(value, np.ndarray):
        output = value

    elif isinstance(value, torch.Tensor):
        if value.is_quantized:
            value = value.dequantize()
        elif value.dtype == torch.bfloat16:
            # TODO: Remove when numpy has a BF16 type
            value = value.to(torch.float)

        output = value.cpu().detach().contiguous().numpy()

    elif isinstance(value, int):
        output = np.array([value], dtype=np.int32)

    elif isinstance(value, float):
        output = np.array([value], dtype=np.float32)

    elif isinstance(value, bool):
        output = np.array([value], dtype=np.bool_)

    if isinstance(output, np.ndarray) or output is None:
        return (
            output
            if (dtype is None or output is None)
            else output.astype(_enums.dtype._from(dtype).to(np.dtype, use_default=True))
        )
    else:
        raise AssertionError(
            f"to_numpy can only be called on None, bool, int, float, np.ndarray, or torch.Tensor, got: {value}"
        )


def flatten_dims(
    input: Sequence[Union[TRTTensor, torch.Tensor, np.ndarray]],
    start_dim: int,
    end_dim: int,
) -> Tuple[int, ...]:
    """
    Given an input, start and end indices of dimension,
    this function will return a flattened new shape.

    Args:
        input (Sequence[Union[TRTTensor, torch.Tensor, np.ndarray]]):
            an input value waiting to be flattened
        start_dim (int): the first dim to flatten
        end_dim (int): the last dim to flatten (this dim is included)

    Returns:
        Tuple[int]: new_shape
    """
    shape = input.shape
    dim_size = len(shape)
    start_dim = get_positive_dim(start_dim, dim_size)
    end_dim = get_positive_dim(end_dim, dim_size)

    num_elements = 1
    for i in range(start_dim, end_dim + 1):
        num_elements *= shape[i]

    new_shape = tuple(shape[:start_dim]) + (num_elements,) + tuple(shape[end_dim + 1 :])

    return new_shape


def append(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    original_tensor: TRTTensor,
    new_value: Union[TRTTensor, int, float, torch.Tensor, np.ndarray],
    dim: int = 0,
) -> TRTTensor:
    """
    Append a new value to the last of the original tensor along the specified dimension (default 0).
    For example, if the original tensor is [1, 2, 3], the new value is 4, and the dim is 0,
    the new tensor will be [1, 2, 3, 4].

    Args:
        ctx (ConversionContext): A ConversionContext containing the TensorRT network
        target (Target): Target of calling node
        source_ir (Optional[SourceIR]): SourceIR of calling converter
        name (str): Name of the calling layer
        original_tensor (TRTTensor): A TRTTensor to append the new value to
        new_value (Union[TRTTensor, int, float, torch.Tensor, np.ndarray]): A new value to append
        dim (int, optional): Dimension to append the new value. Defaults to 0.

    Returns:
        TRTTensor: A new TRTTensor that is the result of appending the new value to the original tensor
    """
    if isinstance(new_value, (int, float)):
        new_value = np.array([new_value])
    new_value = get_trt_tensor(ctx, new_value, name, original_tensor.dtype)

    return impl.cat.cat(
        ctx,
        target,
        source_ir,
        f"{name}_concat",
        [original_tensor, new_value],
        get_positive_dim(dim, len(original_tensor.shape)),
    )


def set_item(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    original_tensor: TRTTensor,
    index: int,
    new_value: Union[TRTTensor, int, float, torch.Tensor, np.ndarray],
) -> TRTTensor:
    """
    Set a new value to the original tensor at the specified index. For example,
    if the original tensor is [1, 2, 3], the new value is 4, and the index is 1,
    the new tensor will be [1, 4, 3].
    If the index is out of bound, the new value will be appended to the end.

    Args:
        ctx (ConversionContext): A ConversionContext containing the TensorRT network
        target (Target): Target of calling node
        source_ir (Optional[SourceIR]): SourceIR of calling converter
        name (str): Name of the calling layer
        original_tensor (TRTTensor): A TRTTensor to set the new value to
        index (int): The index to set the new value
        new_value (Union[TRTTensor, int, float, torch.Tensor, np.ndarray]): A new value to set

    Returns:
        TRTTensor: A new TRTTensor that is the result of setting the new value to the original tensor
    """
    if isinstance(new_value, (int, float)):
        new_value = np.array([new_value])
    new_value = get_trt_tensor(ctx, new_value, name, original_tensor.dtype)

    len_original_tensor = original_tensor.shape[0]
    index = get_positive_dim(index, len_original_tensor)

    front_tensor = impl.slice.slice_op(
        ctx,
        target,
        source_ir,
        f"{name}_slice_front",
        original_tensor,
        dim=0,
        start=0,
        stop=index,
        step=1,
    )
    rear_tensor = impl.slice.slice_op(
        ctx,
        target,
        source_ir,
        f"{name}_slice_rear",
        original_tensor,
        dim=0,
        start=index + 1,
        stop=len_original_tensor,
        step=1,
    )

    ans = impl.cat.cat(
        ctx,
        target,
        source_ir,
        f"{name}_concat",
        [front_tensor, new_value, rear_tensor],
        0,
    )
    return ans


def calculate_strides(shape: Sequence[int]) -> Sequence[int]:
    """
    Calculate the strides for a given shape of a multi-dimensional array.

    The output stride for each dimension indicates the number of elements to skip in
    memory to move to the next element along that dimension. The last dimension always
    has a stride of 1 because elements are stored contiguously along this dimension.

    Example:
        For a 3-dimensional array with shape [2, 3, 4]:
        - shape = [2, 3, 4]
        - The function will calculate the strides as follows:
            1. Initialize strides: [1, 1, 1]
            2. Calculate strides for each dimension from right to left:
               - For i = 1: strides[1] = strides[2] * shape[2] = 1 * 4 = 4
               - For i = 0: strides[0] = strides[1] * shape[1] = 4 * 3 = 12
            - Final strides: [12, 4, 1]

        Therefore, the output will be [12, 4, 1].

        This means:
        - To move along the first dimension, skip 12 elements.
        - To move along the second dimension, skip 4 elements.
        - To move along the third dimension, skip 1 element.
    """
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def broadcast(
    ctx: ConversionContext,
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
        ctx (ConversionContext): A ConversionContext containing the TensorRT network
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
        b = prepend_ones(ctx, b, f"{b_name}_broadcast", diff)
    elif diff < 0:
        a = prepend_ones(ctx, a, f"{a_name}_broadcast", -diff)

    return a, b


def get_axes_for_reduce_op(
    dim: Union[int, Sequence[int]],
) -> int:
    """
    TensorRT reduce layer relies on the binary representation of axes to
    determine which dims to reduce. For example, if we want to reduce on
    dim 1 and 2 then axes should be 6(110).

    Args:
        dim (Union[int, Sequence[int]]): An integer or a sequence of integers
            that will be used to generate axes for TensorRT.

    Returns:
        An integer which binary form can be used as axes for TensorRT reduce
        layer.
    """
    if isinstance(dim, int):
        dim = (dim,)

    axes = 0
    for d in dim:
        axes |= 1 << d

    return axes


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
    return count > 0


def prepend_ones(
    ctx: ConversionContext,
    tensor: TRTTensor,
    name: str,
    num_prepend_ones: int,
) -> TRTTensor:
    """
    Prepend 1s to the shape of TensorRT ITensor `tensor`.

    Args:
        ctx (ConversionContext): A ConversionContext containing the TensorRT network
        tensor (TRTTensor): A TensorRT tensor.
        name (str): Name of the TensorRT Shuffle layer which is used to prepend
            1s.
        num_prepend_ones (int): Number of 1s that will be prepend.

    Returns:
        A Tensorrt ITensor which contains the same value as `tensor` but with
        more 1s prepended to the beginning of `tensor` shape.
    """
    layer = ctx.net.add_shuffle(tensor)

    # If there're dynamic dim in tensor's shape, we need to use shape layer to
    # compute the final shape.
    if has_dynamic_shape(tensor.shape):
        tensor_shape_layer = ctx.net.add_shape(tensor)
        tensor_shape = tensor_shape_layer.get_output(0)
        tensor_shape = cast_trt_tensor(
            ctx, tensor_shape, trt.int32, name + "shape_casted", "shape"
        )
        tensor_shape_layer.name = f"{name}_broadcast_orig_shape"
        prepend_shape_layer = ctx.net.add_constant(
            (num_prepend_ones,), np.ones((num_prepend_ones,), dtype=np.int32)
        )
        prepend_shape_layer.name = f"{name}_broadcast_prepend_ones"
        reshape_dim_layer = ctx.net.add_concatenation(
            [prepend_shape_layer.get_output(0), tensor_shape]
        )
        reshape_dim_layer.axis = 0
        reshape_dim_layer.name = f"{name}_broadcast_final_shape"
        layer.set_input(1, reshape_dim_layer.get_output(0))
    else:
        layer.reshape_dims = (1,) * num_prepend_ones + tuple(tensor.shape)

    layer.name = name
    return layer.get_output(0)


def set_layer_name(
    layer: TRTLayer,
    target: Union[Target, torch.nn.Module, str],
    name: str,
    source_ir: Optional[SourceIR] = None,
) -> None:
    """
    Set the TensorRT layer name to "[TensorRT Layer Type]_[Original Op Name]_[FX Node Name with Suffix]"

    Args:
        layer (TRTLayer): A TensorRT layer of which we want to set the name.
        target (Target): A fx node.target or submodule. For call_function node, it's the function that
            the node represents.
        name (str): Consists of fx node.name with optional suffix.
        source_ir: (Optional[SourceIR]): The IR producing the op.
    """

    source_ir = source_ir if source_ir is not None else SourceIR.UNKNOWN

    target_name = (
        f"{source_ir}_ops.{target}"
        if isinstance(target, str)
        else f"{source_ir}_ops.{target.__name__}"
    )
    layer.name = f"[{layer.type.name}]-[{target_name}]-[{name}]"
