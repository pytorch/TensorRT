import operator
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from enum import Enum, auto
import numpy as np

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch
from torch.fx.node import Argument, Target

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
from ..utils import torch_dtype_from_trt


class SourceIR(Enum):
    NN = auto()
    ACC = auto()
    ATEN = auto()
    PRIM = auto()
    TORCHTRT_LOWERED = auto()
    UNKNOWN = auto()

    def __str__(self):
        if self == SourceIR.NN:
            return "nn"
        elif self == SourceIR.ACC:
            return "acc"
        elif self == SourceIR.ATEN:
            return "aten"
        elif self == SourceIR.PRIM:
            return "prim"
        elif self == SourceIR.TORCHTRT_LOWERED:
            return "torchtrt_lowered"
        else:
            return "unknown_ir"


def get_trt_plugin(
    plugin_name: str,
    field_collection: List[TRTPluginFieldCollection],
    version: str,
    plugin_namespace: str = "",
) -> TRTPlugin:
    """
    Get a TensorRT plugin based on the given parameters.

    Args:
        plugin_name (str): Name of the plugin.
        field_collection (List[TRTPluginFieldCollection]): Parameters that needed
            to create a plugin using the plugin creator.
        version (str): Version of the plugin.
        plugin_namespace (str): Namespace of the plugin.

    Returns:
        A TensorRT plugin that can be added to TensorRT network as Plugin layer.
    """
    # print the registered plugins
    # PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list
    # for plugin_creator in PLUGIN_CREATORS:
    #     print(plugin_creator.name)

    plugin_registry = trt.get_plugin_registry()
    plugin_creator = plugin_registry.get_plugin_creator(
        plugin_name, version, plugin_namespace
    )
    assert plugin_creator, f"Unabled to find plugin creator with name {plugin_name}"
    plugin = plugin_creator.create_plugin(
        name=plugin_name, field_collection=field_collection
    )

    assert plugin is not None, f"Plugin: {plugin_name} could not be fetched"
    return plugin


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


def set_layer_name(
    layer: TRTLayer, target: Target, name: str, source_ir: Optional[SourceIR] = None
) -> None:
    """
    Set the TensorRT layer name to "[TensorRT Layer Type]_[Original Op Name]_[FX Node Name with Suffix]"

    Args:
        layer (TRTLayer): A TensorRT layer of which we want to set the name.
        target (Target): A fx node.target. For call_function node, it's the function that
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
    if isinstance(val, list):
        val = tuple(val)
    return val


def extend_mod_attr_to_tuple(mod: torch.nn.Module, name: str, size: int):
    """
    Extend an attribute of `mod` that named `name` to a tuple of `size`.
    """
    val = getattr(mod, name)
    return extend_attr_to_tuple(val, size)


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


def get_axes_for_reduce_op(
    dim: Union[int, Sequence[int]],
    has_implicit_batch_dimension: bool,
) -> int:
    """
    TensorRT reduce layer relies on the binary representation of axes to
    determine which dims to reduce. For example, if we want to reduce on
    dim 1 and 2 then axes should be 6(110).

    Args:
        dim (Union[int, Sequence[int]]): An integer or a sequence of integers
            that will be used to generate axes for TensorRT.
        has_implicit_batch_dimension (bool): Whether the TensorRT network is
            using implicit batch dimension.

    Returns:
        An integer which binary form can be used as axes for TensorRT reduce
        layer.
    """
    if isinstance(dim, int):
        dim = (dim,)

    if has_implicit_batch_dimension:
        assert 0 not in dim, "Can't reduce over batch dimension when it's implicit."

    axes = 0
    for d in dim:
        axes |= 1 << (d - (1 if has_implicit_batch_dimension else 0))

    return axes


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


def squeeze_left(const: torch.Tensor):
    """
    Squeeze the size-1 dimensions on the left side of the shape tuple.
    PyTorch's `squeeze()` doesn't support passing multiple `dim`s at once, so
    we do it iteratively.
    """
    while len(const.shape) > 0 and const.shape[0] == 1:
        const = const.squeeze(dim=0)
    return const


def add_reduce_layer(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    operation_type: trt.ActivationType,
    name: str,
) -> TRTTensor:
    """
    Add a TensorRT Reduce layer to `network`.

    Args:
        network (TRTNetwork): TensorRT network object.
        target (Target): Target of fx node.
        args (Tuple[Argument, ...]): Args of the fx node.
        kwargs (Dict[str, Argument]): Kwargs of the fx node.
        operation_type (trt.ElementWiseOperation): Type of the TensorRT activation
            operation.
        name (str): The name we want to assign to the created TensorRT layer.

    Returns:
        The output of TensorRT Reduce layer.
    """
    input_val = kwargs["input"]
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"{name} received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    # If dim is specified, then the op is reducing over certain dimensions.
    # Otherwise, it's reducing over all elements, which is only supported in
    # explicit batch dimension.
    if "dim" not in kwargs:
        assert (
            not network.has_implicit_batch_dimension
        ), f"We don't support reduce({name}) over all the elements if batch dim is implicit."
        dim = range(0, len(input_val.shape))
    else:
        dim = kwargs["dim"]  # type: ignore[assignment]

    if not isinstance(dim, Sequence):
        dim = (dim,)

    if not network.has_implicit_batch_dimension:
        dim = tuple(len(input_val.shape) + i if i < 0 else i for i in dim)
    else:
        dim = tuple(len(input_val.shape) + i + 1 if i < 0 else i for i in dim)

    keepdim = False if "keepdim" not in kwargs else kwargs["keepdim"]
    layer = network.add_reduce(
        input_val,
        operation_type,
        get_axes_for_reduce_op(dim, network.has_implicit_batch_dimension),
        keepdim,
    )
    set_layer_name(layer, target, name)
    return layer.get_output(0)


def get_dyn_range(scale, zero_point, dtype):
    """
    Get the dynamic range of a tensor based on its scale, zero_point and dtype.
    """
    if dtype == torch.quint8:
        min_val, max_val = 0, 255
    elif dtype == torch.qint8:
        min_val, max_val = -128, 127
    else:
        raise RuntimeError(f"Unsupported quantized dtype {dtype}")

    return (min_val - zero_point) * scale, (max_val - zero_point) * scale


def mark_as_int8_layer(layer, dynamic_range):
    """
    Set the precision of a layer to int8 as well as the type of its first output.
    Also set the dynamic range of its first output.
    """
    if layer.type not in {
        trt.LayerType.SHUFFLE,
        trt.LayerType.CONCATENATION,
        trt.LayerType.CONSTANT,
        trt.LayerType.SHAPE,
    }:
        layer.precision = trt.int8

    for i in range(layer.num_outputs):
        output_val = layer.get_output(i)
        output_val.dynamic_range = dynamic_range
        layer.set_output_type(i, trt.int8)
        # output_val.dtype = trt.int8


def get_inputs_from_args_and_kwargs(args, kwargs, input_names):
    inputs = []
    for i, key in enumerate(input_names):
        if key not in kwargs:
            inputs.append(args[i])
        else:
            inputs.append(kwargs[key])
    return inputs


def dtype_uniform(
    network: TRTNetwork, target: Target, name: str, input: TRTTensor, other: TRTTensor
):
    table = {trt.bool: 0, trt.int32: 1, trt.float16: 2, trt.float32: 3}
    input_dtype = input.dtype
    other_dtype = other.dtype
    if table[input_dtype] > table[other_dtype]:
        layer = network.add_identity(other)
        layer.set_output_type(0, input_dtype)
        set_layer_name(layer, target, f"{name}_other_dtype_change")
        other = layer.get_output(0)
    elif table[input_dtype] < table[other_dtype]:
        layer = network.add_identity(input)
        layer.set_output_type(0, other_dtype)
        set_layer_name(layer, target, f"{name}_input_dtype_change")
        input = layer.get_output(0)
    elif table[input_dtype] == 0 and table[other_dtype] == 0:
        layer_i = network.add_identity(input)
        layer_i.set_output_type(0, trt.int32)
        set_layer_name(layer_i, target, f"{name}_input_dtype_change")
        input = layer_i.get_output(0)

        layer_o = network.add_identity(other)
        layer_o.set_output_type(0, trt.int32)
        set_layer_name(layer_o, target, f"{name}_other_dtype_change")
        other = layer_o.get_output(0)
    return input, other


def type_cast(
    network: TRTNetwork,
    target: Target,
    name: str,
    input: TRTTensor,
    cast_type: TRTDataType,
):
    """
    This function helps to cast the input type to cast_type
    """
    layer_i = network.add_identity(input)
    layer_i.set_output_type(0, cast_type)
    set_layer_name(layer_i, target, f"{name}_dtype_change")
    return layer_i.get_output(0)


def trt_dtype_to_torch_dtype(trt_dtype):
    table = {
        trt.bool: torch.bool,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32,
    }
    return table[trt_dtype]
