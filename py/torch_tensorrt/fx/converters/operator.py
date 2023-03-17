import numpy as np
import operator
import warnings
import logging
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple, Union

import tensorrt as trt
import torch
from torch.fx.node import Argument, Target
from ..utils import get_dynamic_dims, torch_dtype_from_trt, torch_dtype_to_trt

from ..tracer.acc_tracer import acc_ops

from .converter_utils import get_trt_tensor
from .converter_utils import set_layer_name
from .converter_utils import get_trt_tensor
from .converter_utils import broadcast
from .converter_utils import squeeze_left
from .converter_utils import dtype_uniform
from .converter_utils import get_trt_plugin
from .converter_utils import get_positive_dim
from .converter_utils import prepend_ones
from .converter_utils import has_dynamic_shape
from .converter_utils import get_shape_with_dynamic_shape

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

_LOGGER: logging.Logger = logging.getLogger(__name__)

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

def trunc_div(
    input: TRTTensor, other: TRTTensor, network: TRTNetwork, target: Target, name: str
) -> TRTTensor:
    """
    Perform trunc divide on Tensor, result of divide will be round toward zero.
    This means for positive number, it will be floor round; for negative number,
    it will be ceil round. Example: [2.1, 0.8, -3.2] -> [2, 0, -3].

    Args:
        input: divisor.
        other: dividend.
        network: INetworkDefinition.
        target: node target.
        name: namespace for the op

    Returns:
        A TensorRT tensor represent the result of trunc divide.
    """
    prod_output = add_binary_elementwise_layer(
        network, input, other, trt.ElementWiseOperation.PROD, target, f"{name}_prod"
    )
    sign_output = sign(network, prod_output, target, name)

    # Convert constant input into ITensor for UnaryOperation
    if not isinstance(input, trt.tensorrt.ITensor):
        input = get_trt_tensor(network, input, f"{name}_input")
    if not isinstance(other, trt.tensorrt.ITensor):
        other = get_trt_tensor(
            network, other, f"{name}_other", dtype=torch_dtype_from_trt(input.dtype)
        )

    abs_input_output = add_unary_layer(
        network, input, trt.UnaryOperation.ABS, target, f"{name}_abs_input"
    )
    abs_other_output = add_unary_layer(
        network, other, trt.UnaryOperation.ABS, target, f"{name}_abs_other"
    )
    abs_floor_output = add_binary_elementwise_layer(
        network,
        abs_input_output,
        abs_other_output,
        trt.ElementWiseOperation.FLOOR_DIV,
        target,
        f"{name}_floor_div",
    )
    output = add_binary_elementwise_layer(
        network,
        abs_floor_output,
        sign_output,
        trt.ElementWiseOperation.PROD,
        target,
        f"{name}_output",
    )

    return output

def add_tile(network, target, kwargs, name):
    input_t = kwargs["input"]
    input_val = get_trt_tensor(network, input_t, f"{name}_input")

    dims = tuple(cast(Sequence[int], kwargs["dims"]))
    n_input_dims = len(input_val.shape) + (
        1 if network.has_implicit_batch_dimension else 0
    )

    if len(dims) > n_input_dims:
        assert not network.has_implicit_batch_dimension
        layer = network.add_shuffle(input_val)
        layer.name = f"{name}_reshape"
        num_preceding_ones = len(dims) - n_input_dims

        if len(get_dynamic_dims(input_val.shape)) > 1:
            input_shape_layer = network.add_shape(input_val)
            input_shape_layer.name = f"{name}_input_shape"
            preceding_ones = network.add_constant(
                (num_preceding_ones,),
                np.ascontiguousarray([1] * num_preceding_ones, np.int32),
            ).get_output(0)
            reshape_layer = network.add_concatenation(
                [preceding_ones, input_shape_layer.get_output(0)]
            )
            reshape_layer.axis = 0
            reshape_layer.name = f"{name}_reshape_dims"
            layer.set_input(1, reshape_layer.get_output(0))
        else:
            layer.reshape_dims = (1,) * (len(dims) - n_input_dims) + tuple(
                input_val.shape
            )
        input_val = layer.get_output(0)
    else:
        dims = (1,) * (n_input_dims - len(dims)) + dims

    if network.has_implicit_batch_dimension:
        assert dims[0] == 1, "Can't tile the batch dim when it's implicit."
        dims = dims[1:]
    starts = [0] * len(dims)
    shapes = []
    if all(isinstance(d, int) for d in dims):
        shapes = [i * j for i, j in zip(input_val.shape, dims)]  # type: ignore[union-attr]
    else:
        shape = []
        for i, (s, d) in enumerate(zip(input_val.shape, dims)):
            if isinstance(d, TRTTensor) and len(d.shape) == 0:
                d = prepend_ones(network, d, f"{name}_{i}", 1)
            else:
                d = get_trt_tensor(network, d, f"{name}_{i}")
            shape.append(d)
            mul = add_binary_elementwise_layer(
                network,
                s,
                d,
                trt.ElementWiseOperation.PROD,
                target,
                f"{name}_mul_{i}",
            )
            shapes.append(mul)
        dims = shape
    # If there's dynmaic dim then there would be negative dims in shapes which is not allowed.
    # Here we build a dummy shapes array.
    if has_dynamic_shape(input_val.shape):  # type: ignore[union-attr]
        shapes = [1] * len(dims)
    strides = [1] * len(dims)
    layer = network.add_slice(input_val, starts, shapes, strides)
    layer.mode = trt.SliceMode.WRAP
    set_layer_name(layer, target, name)

    if has_dynamic_shape(input_val.shape):  # type: ignore[union-attr]
        starts_tensor = network.add_constant(
            (len(dims),), np.ascontiguousarray([0] * len(dims), np.int32)
        ).get_output(0)
        if all(isinstance(d, int) for d in dims):
            dims_tensor = network.add_constant(
                (len(dims),), np.ascontiguousarray(dims, np.int32)
            ).get_output(0)
        else:
            assert all(isinstance(d, TRTTensor) for d in dims)
            concat_dims_layer = network.add_concatenation(inputs=dims)
            concat_dims_layer.axis = 0
            concat_dims_layer.name = f"{name}_tile_dim"
            dims_tensor = concat_dims_layer.get_output(0)
        input_shape_layer = network.add_shape(input_val)
        input_shape_layer.name = f"{name}_slice_input_shape"
        slice_shapes_tensor = add_binary_elementwise_layer(
            network,
            input_shape_layer.get_output(0),
            dims_tensor,
            trt.ElementWiseOperation.PROD,
            target,
            f"{name}_slice_shapes",
        )
        layer.set_input(1, starts_tensor)
        layer.set_input(2, slice_shapes_tensor)

    return layer.get_output(0)

def add_linear(network, target, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"Linear received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    dynamic_dims = get_dynamic_dims(input_val.shape)
    assert len(dynamic_dims) < 2 and input_val.shape[-1] != -1, (
        "Currently we only support one dynmaic "
        "dim for linear and it can't be the last dim."
    )

    if isinstance(kwargs["weight"], torch.Tensor):
        weight = get_trt_tensor(network, kwargs["weight"].t(), f"{name}_weight")
        if target not in (acc_ops.linear, torch.ops.aten.linear):
            weight_op = trt.MatrixOperation.TRANSPOSE
        else:
            weight_op = trt.MatrixOperation.NONE
    else:
        assert isinstance(
            kwargs["weight"], TRTTensor
        ), f"Expect weight to be trt tensor but got {type(kwargs['weight'])}"
        weight = kwargs["weight"]
        weight_op = trt.MatrixOperation.TRANSPOSE

    preset_diff = 0
    if len(input_val.shape) == 1:
        preset_diff -= 1
        input_op = trt.MatrixOperation.VECTOR
    else:
        input_op = trt.MatrixOperation.NONE

    input_val, weight = broadcast(
        network, input_val, weight, f"{name}_input", f"{name}_weight", preset_diff
    )
    matmul_layer = network.add_matrix_multiply(input_val, input_op, weight, weight_op)
    set_layer_name(matmul_layer, target, f"{name}_matmul")
    res = matmul_layer.get_output(0)

    if kwargs["bias"] is not None:
        bias = get_trt_tensor(network, kwargs["bias"], f"{name}_bias")  # type: ignore[arg-type]
        res = add_binary_elementwise_layer(
            network,
            matmul_layer.get_output(0),
            bias,
            trt.ElementWiseOperation.SUM,
            target,
            f"{name}_add",
        )
    return res

def add_unary_layer(
    network: TRTNetwork,
    input_val: TRTTensor,
    operation_type: trt.UnaryOperation,
    target: Target,
    name: str,
) -> TRTTensor:
    """
    Add a TensorRT Unary layer to `network`.

    Args:
        network (TRTNetwork): TensorRT network object.
        input_val (TRTTensor): Input to the unary op. Must be a TensorRT tensor.
        op_type (trt.ElementWiseOperation): Type of the TensorRT unary operation.
        target (Target): Target of fx node.
        name (str): The name we want to assign to the created TensorRT layer.

    Returns:
        The output of TensorRT Unary layer.
    """
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"{operation_type} received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    layer = network.add_unary(input_val, operation_type)
    set_layer_name(layer, target, name)
    output = layer.get_output(0)
    output.name = output.name + "_" + target.__name__
    return layer.get_output(0)

def layer_norm(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"LayerNorm received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    shape = kwargs["weight"].shape  # type: ignore[union-attr]
    broadcasted_shape = (1,) * (len(input_val.shape) - len(shape)) + shape
    gamma = to_numpy(kwargs["weight"].reshape(*shape))  # type: ignore[union-attr]
    beta = to_numpy(kwargs["bias"].reshape(*shape))  # type: ignore[union-attr]
    eps = kwargs["eps"]

    axes = 0
    for d in range(len(shape)):
        axes |= 1 << (len(input_val.shape) - d - 1)

    # E[x]
    mean_expected_layer = network.add_reduce(
        input_val, trt.ReduceOperation.AVG, axes, keep_dims=True
    )
    set_layer_name(mean_expected_layer, target, f"{name}_mean_expected")

    # X-E[x]
    sub_trt = operator.add_binary_elementwise_layer(
        network,
        input_val,
        mean_expected_layer.get_output(0),
        trt.ElementWiseOperation.SUB,
        target,
        f"{name}_sub",
    )
    # Variance = mean(pow(x_sub_mean,2))
    pow_tensor = network.add_constant(
        (1,) * len(input_val.shape),
        trt.Weights(np.ascontiguousarray([2.0], dtype=np.float32)),
    )
    pow_tensor.name = f"{name}_power"
    pow_var = operator.add_binary_elementwise_layer(
        network,
        sub_trt,
        pow_tensor.get_output(0),
        trt.ElementWiseOperation.POW,
        target,
        f"{name}_pow_var",
    )
    mean_trt_layer = network.add_reduce(
        pow_var, trt.ReduceOperation.AVG, axes, keep_dims=True
    )
    set_layer_name(mean_trt_layer, target, f"{name}_mean")
    # Variance + eps
    eps_tensor = network.add_constant(
        (1,) * len(input_val.shape),
        trt.Weights(np.ascontiguousarray([eps], dtype=np.float32)),
    )
    eps_tensor.name = f"{name}_eps"
    add_trt = add_binary_elementwise_layer(
        network,
        mean_trt_layer.get_output(0),
        eps_tensor.get_output(0),
        trt.ElementWiseOperation.SUM,
        target,
        f"{name}_add",
    )
    # SQRT((Var + eps))
    sqrt_trt = add_unary_layer(
        network, add_trt, trt.UnaryOperation.SQRT, target, f"{name}_sqrt"
    )
    # (x - E[x]) / sqrt((var + eps))
    div_trt = add_binary_elementwise_layer(
        network,
        sub_trt,
        sqrt_trt,
        trt.ElementWiseOperation.DIV,
        target,
        f"{name}_div_trt",
    )

    assert gamma is not None
    gamma_tensor = network.add_constant(gamma.shape, trt.Weights(np.ascontiguousarray(gamma)))  # type: ignore[attr-defined]
    gamma_tensor.name = f"{name}_gamma"
    assert beta is not None
    beta_tensor = network.add_constant(gamma.shape, trt.Weights(np.ascontiguousarray(beta)))  # type: ignore[attr-defined]
    beta_tensor.name = f"{name}_beta"
    # y * gamma + beta
    scale_layer = add_binary_elementwise_layer(
        network,
        div_trt,
        gamma_tensor.get_output(0),
        trt.ElementWiseOperation.PROD,
        target,
        f"{name}_scale",
    )
    return add_binary_elementwise_layer(
        network,
        scale_layer,
        beta_tensor.get_output(0),
        trt.ElementWiseOperation.SUM,
        target,
        name,
    )

def add_add(network, target, kwargs, name):
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        kwargs["other"],
        trt.ElementWiseOperation.SUM,
        target,
        name,
    )

def add_matmul(network, target, kwargs, name):
    input_val = get_trt_tensor(network, kwargs["input"], f"{name}_input")
    other_val = get_trt_tensor(network, kwargs["other"], f"{name}_other")

    for i in [input_val, other_val]:
        if not isinstance(i, TRTTensor):
            raise RuntimeError(
                f"matmul received input {i} that is not part of the TensorRT region!"
            )

    input_matrix_op = other_matrix_op = trt.MatrixOperation.NONE
    preset_diff = 0

    if len(input_val.shape) == 1:
        preset_diff -= 1
        input_matrix_op = trt.MatrixOperation.VECTOR

    if len(other_val.shape) == 1:
        preset_diff += 1
        other_matrix_op = trt.MatrixOperation.VECTOR

    input_val, other_val = broadcast(
        network, input_val, other_val, f"{name}_input", f"{name}_other", preset_diff
    )
    layer = network.add_matrix_multiply(
        input_val, input_matrix_op, other_val, other_matrix_op
    )
    set_layer_name(layer, target, name)
    return layer.get_output(0)

def add_layer_norm(network, target, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"LayerNorm received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    gamma = kwargs["weight"].detach().cpu().float().numpy()
    gamma_field = trt.PluginField("gamma", gamma, trt.PluginFieldType.FLOAT32)
    beta = kwargs["bias"].detach().cpu().float().numpy()
    beta_field = trt.PluginField("beta", beta, trt.PluginFieldType.FLOAT32)
    eps_field = trt.PluginField(
        "eps", np.array([kwargs["eps"]], dtype=np.float32), trt.PluginFieldType.FLOAT32
    )
    try:
        normalized_shape = np.array(kwargs["normalized_shape"], dtype=np.int32)
    except TypeError:
        _LOGGER.error("Unable to convert normalized_shape to a field, fall back to []")
        normalized_shape = np.array([], dtype=np.int32)

    normalized_shape_filed = trt.PluginField(
        "normalized_shape", normalized_shape, trt.PluginFieldType.INT32
    )
    field_collection = trt.PluginFieldCollection(
        [gamma_field, beta_field, eps_field, normalized_shape_filed]
    )

    try:
        if network.has_implicit_batch_dimension:
            plugin = get_trt_plugin("layer_norm", field_collection, "1", "fx2trt")
        else:
            plugin = get_trt_plugin("LayerNormDynamic", field_collection, "1", "fx2trt")
    except AssertionError:
        _LOGGER.error(
            "Unable to find layer norm plugin, fall back to TensorRT implementation."
        )
        return layer_norm(network, target, args, kwargs, name)
    layer = network.add_plugin_v2([input_val], plugin)
    layer.name = name
    return layer.get_output(0)

def add_cumsum(network, target, kwargs, name):
    input_val = kwargs["input"]
    dim = cast(int, kwargs["dim"])
    input_shape = input_val.shape  # type: ignore[union-attr]
    input_dim_size = len(input_val.shape)  # type: ignore[union-attr]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"cumsum received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "cumsum converter currently doesn't support implicit batch dimension"
        )
    dim = get_positive_dim(dim, input_dim_size)
    loop = network.add_loop()
    trip_limit = None
    if input_shape[dim] > 0:
        axis = torch.tensor(input_shape[dim], dtype=torch.int32)
        trip_limit_layer = network.add_constant(axis.shape, to_numpy(axis))
    else:
        input_shape = network.add_shape(input_val).get_output(0)
        dim_value = torch.tensor(dim, dtype=torch.int32)
        axis = network.add_constant(dim_value.shape, to_numpy(dim_value)).get_output(0)
        trip_limit_layer = network.add_gather(input_shape, axis, 0)
    set_layer_name(trip_limit_layer, target, f"{name}_trip_limit")
    trip_limit = trip_limit_layer.get_output(0)

    loop.add_trip_limit(trip_limit, trt.TripLimit(0))
    iterator = loop.add_iterator(input_val, dim, False)
    data = iterator.get_output(0)
    new_dims = tuple(data.shape)
    zero_tensor = torch.zeros(new_dims, dtype=trt_dtype_to_torch_dtype(input_val.dtype))
    zero_tensor = network.add_constant(
        zero_tensor.shape, to_numpy(zero_tensor)
    ).get_output(0)

    running_sum = loop.add_recurrence(zero_tensor)
    set_layer_name(running_sum, target, f"{name}_running_sum_1")
    running_sum_tensor = running_sum.get_output(0)

    current_sum = add_binary_elementwise_layer(
        network,
        data,
        running_sum_tensor,
        trt.ElementWiseOperation.SUM,
        target,
        f"{name}_sum_1",
    )
    running_sum.set_input(1, current_sum)

    running_sum = loop.add_recurrence(zero_tensor)
    set_layer_name(running_sum, target, f"{name}_running_sum_2")
    running_sum_tensor = running_sum.get_output(0)

    current_sum = add_binary_elementwise_layer(
        network,
        data,
        running_sum_tensor,
        trt.ElementWiseOperation.SUM,
        target,
        f"{name}_sum_2",
    )
    running_sum.set_input(1, current_sum)

    loop_output = loop.add_loop_output(current_sum, trt.LoopOutput.CONCATENATE, dim)
    set_layer_name(loop_output, target, f"{name}_loop_output")
    loop_output.set_input(1, trip_limit)
    return loop_output.get_output(0)

def add_maximum(network, target, kwargs, name):
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        kwargs["other"],
        trt.ElementWiseOperation.MAX,
        target,
        name,
    )

def add_mul(network, target, kwargs, name):
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        kwargs["other"],
        trt.ElementWiseOperation.PROD,
        target,
        name,
    )

def add_pow(network, target, kwargs, name):
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        kwargs["other"],
        trt.ElementWiseOperation.POW,
        target,
        name,
    )

def add_floor_div(network, target, kwargs, name):
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        kwargs["other"],
        trt.ElementWiseOperation.FLOOR_DIV,
        target,
        name,
    )

def add_div(network, target, kwargs, name):
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        kwargs["other"],
        trt.ElementWiseOperation.DIV,
        target,
        name,
    )

def add_sub(network, target, kwargs, name):
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        kwargs["other"],
        trt.ElementWiseOperation.SUB,
        target,
        name,
    )
def add_minimum(network, target, kwargs, name):
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        kwargs["other"],
        trt.ElementWiseOperation.MIN,
        target,
        name,
    )

def add_logical_and(network, target, kwargs, name):
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "The `ne` function should be called with explicit batch dimension."
        )

    input_t = kwargs["input"]
    other_t = kwargs["other"]

    input_t = get_trt_tensor(network, input_t, f"{name}_input_t")
    other_t = get_trt_tensor(network, other_t, f"{name}_other_t")

    input_t, other_t = dtype_uniform(network, target, name, input_t, other_t)
    eq_t = add_binary_elementwise_layer(
        network, input_t, other_t, trt.ElementWiseOperation.EQUAL, target, name
    )

    return add_unary_layer(network, eq_t, trt.UnaryOperation.NOT, target, name)

def add_ne(network, target, kwargs, name):
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "The `ne` function should be called with explicit batch dimension."
        )

    input_t = kwargs["input"]
    other_t = kwargs["other"]

    input_t = get_trt_tensor(network, input_t, f"{name}_input_t")
    other_t = get_trt_tensor(network, other_t, f"{name}_other_t")

    input_t, other_t = dtype_uniform(network, target, name, input_t, other_t)
    eq_t = add_binary_elementwise_layer(
        network, input_t, other_t, trt.ElementWiseOperation.EQUAL, target, name
    )

    return add_unary_layer(network, eq_t, trt.UnaryOperation.NOT, target, name)

def add_eq(network, target, kwargs, name):
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "The `eq` function should be called with explicit batch dimension."
        )

    input_t = kwargs["input"]
    other_t = kwargs["other"]

    input_t = get_trt_tensor(network, input_t, f"{name}_input_t")
    other_t = get_trt_tensor(network, other_t, f"{name}_other_t")

    input_t, other_t = dtype_uniform(network, target, name, input_t, other_t)
    return add_binary_elementwise_layer(
        network, input_t, other_t, trt.ElementWiseOperation.EQUAL, target, name
    )

def add_gt(network, target, kwargs, name):
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "The `gt` function should be called with explicit batch dimension."
        )

    input_t = kwargs["input"]
    other_t = kwargs["other"]

    input_t = get_trt_tensor(network, input_t, f"{name}_input_t")
    other_t = get_trt_tensor(network, other_t, f"{name}_other_t")

    input_t, other_t = dtype_uniform(network, target, name, input_t, other_t)
    return add_binary_elementwise_layer(
        network, input_t, other_t, trt.ElementWiseOperation.GREATER, target, name
    )

def add_lt(network, target, kwargs, name):
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "The `le` function should be called with explicit batch dimension."
        )

    input_t = kwargs["input"]
    other_t = kwargs["other"]

    input_t = get_trt_tensor(network, input_t, f"{name}_input_t")
    other_t = get_trt_tensor(network, other_t, f"{name}_other_t")

    input_t, other_t = dtype_uniform(network, target, name, input_t, other_t)
    return add_binary_elementwise_layer(
        network, input_t, other_t, trt.ElementWiseOperation.LESS, target, name
    )

def add_logical_or(network, target, kwargs, name):
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "The `logical_or` function should be called with explicit batch dimension."
        )

    input_t = kwargs["input"]
    other_t = kwargs["other"]
    if isinstance(other_t, (torch.Tensor, bool)):
        if isinstance(other_t, bool):
            other_t = int(other_t)
        elif other_t.dtype == torch.bool:
            other_t = other_t.to(torch.int32)
    other_t = get_trt_tensor(network, other_t, f"{name}_other_t")
    if input_t.dtype != trt.bool:
        layer_i = network.add_identity(input_t)
        layer_i.set_output_type(0, trt.bool)
        set_layer_name(layer_i, target, f"{name}_input_dtype_change")
        input_t = layer_i.get_output(0)
    if other_t.dtype != trt.bool:
        layer_o = network.add_identity(other_t)
        layer_o.set_output_type(0, trt.bool)
        set_layer_name(layer_o, target, f"{name}_other_dtype_change")
        other_t = layer_o.get_output(0)

    return add_binary_elementwise_layer(
        network, input_t, other_t, trt.ElementWiseOperation.OR, target, name
    )

def add_logical_xor(network, target, kwargs, name):
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "The `logical_xor` function should be called with explicit batch dimension."
        )

    input_t = kwargs["input"]
    other_t = kwargs["other"]
    if isinstance(other_t, (torch.Tensor, bool)):
        if isinstance(other_t, bool):
            other_t = int(other_t)
        elif other_t.dtype == torch.bool:
            other_t = other_t.to(torch.int32)
    other_t = get_trt_tensor(network, other_t, f"{name}_other_t")
    if input_t.dtype != trt.bool:
        layer_i = network.add_identity(input_t)
        layer_i.set_output_type(0, trt.bool)
        set_layer_name(layer_i, target, f"{name}_input_dtype_change")
        input_t = layer_i.get_output(0)
    if other_t.dtype != trt.bool:
        layer_o = network.add_identity(other_t)
        layer_o.set_output_type(0, trt.bool)
        set_layer_name(layer_o, target, f"{name}_other_dtype_change")
        other_t = layer_o.get_output(0)

    return add_binary_elementwise_layer(
        network, input_t, other_t, trt.ElementWiseOperation.XOR, target, name
    )

def add_fmod(network, target, kwargs, name):
     # NOTE: TRT doesnt currently implement fmod so we need multiple operations to perform it
    trunc_div_value = trunc_div(
        kwargs["input"], kwargs["other"], network, target, name + "_trunc_div"
    )
    prod_value = add_binary_elementwise_layer(
        network,
        trunc_div_value,
        kwargs["other"],
        trt.ElementWiseOperation.PROD,
        target,
        name + "_prod",
    )
    sub_value = add_binary_elementwise_layer(
        network,
        kwargs["input"],
        prod_value,
        trt.ElementWiseOperation.SUB,
        target,
        name + "_sub",
    )
    return sub_value

def add_trunc_div(network, target, kwargs, name):
    return trunc_div(kwargs["input"], kwargs["other"], network, target, name)

def add_expand(network, target, kwargs, name):
    input_t = kwargs["input"]
    shape = list(kwargs["sizes"])

    input_val = get_trt_tensor(network, input_t, f"{name}_input")

    if network.has_implicit_batch_dimension:
        shape = shape[1:]

    ranks = len(input_val.shape)
    # TRT does not support different dimension size
    assert len(shape) == ranks
    shape = [input_val.shape[i] if shape[i] == -1 else shape[i] for i in range(ranks)]

    inshape = tuple(input_val.shape)
    shape = tuple(shape)
    start = tuple([0] * ranks)
    stride = tuple(
        [int(i == o) for i, o in zip(inshape, shape)]
    )  # stride == 1 if dimensions match, 0 otherwise
    layer = network.add_slice(input_val, start=start, shape=shape, stride=stride)
    set_layer_name(layer, target, name)
    return layer.get_output(0)

def add_slice(network, target, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"slice_tensor received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    ranks = len(input_val.shape) + (1 if network.has_implicit_batch_dimension else 0)
    dim = get_positive_dim(cast(int, kwargs["dim"]), ranks)
    dynamic_shape = has_dynamic_shape(input_val.shape)
    if network.has_implicit_batch_dimension:
        if dim == 0:
            raise RuntimeError(
                f"We do not support slice_tensor at batch dim when it's implicit, got {dim}!"
            )
        dim = dim - 1
    else:
        if dynamic_shape:
            # Check whether slice target dim is dynamic shape dim
            assert input_val.shape[dim] != -1, "Can't chunk on dynamic shape dimension!"

    start_int = cast(int, kwargs["start"])
    stop_int = cast(int, kwargs["stop"])
    step_int = cast(int, kwargs["step"])
    start = [0] * len(input_val.shape)
    start[dim] = start_int
    stride = [1] * len(start)
    stride[dim] = step_int
    output_shape = list(input_val.shape)
    output_shape[dim] = (stop_int - start_int) // step_int

    if dynamic_shape > 0:
        output_shape = get_shape_with_dynamic_shape(
            network, output_shape, input_val, target, name
        )
    layer = network.add_slice(
        input_val,
        start=start,
        shape=[] if dynamic_shape else output_shape,
        stride=stride,
    )
    if dynamic_shape:
        layer.set_input(2, output_shape)
    set_layer_name(layer, target, name)
    return layer.get_output(0)

def add_select(network, target, kwargs, name):
    input_val = kwargs["input"]
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"slice_tensor received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    
    ranks = len(input_val.shape) + (1 if network.has_implicit_batch_dimension else 0)
    dim = get_positive_dim(cast(int, kwargs["dim"]), ranks)
    dynamic_shape = has_dynamic_shape(input_val.shape)
    if network.has_implicit_batch_dimension:
        if dim == 0:
            raise RuntimeError(
                f"We do not support slice_tensor at batch dim when it's implicit, got {dim}!"
            )
        dim = dim - 1
    else:
        if dynamic_shape:
            # Check whether slice target dim is dynamic shape dim
            assert input_val.shape[dim] != -1, "Can't select on negative shape dimension!"
    index = kwargs[2]
    if index >= input_val.shape[dim]:
        raise RuntimeError(
            f"cannot have index greater than the dimension length! {input_val.shape[dim]}"
        )
    output_shape = list(input_val.shape)
    output_shape[dim] = 1
    if dynamic_shape > 0:
        output_shape = get_shape_with_dynamic_shape(
            network, output_shape, input_val, target, name
        )
    layer = network.add_gather(
        input_val,
        dim,
        index
    )
    out = layer.getOutput(0)
    if(len(out.shape) != 1):
        layer = network.add_shuffle(out)
    return layer.getOutput(0)

        
    
