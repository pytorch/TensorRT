import numpy as np
import operator
import warnings
import logging
import math
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple, Union

import tensorrt as trt
import torch
from torch.fx.node import Argument, Target

_LOGGER: logging.Logger = logging.getLogger(__name__)


from .converter_utils import add_binary_elementwise_layer
from .converter_utils import broadcast
from .converter_utils import broadcastable
from .converter_utils import get_python_op_from_trt_elementwise_op
from .converter_utils import get_positive_dim
from .converter_utils import get_shape_with_dynamic_shape
from .converter_utils import get_trt_tensor
from .converter_utils import has_dynamic_shape
from .converter_utils import set_layer_name
from .converter_utils import squeeze_left
from .converter_utils import to_numpy

from ..utils import torch_dtype_from_trt, get_dynamic_dims

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


def add_expand(network, target, kwargs, name):
    input_t = kwargs["input"]
    shape = list(kwargs["sizes"])

    input_val = get_trt_tensor(network, input_t, f"{name}_input")

    if network.has_implicit_batch_dimension:
        shape = shape[1:]

    ranks = len(input_val.shape)
    # TRT does not support different dimension size
    # though this condition is not seen in the case of bmm
    # where input_t and shape dimensions are not equal
    assert len(shape) >= ranks
    if len(shape) != ranks:
        shape_tuple = tuple([0] * len(shape))
        shape_tensor = get_trt_tensor(network, input_t, f"{name}_shape")
        input_val, shape_tensor = broadcast(
            network, input_val, shape_tensor, f"{name}_input_val", f"{name}_shape_val"
        )
        ranks = len(shape)

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


def add_mul(network, target, kwargs, name):
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        kwargs["other"],
        trt.ElementWiseOperation.PROD,
        target,
        name,
    )


def add_rsub(network, target, kwargs, name):
    kwargs_new = {}
    if "alpha" in kwargs:
        kwargs_new["input"] = kwargs["other"]
        kwargs_new["other"] = kwargs["alpha"]
        scaled_tensor = add_mul(network, target, kwargs_new, name + "_mul")
    else:
        scaled_tensor = kwargs["other"]
    input = kwargs["input"]
    return add_binary_elementwise_layer(
        network,
        kwargs["input"],
        scaled_tensor,
        trt.ElementWiseOperation.SUB,
        target,
        name + "_sub",
    )


def add_rsqrt(network, target, kwargs, name):
    sqrt_trt = add_sqrt(network, target, kwargs, name)
    return add_binary_elementwise_layer(
        network,
        1,
        sqrt_trt,
        trt.ElementWiseOperation.DIV,
        target,
        f"{name}_div_trt",
    )


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
            assert (
                input_val.shape[dim] != -1
            ), "Can't select on negative shape dimension!"
    index = kwargs["index"]

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
    index_value = torch.tensor(index, dtype=torch.int32)
    indices_tensor = network.add_constant(
        index_value.shape, to_numpy(index_value)
    ).get_output(0)
    layer = network.add_gather(input_val, indices_tensor, dim)
    out = layer.get_output(0)
    if len(out.shape) != 1:
        layer = network.add_shuffle(out)
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
    output_shape[dim] = math.ceil((stop_int - start_int) / step_int)

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


def add_softmax(network, target, kwargs, name):
    input_val = kwargs["input"]
    input_ranks = len(input_val.shape) + (1 if network.has_implicit_batch_dimension else 0)  # type: ignore[union-attr]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"softmax received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    # Used to get dim when dim is None. Copied from PyTorch softmax implementation.
    def get_softmax_dim(ndim: int) -> int:
        if ndim == 0 or ndim == 1 or ndim == 3:
            ret = 0
        else:
            ret = 1
        return ret

    if kwargs["dim"] is None:
        dim = get_softmax_dim(input_ranks)
    else:
        dim = cast(int, kwargs["dim"])

    dim = get_positive_dim(dim, input_ranks)
    if network.has_implicit_batch_dimension:
        assert dim != 0, "Can't apply softmax on batch dimension when it's implicit."
        dim -= 1

    layer = network.add_softmax(input_val)
    layer.axes = 1 << dim
    set_layer_name(layer, target, name)
    return layer.get_output(0)


def add_squeeze(network, target, kwargs, name):
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"squeeze received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    dims = []
    if "dim" in kwargs:
        if isinstance(kwargs["dim"], int):
            dims.append(cast(Optional[int], kwargs["dim"]))
        else:
            for dim in kwargs["dim"]:
                dims.append(cast(Optional[int], dim))

    # dim = cast(Optional[int], kwargs["dim"] if "dim" in kwargs else None)
    # Squeeze with dim=None would only work in explicit batch dim mode without any dynamic
    # dim, which is a very rare case. For now we just claim not supporting dim=None.
    assert not (len(dims) == 0), "We don't support dim=None right now for squeeze."

    for dim in dims:
        dim = get_positive_dim(
            dim,
            len(input_val.shape) + (1 if network.has_implicit_batch_dimension else 0),
        )
        if network.has_implicit_batch_dimension:
            assert dim != 0, "We don't support squeeze batch dim when it's implicit."
            dim -= 1

        assert input_val.shape[dim] != -1, "We don't support squeeze dynamic dim."
        assert (
            len(get_dynamic_dims(input_val.shape)) <= 1
        ), "Currently more than one dynamic dim for input to squeeze is not supported."

    output_shape = []
    for i, s in enumerate(input_val.shape):
        if (i in dims) and s == 1:
            continue
        output_shape.append(s)
    layer = network.add_shuffle(input_val)
    layer.reshape_dims = tuple(output_shape)
    set_layer_name(layer, target, name)
    return layer.get_output(0)


def add_sqrt(network, target, kwargs, name):
    input_val = kwargs["input"]
    operation_type = trt.UnaryOperation.SQRT
    return add_unary_layer(network, input_val, operation_type, target, name)


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
    sub_trt = add_binary_elementwise_layer(
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
    pow_var = add_binary_elementwise_layer(
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


def add_where(network, target, kwargs, name):
    condition_t = kwargs["condition"]
    x_t = kwargs["x"]
    y_t = kwargs["y"]

    x_t_dim = len(tuple(x_t.shape))
    y_t_dim = len(tuple(y_t.shape))
    condition_t_dim = len(tuple(condition_t.shape))

    if type(x_t) != TRTTensor:
        assert type(x_t) is torch.Tensor, f"value {x_t} is not torch.Tensor!"

    if type(y_t) != TRTTensor:
        assert type(y_t) is torch.Tensor, f"value {y_t} is not torch.Tensor!"

    if not (broadcastable(x_t, y_t)):
        assert f"The two torch tensors should be broadcastable"

    # get output shape
    # purpose of this is to bring x_t and y_t rank same as
    # output_shape to input it to the add_expand operation
    # condition_t will have dimension of either x_t or y_t
    x_t, y_t = broadcast(network, x_t, y_t, f"{name}_x", f"{name}_y")
    if len(tuple(condition_t.shape)) != len(tuple(x_t.shape)):
        condition_t, x_t = broadcast(
            network, condition_t, x_t, f"{name}_condition", f"{name}_x"
        )

    print("x_t shape", x_t.shape)
    print("y_t shape", y_t.shape)
    print("condition_t shape", condition_t.shape)
    x_shape = list(x_t.shape)
    y_shape = list(y_t.shape)
    condition_shape = list(condition_t.shape)
    output_shape = list(torch.broadcast_shapes(condition_shape, x_shape, y_shape))

    # expand shape
    if type(condition_t) != TRTTensor:
        assert condition_t.dtype == torch.bool, "condition dtype is not bool"
        if condition_shape != output_shape:
            condition_t.expand(output_shape)
        condition_t = condition_t.to(torch.int32)
        condition_const = get_trt_tensor(network, condition_t, f"{name}_condition")
        condition_layer = network.add_identity(condition_const)
        condition_layer.set_output_type(0, trt.bool)
        set_layer_name(condition_layer, target, f"{name}_condition")
        condition_val = condition_layer.get_output(0)
    else:
        assert condition_t.dtype == trt.bool, "mask dtype is not bool!"
        if condition_shape != condition_t_dim:
            condition_val = add_expand(
                network,
                target,
                {"input": condition_t, "sizes": output_shape},
                name=f"{name}_expand",
            )
        else:
            condition_val = condition_t

    if type(x_t) != TRTTensor:
        if x_shape != x_t_dim:
            # special case where 1 element in x_t
            if len(x_t.shape) == 0:
                x_t = x_t.unsqueeze(0)
            x_t = x_t.expand(output_shape)
        x_val = get_trt_tensor(network, x_t, f"{name}_x")
    else:
        x_val = x_t
        if x_shape != output_shape:
            x_val = add_expand(
                network,
                target,
                {"input": x_val, "sizes": output_shape},
                name=f"{name}_x_expand",
            )

    if type(y_t) != TRTTensor:
        if y_shape != output_shape:
            # special case where 1 element in y_t
            if len(y_t.shape) == 0:
                y_t = y_t.unsqueeze(0)
            y_t = y_t.expand(output_shape)
        y_val = get_trt_tensor(network, y_t, f"{name}_y")
    else:
        y_val = y_t
        if y_shape != y_t_dim:
            y_val = add_expand(
                network,
                target,
                {"input": y_val, "sizes": output_shape},
                name=f"{name}_y_expand",
            )

    select_layer = network.add_select(condition_val, x_val, y_val)

    set_layer_name(select_layer, target, f"{name}_select")

    return select_layer.get_output(0)