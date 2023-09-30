import logging
from typing import Any, List, Optional, Sequence, Union, cast, Tuple

import numpy as np
import tensorrt as trt
import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import get_positive_dim, to_numpy
from torch_tensorrt.dynamo.conversion.impl.elementwise.base import (
    convert_binary_elementwise,
)
from torch_tensorrt.dynamo.conversion.impl.unary.base import convert_unary
from torch_tensorrt.fx.converters.converter_utils import (
    get_positive_dim,
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.fx.types import TRTTensor
from torch_tensorrt.fx.utils import get_dynamic_dims

_LOGGER: logging.Logger = logging.getLogger(__name__)


def native_batch_norm(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: Optional[Union[torch.Tensor, np.ndarray]],
    bias: Optional[Union[torch.Tensor, np.ndarray]],
    running_mean: Optional[Union[torch.Tensor, np.ndarray]],
    running_var: Optional[Union[torch.Tensor, np.ndarray]],
    training: bool,
    momentum: float,
    eps: float,
) -> Tuple[TRTTensor, TRTTensor, TRTTensor]:
    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for batch norm."

    if weight is None:
        weight = 1.0

    if bias is None:
        bias = 0.0

    if running_mean is None:
        running_mean = 0.0

    if running_var is None:
        running_var = 1.0

    scale = to_numpy(weight) / np.sqrt(to_numpy(running_var) + eps)
    bias = to_numpy(bias) - to_numpy(running_mean) * scale
    power = np.ones_like(scale)

    # For BatchNorm1d, reshape 1d to 2d
    output_shape = input.shape
    if not ctx.net.has_implicit_batch_dimension and len(input.shape) < 4:
        assert (
            len(get_dynamic_dims(input.shape)) <= 1
        ), "BatchNorm1D with more than one dynamic dims is not currently supported."
        reshape_layer = ctx.net.add_shuffle(input)
        if len(input.shape) == 2:
            reshape_layer.reshape_dims = (input.shape[0], input.shape[1], 1, 1)
        else:  # len(input_val.shape) == 3
            reshape_layer.reshape_dims = (
                input.shape[0],
                input.shape[1],
                input.shape[2],
                1,
            )
        set_layer_name(reshape_layer, target, f"{name}_reshape_2d", source_ir)
        input = reshape_layer.get_output(0)
    layer = ctx.net.add_scale(input, trt.ScaleMode.CHANNEL, bias, scale, power)
    set_layer_name(layer, target, name)

    # For BatchNorm1d, reshape output back to 1d
    if not ctx.net.has_implicit_batch_dimension and len(output_shape) < 4:
        reshape_output_layer = ctx.net.add_shuffle(layer.get_output(0))
        reshape_output_layer.reshape_dims = tuple(output_shape)
        set_layer_name(reshape_output_layer, target, f"{name}_reshape_1d", source_ir)
        layer = reshape_output_layer
    
    
    # 1 / sqrt((var + eps))
    save_rstd = 1 / (torch.sqrt(running_var + eps))
    
    # eps_tensor = ctx.net.add_constant(
    #     (1,) * len(running_var.shape),
    #     trt.Weights(np.ascontiguousarray([eps], dtype=np.float32)),
    # )
    
    # eps_tensor = ctx.net.add_constant(
    #     (1,) * len(input.shape),
    #     trt.Weights(np.ascontiguousarray([eps], dtype=np.float32)),
    # )
    
    # add_trt = convert_binary_elementwise(
    #     ctx,
    #     target,
    #     source_ir,
    #     f"{name}_add",
    #     running_var,
    #     eps_tensor,
    # )
    
    # sqrt_trt = convert_unary(
    #     ctx,
    #     target,
    #     source_ir,
    #     f"{name}_sqrt",
    #     add_trt,
    # )
    
            
    return layer.get_output(0), running_mean, save_rstd


def layer_norm(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    normalized_shape: List[int],
    weight: Optional[Union[torch.Tensor, np.ndarray]],
    bias: Optional[Union[torch.Tensor, np.ndarray]],
    eps: float,
    cudnn_enable: bool,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    if weight is None:
        weight = to_numpy(1.0)

    if bias is None:
        bias = to_numpy(0.0)

    shape = weight.shape
    broadcasted_shape = (1,) * (len(input.shape) - len(shape)) + shape
    gamma = to_numpy(weight).reshape(shape)
    beta = to_numpy(bias).reshape(shape)

    dims = list(range(len(input.shape) - len(shape), len(input.shape)))
    axes = get_axes_for_reduce_op(dims)

    # E[x]
    mean_expected_layer = ctx.net.add_reduce(
        input, trt.ReduceOperation.AVG, axes, keep_dims=True
    )
    set_layer_name(mean_expected_layer, target, f"{name}_mean_expected", source_ir)

    # X-E[x]
    sub_trt = convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        f"{name}_sub",
        input,
        mean_expected_layer.get_output(0),
    )
    # Variance = mean(pow(x_sub_mean,2))
    pow_tensor = ctx.net.add_constant(
        (1,) * len(input.shape),
        trt.Weights(np.ascontiguousarray([2.0], dtype=np.float32)),
    )
    pow_tensor.name = f"{name}_power"
    pow_var = convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        f"{name}_pow_var",
        sub_trt,
        pow_tensor.get_output(0),
    )
    mean_trt_layer = ctx.net.add_reduce(
        pow_var, trt.ReduceOperation.AVG, axes, keep_dims=True
    )
    set_layer_name(mean_trt_layer, target, f"{name}_mean", source_ir)
    # Variance + eps
    eps_tensor = ctx.net.add_constant(
        (1,) * len(input.shape),
        trt.Weights(np.ascontiguousarray([eps], dtype=np.float32)),
    )
    eps_tensor.name = f"{name}_eps"
    add_trt = convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        f"{name}_add",
        mean_trt_layer.get_output(0),
        eps_tensor.get_output(0),
    )
    # SQRT((Var + eps))
    sqrt_trt = convert_unary(
        ctx,
        target,
        source_ir,
        f"{name}_sqrt",
        add_trt,
    )
    # (x - E[x]) / sqrt((var + eps))
    div_trt = convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        f"{name}_div_trt",
        sub_trt,
        sqrt_trt,
    )

    assert gamma is not None
    gamma_tensor = ctx.net.add_constant(
        gamma.shape, trt.Weights(np.ascontiguousarray(gamma))
    )
    gamma_tensor.name = f"{name}_gamma"

    assert beta is not None
    beta_tensor = ctx.net.add_constant(
        gamma.shape, trt.Weights(np.ascontiguousarray(beta))
    )
    beta_tensor.name = f"{name}_beta"

    # y * gamma + beta
    scale_layer = convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        f"{name}_scale",
        div_trt,
        gamma_tensor.get_output(0),
    )
    return convert_binary_elementwise(
        ctx,
        target,
        source_ir,
        name,
        scaled_y,
        beta_tensor.get_output(0),
    )


def native_group_norm(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: Optional[Union[torch.Tensor, np.ndarray]],
    bias: Optional[Union[torch.Tensor, np.ndarray]],
    N: int,
    C: int,
    HxW: int,
    group: int,
    eps: float,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return group_norm(
        ctx,
        target,
        source_ir,
        name,
        input,
        group,
        weight,
        bias,
        eps,
        cudnn_enabled=True,
    )


def group_norm(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    num_groups: int,
    weight: Optional[Union[torch.Tensor, np.ndarray]],
    bias: Optional[Union[torch.Tensor, np.ndarray]],
    eps: float,
    cudnn_enabled: bool,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    if weight is None:
        weight = to_numpy(1.0)

    if bias is None:
        bias = to_numpy(0.0)

    assert (
        len(input.shape) >= 3
    ), f"The input dimension should not be less than 3, got {len(input.shape)}!"
    B, C = input.shape[0], input.shape[1]

    # Groups are a subdivision of the channel dimension.
    assert (
        C % num_groups == 0
    ), f"The num of channels ({C}) should be divisible by num_groups ({num_groups})!"

    # Normalize every group.
    reshaped_input = impl.shuffle.reshape(
        network,
        target,
        SourceIR.ATEN,
        name,
        input,
        shape=(B * num_groups, -1),
    )
    dim = (
        len(reshaped_input.shape) - 1
    )  # TODO: PR #2347 supported negtive dimension in reduce, could be -1

    # E[X]
    mean_trt = impl.reduce.mean(
        network,
        target,
        SourceIR.ATEN,
        f"{name}_mean",
        reshaped_input,
        dim=dim,
        keepdim=True,
    )

    # X - E[X]
    sub_trt = impl.elementwise.sub(
        network,
        target,
        source_ir,
        f"{name}_sub",
        reshaped_input,
        mean_trt,
    )

    # variance = mean(pow(sub_trt, 2))
    pow_layer = network.add_constant(
        (1,) * len(sub_trt.shape),
        trt.Weights(np.ascontiguousarray([2.0], dtype=np.float32)),
    )
    pow_layer.name = f"{name}_power"

    pow_var = impl.elementwise.pow(
        network,
        target,
        source_ir,
        f"{name}_pow",
        sub_trt,
        pow_layer.get_output(0),
    )

    var_trt = impl.reduce.mean(
        network,
        target,
        SourceIR.ATEN,
        f"{name}_mean_var",
        pow_var,
        dim=dim,
        keepdim=True,
    )

    # sqrt((var + eps))
    eps_layer = network.add_constant(
        (1,) * len(reshaped_input.shape),
        trt.Weights(np.ascontiguousarray([eps], dtype=np.float32)),
    )
    eps_layer.name = f"{name}_eps"

    add_trt = impl.elementwise.add(
        network,
        target,
        source_ir,
        f"{name}_add",
        var_trt,
        eps_layer.get_output(0),
    )
    sqrt_trt = impl.unary.sqrt(
        network,
        target,
        source_ir,
        f"{name}_sqrt",
        add_trt,
    )

    # (X - E[X]) / sqrt((var + eps))
    div_trt = impl.elementwise.div(
        network,
        target,
        source_ir,
        f"{name}_div",
        sub_trt,
        sqrt_trt,
    )

    # Apply per-channel scale and bias.
    output = impl.shuffle.reshape(
        network,
        target,
        SourceIR.ATEN,
        f"{name}_reshape_div",
        div_trt,
        shape=input.shape,
    )

    weight_bias_shape = (1, C) + (1,) * (len(input.shape) - 2)

    reshaped_weight = impl.shuffle.reshape(
        network,
        target,
        SourceIR.ATEN,
        f"{name}_reshape_weight",
        weight,
        shape=weight_bias_shape,
    )

    output = impl.elementwise.mul(
        network,
        target,
        SourceIR.ATEN,
        f"{name}_mul_scale",
        output,
        reshaped_weight,
    )

    reshaped_bias = impl.shuffle.reshape(
        network,
        target,
        SourceIR.ATEN,
        f"{name}_reshape_bias",
        bias,
        shape=weight_bias_shape,
    )

    add_trt = impl.elementwise.add(
        network,
        target,
        source_ir,
        f"{name}_add_bias",
        output,
        reshaped_bias,
    )

    # TODO: compute the last two return values
    # const1_layer = network.add_constant(
    #     (1,) * len(sqrt_trt.shape),
    #     trt.Weights(np.ascontiguousarray([1.0], dtype=np.float32)),
    # )
    # const1_layer.name = f"{name}_const1"

    # rsqrt_trt = impl.elementwise.div(
    #     network,
    #     target,
    #     source_ir,
    #     f"{name}_rsqrt",
    #     const1_layer.get_output(0),
    #     sqrt_trt,
    # )

    return add_trt, torch.tensor(0), torch.tensor(0)


def softmax(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Optional[Any] = None,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_ranks = len(input.shape) + (1 if ctx.net.has_implicit_batch_dimension else 0)

    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"softmax received input {input} that is not part "
            "of the TensorRT region!"
        )

    # Used to get dim when dim is None. Copied from PyTorch softmax implementation.
    def get_softmax_dim(ndim: int) -> int:
        if ndim == 0 or ndim == 1 or ndim == 3:
            ret = 0
        else:
            ret = 1
        return ret

    if dim is None:
        dim = get_softmax_dim(input_ranks)
    else:
        dim = cast(int, dim)

    dim = get_positive_dim(dim, input_ranks)
    if ctx.net.has_implicit_batch_dimension:
        assert dim != 0, "Can't apply softmax on batch dimension when it's implicit."
        dim -= 1

    layer = ctx.net.add_softmax(input)
    layer.axes = 1 << dim
    set_layer_name(layer, target, name)
    return layer.get_output(0)
