from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import tensorrt as trt
import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_positive_dim,
    get_trt_tensor,
    to_numpy,
)
from torch_tensorrt.fx.converters.converter_utils import (
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.fx.types import TRTTensor
from torch_tensorrt.fx.utils import get_dynamic_dims


def batch_norm(
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
    cudnn_enabled: bool,
    return_mean_rstd: bool,
) -> Union[TRTTensor, Tuple[TRTTensor, torch.Tensor, torch.Tensor]]:
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
        new_shape = (
            (input.shape[0], input.shape[1], 1, 1)
            if len(input.shape) == 2
            else (input.shape[0], input.shape[1], input.shape[2], 1)
        )
        input = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_reshape_2d", input, new_shape
        )
    layer = ctx.net.add_scale(input, trt.ScaleMode.CHANNEL, bias, scale, power)
    set_layer_name(layer, target, name, source_ir)
    output = layer.get_output(0)

    # For BatchNorm1d, reshape output back to 1d
    if not ctx.net.has_implicit_batch_dimension and len(output_shape) < 4:
        output = impl.shuffle.reshape(
            ctx,
            target,
            source_ir,
            f"{name}_reshape_1d",
            layer.get_output(0),
            output_shape,
        )

    if return_mean_rstd:
        # return fake mean and rstd for now
        return output, None, None

    return output


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
    return_mean_rstd: bool,
) -> Union[TRTTensor, Tuple[TRTTensor, torch.Tensor, torch.Tensor]]:
    if weight is None:
        weight = to_numpy(1.0)

    if bias is None:
        bias = to_numpy(0.0)

    shape = weight.shape
    gamma = to_numpy(weight).reshape(shape)
    beta = to_numpy(bias).reshape(shape)

    dims = list(range(len(input.shape) - len(shape), len(input.shape)))

    # E[x]
    mean_expected_trt = impl.reduce.mean(
        ctx, target, source_ir, f"{name}_mean_expected", input, dims, True
    )

    # X-E[x]
    sub_trt = impl.elementwise.sub(
        ctx,
        target,
        source_ir,
        f"{name}_sub",
        input,
        mean_expected_trt,
    )

    # Variance = mean(pow(x_sub_mean, 2))
    pow_trt = get_trt_tensor(ctx, 2, f"{name}_power", np.float32)
    pow_var = impl.elementwise.pow(
        ctx,
        target,
        source_ir,
        f"{name}_pow_var",
        sub_trt,
        pow_trt,
    )
    mean_trt = impl.reduce.mean(
        ctx, target, source_ir, f"{name}_mean", pow_var, dims, True
    )

    # sqrt((var + eps))
    eps_trt = get_trt_tensor(ctx, eps, f"{name}_eps", np.float32)
    add_trt = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_add",
        mean_trt,
        eps_trt,
    )
    sqrt_trt = impl.unary.sqrt(
        ctx,
        target,
        source_ir,
        f"{name}_sqrt",
        add_trt,
    )

    # (X - E[X]) / sqrt((var + eps))
    div_trt = impl.elementwise.div(
        ctx,
        target,
        source_ir,
        f"{name}_div",
        sub_trt,
        sqrt_trt,
    )

    gamma_trt = get_trt_tensor(ctx, weight, f"{name}_gamma")
    beta_trt = get_trt_tensor(ctx, bias, f"{name}_beta")

    # y * gamma + beta
    scaled_y = impl.elementwise.mul(
        ctx,
        target,
        source_ir,
        f"{name}_mul_gamma",
        div_trt,
        gamma_trt,
    )

    output = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_add_beta",
        scaled_y,
        beta_trt,
    )

    if return_mean_rstd:
        # return fake mean and rstd for now
        return output, None, None

    return output


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
    return_mean_rstd: bool = True,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    assert (
        len(input.shape) >= 3
    ), f"The input dimension should not be less than 3, got {len(input.shape)}!"
    B, C = input.shape[0], input.shape[1]

    # Groups are a subdivision of the channel dimension.
    assert (
        C % group == 0
    ), f"The num of channels ({C}) should be divisible by num_groups ({group})!"

    if weight is None:
        weight = to_numpy(1.0)

    if bias is None:
        bias = to_numpy(0.0)

    # Normalize every group.
    reshaped_input = impl.shuffle.reshape(
        ctx,
        target,
        source_ir,
        name,
        input,
        (B * group, -1),
    )

    dim = 1

    # E[X]
    mean_trt = impl.reduce.mean(
        ctx,
        target,
        source_ir,
        f"{name}_mean",
        reshaped_input,
        dim,
        True,
    )

    # X - E[X]
    sub_trt = impl.elementwise.sub(
        ctx,
        target,
        source_ir,
        f"{name}_sub",
        reshaped_input,
        mean_trt,
    )

    # variance = mean(pow(sub_trt, 2))
    pow_trt = get_trt_tensor(ctx, 2, f"{name}_power", np.float32)
    pow_var = impl.elementwise.pow(
        ctx,
        target,
        source_ir,
        f"{name}_pow",
        sub_trt,
        pow_trt,
    )

    var_trt = impl.reduce.mean(
        ctx,
        target,
        source_ir,
        f"{name}_mean_var",
        pow_var,
        dim,
        True,
    )

    # sqrt((var + eps))
    eps_trt = get_trt_tensor(ctx, eps, f"{name}_eps", np.float32)
    add_trt = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_add",
        var_trt,
        eps_trt,
    )
    sqrt_trt = impl.unary.sqrt(
        ctx,
        target,
        source_ir,
        f"{name}_sqrt",
        add_trt,
    )

    # y = (X - E[X]) / sqrt((var + eps))
    div_trt = impl.elementwise.div(
        ctx,
        target,
        source_ir,
        f"{name}_div",
        sub_trt,
        sqrt_trt,
    )

    # y * gamma + beta
    gamma_trt = get_trt_tensor(ctx, weight, f"{name}_gamma")
    beta_trt = get_trt_tensor(ctx, bias, f"{name}_beta")

    output = impl.shuffle.reshape(
        ctx,
        target,
        source_ir,
        f"{name}_reshape_div",
        div_trt,
        input.shape,
    )

    weight_bias_shape = (1, C) + (1,) * (len(input.shape) - 2)

    reshaped_gamma = impl.shuffle.reshape(
        ctx,
        target,
        source_ir,
        f"{name}_reshape_gamma",
        gamma_trt,
        weight_bias_shape,
    )

    output = impl.elementwise.mul(
        ctx,
        target,
        source_ir,
        f"{name}_mul_gamma",
        output,
        reshaped_gamma,
    )

    reshaped_bias = impl.shuffle.reshape(
        ctx,
        target,
        source_ir,
        f"{name}_reshape_beta",
        beta_trt,
        weight_bias_shape,
    )

    output = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_add_beta",
        output,
        reshaped_bias,
    )

    if return_mean_rstd:
        # return fake mean and rstd for now
        return output, None, None

    return output


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
    return native_group_norm(
        ctx,
        target,
        source_ir,
        name,
        input,
        weight,
        bias,
        0,
        0,
        0,
        num_groups,
        eps,
        return_mean_rstd=False,
    )


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
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)
