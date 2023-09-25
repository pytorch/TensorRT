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
    get_trt_plugin,
    get_trt_tensor,
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

    scale = cast(torch.Tensor, to_numpy(weight)) / np.sqrt(
        cast(torch.Tensor, to_numpy(running_var)) + eps
    )

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
    if not isinstance(input, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"LayerNorm received input {input} that is not part "
            "of the TensorRT region!"
        )

    if weight is None:
        weight = to_numpy(1.0)

    if bias is None:
        bias = to_numpy(0.0)

    gamma = (
        weight.detach().cpu().float().numpy()
        if isinstance(weight, torch.Tensor)
        else weight
    )
    gamma_field = trt.PluginField("gamma", gamma, trt.PluginFieldType.FLOAT32)
    beta = (
        bias.detach().cpu().float().numpy() if isinstance(bias, torch.Tensor) else bias
    )
    beta_field = trt.PluginField("beta", beta, trt.PluginFieldType.FLOAT32)
    eps_field = trt.PluginField(
        "eps", np.array(eps, dtype=np.float32), trt.PluginFieldType.FLOAT32
    )
    try:
        normalized_shape_arr = np.array(normalized_shape, dtype=np.int32)
    except TypeError:
        _LOGGER.error("Unable to convert normalized_shape to a field, fall back to []")
        normalized_shape_arr = np.array([], dtype=np.int32)

    normalized_shape_filed = trt.PluginField(
        "normalized_shape", normalized_shape_arr, trt.PluginFieldType.INT32
    )
    field_collection = trt.PluginFieldCollection(
        [gamma_field, beta_field, eps_field, normalized_shape_filed]
    )

    try:
        if ctx.net.has_implicit_batch_dimension:
            plugin = get_trt_plugin("layer_norm", field_collection, "1", "fx2trt")
        else:
            plugin = get_trt_plugin("LayerNormDynamic", field_collection, "1", "fx2trt")
    except AssertionError:
        _LOGGER.error(
            "Unable to find layer norm plugin, fall back to TensorRT implementation."
        )
        return layer_norm_no_plugin(
            ctx, target, source_ir, name, input, normalized_shape, weight, bias, eps
        )
    layer = ctx.net.add_plugin_v2([input], plugin)
    layer.name = name
    return layer.get_output(0)


def layer_norm_no_plugin(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    normalized_shape: List[int],
    weight: Optional[Union[torch.Tensor, np.ndarray]],
    bias: Optional[Union[torch.Tensor, np.ndarray]],
    eps: float,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"LayerNorm received input {input} that is not part "
            "of the TensorRT region!"
        )

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
    if not isinstance(input, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"LayerNorm received input {input} that is not part "
            "of the TensorRT region!"
        )

    if weight is None:
        weight = to_numpy(1.0)

    if bias is None:
        bias = to_numpy(0.0)

    scale = get_trt_tensor(network, weight, "scale")
    bias = get_trt_tensor(network, bias, "bias")

    eps_field = trt.PluginField(
        "eps", np.array(eps, dtype=np.float32), trt.PluginFieldType.FLOAT32
    )
    num_groups_filed = trt.PluginField(
        "num_groups", np.array(num_groups), trt.PluginFieldType.INT32
    )

    field_collection = trt.PluginFieldCollection([eps_field, num_groups_filed])

    try:
        # Here's the schema of the plugin:
        # https://github.com/NVIDIA/TensorRT/blob/release/8.6/plugin/groupNormalizationPlugin/GroupNormalizationPlugin_PluginConfig.yaml
        plugin = get_trt_plugin("GroupNormalizationPlugin", field_collection, "1")
    except AssertionError:
        _LOGGER.error(
            "Unable to find group norm plugin, fall back to TensorRT implementation."
        )

    layer = network.add_plugin_v2([input, scale, bias], plugin)
    set_layer_name(layer, target, f"{name}_GroupNormalizationPlugin", source_ir)

    # PyTorch requires three return values: (out, mean, rstd)
    dummy_tensor = torch.tensor(0)
    return layer.get_output(0), dummy_tensor, dummy_tensor


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
