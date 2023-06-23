from typing import cast, Union, Any, Optional, Sequence

import numpy as np

import tensorrt as trt
import torch
from torch.fx.node import Target

import logging

from torch_tensorrt.fx.types import TRTNetwork, TRTTensor
from torch_tensorrt.fx.utils import get_dynamic_dims
from torch_tensorrt.dynamo._SourceIR import SourceIR

from torch_tensorrt.fx.converters.converter_utils import (
    get_trt_plugin,
    set_layer_name,
    to_numpy,
    has_dynamic_shape,
    get_positive_dim,
)

from torch_tensorrt.dynamo.conversion.impl.unary.base import (
    convert_unary,
)

from torch_tensorrt.dynamo.conversion.impl.elementwise.base import (
    convert_binary_elementwise,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)


def batch_norm(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    training: torch.Tensor,
    momentum: torch.Tensor,
    eps: list,
) -> Union[TRTTensor, Sequence[TRTTensor]]:

    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"BatchNorm2d received input {input} that is not part "
            "of the TensorRT region!"
        )

    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for batch norm."

    scale = cast(torch.Tensor, to_numpy(cast(torch.Tensor, weight))) / np.sqrt(
        cast(torch.Tensor, to_numpy(cast(torch.Tensor, running_var))) + cast(float, eps)
    )

    bias = (
        to_numpy(cast(torch.Tensor, bias))
        - to_numpy(cast(torch.Tensor, running_mean)) * scale
    )
    power = np.ones_like(scale)

    # For BatchNorm1d, reshape 1d to 2d
    output_shape = input.shape
    if not network.has_implicit_batch_dimension and len(input.shape) < 4:
        assert (
            len(get_dynamic_dims(input.shape)) <= 1
        ), "BatchNorm1D with more than one dynamic dims is not currently supported."
        reshape_layer = network.add_shuffle(input)
        if len(input.shape) == 2:
            reshape_layer.reshape_dims = (input.shape[0], input.shape[1], 1, 1)
        else:  # len(input_val.shape) == 3
            reshape_layer.reshape_dims = (
                input.shape[0],
                input.shape[1],
                input.shape[2],
                1,
            )
        set_layer_name(reshape_layer, target, f"{name}_reshape_2d")
        input = reshape_layer.get_output(0)
    layer = network.add_scale(input, trt.ScaleMode.CHANNEL, bias, scale, power)
    set_layer_name(layer, target, name)

    # For BatchNorm1d, reshape output back to 1d
    if not network.has_implicit_batch_dimension and len(output_shape) < 4:
        reshape_output_layer = network.add_shuffle(layer.get_output(0))
        reshape_output_layer.reshape_dims = tuple(output_shape)
        set_layer_name(reshape_output_layer, target, f"{name}_reshape_1d")
        layer = reshape_output_layer
    return layer.get_output(0)


def layer_norm(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    normalized_shape: list,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: list,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    if not isinstance(input, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"LayerNorm received input {input} that is not part "
            "of the TensorRT region!"
        )

    gamma = weight.detach().cpu().float().numpy()
    gamma_field = trt.PluginField("gamma", gamma, trt.PluginFieldType.FLOAT32)
    beta = bias.detach().cpu().float().numpy()
    beta_field = trt.PluginField("beta", beta, trt.PluginFieldType.FLOAT32)
    eps_field = trt.PluginField(
        "eps", np.array(eps, dtype=np.float32), trt.PluginFieldType.FLOAT32
    )
    try:
        normalized_shape = np.array(normalized_shape, dtype=np.int32)
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
        return layer_norm_no_plugin(
            network, target, source_ir, name, input, normalized_shape, weight, bias, eps
        )
    layer = network.add_plugin_v2([input], plugin)
    layer.name = name
    return layer.get_output(0)


def layer_norm_no_plugin(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    normalized_shape: list,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: list,
) -> Union[TRTTensor, Sequence[TRTTensor]]:

    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"LayerNorm received input {input} that is not part "
            "of the TensorRT region!"
        )

    shape = weight.shape  # type: ignore[union-attr]
    broadcasted_shape = (1,) * (len(input.shape) - len(shape)) + shape
    gamma = to_numpy(weight.reshape(*shape))  # type: ignore[union-attr]
    beta = to_numpy(bias.reshape(*shape))  # type: ignore[union-attr]

    axes = 0
    for d in range(len(shape)):
        axes |= 1 << (len(input.shape) - d - 1)

    # E[x]
    mean_expected_layer = network.add_reduce(
        input, trt.ReduceOperation.AVG, axes, keep_dims=True
    )
    set_layer_name(mean_expected_layer, target, f"{name}_mean_expected")

    # X-E[x]
    sub_trt = convert_binary_elementwise(
        network,
        target,
        source_ir,
        f"{name}_sub",
        trt.ElementWiseOperation.SUB,
        input,
        mean_expected_layer.get_output(0),
    )
    # Variance = mean(pow(x_sub_mean,2))
    pow_tensor = network.add_constant(
        (1,) * len(input.shape),
        trt.Weights(np.ascontiguousarray([2.0], dtype=np.float32)),
    )
    pow_tensor.name = f"{name}_power"
    pow_var = convert_binary_elementwise(
        network,
        target,
        source_ir,
        f"{name}_pow_var",
        trt.ElementWiseOperation.POW,
        sub_trt,
        pow_tensor.get_output(0),
    )
    mean_trt_layer = network.add_reduce(
        pow_var, trt.ReduceOperation.AVG, axes, keep_dims=True
    )
    set_layer_name(mean_trt_layer, target, f"{name}_mean")
    # Variance + eps
    eps_tensor = network.add_constant(
        (1,) * len(input.shape),
        trt.Weights(np.ascontiguousarray([eps], dtype=np.float32)),
    )
    eps_tensor.name = f"{name}_eps"
    add_trt = convert_binary_elementwise(
        network,
        target,
        source_ir,
        f"{name}_add",
        trt.ElementWiseOperation.SUM,
        mean_trt_layer.get_output(0),
        eps_tensor.get_output(0),
    )
    # SQRT((Var + eps))
    sqrt_trt = convert_unary(
        network,
        target,
        source_ir,
        f"{name}_sqrt",
        trt.UnaryOperation.SQRT,
        add_trt,
    )
    # (x - E[x]) / sqrt((var + eps))
    div_trt = convert_binary_elementwise(
        network,
        target,
        source_ir,
        f"{name}_div_trt",
        trt.ElementWiseOperation.DIV,
        sub_trt,
        sqrt_trt,
    )

    assert gamma is not None
    gamma_tensor = network.add_constant(gamma.shape, trt.Weights(np.ascontiguousarray(gamma)))  # type: ignore[attr-defined]
    gamma_tensor.name = f"{name}_gamma"
    assert beta is not None
    beta_tensor = network.add_constant(gamma.shape, trt.Weights(np.ascontiguousarray(beta)))  # type: ignore[attr-defined]
    beta_tensor.name = f"{name}_beta"
    # y * gamma + beta
    scale_layer = convert_binary_elementwise(
        network,
        target,
        source_ir,
        f"{name}_scale",
        trt.ElementWiseOperation.PROD,
        div_trt,
        gamma_tensor.get_output(0),
    )
    return convert_binary_elementwise(
        network,
        target,
        source_ir,
        name,
        trt.ElementWiseOperation.SUM,
        scale_layer,
        beta_tensor.get_output(0),
    )


def softmax(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Optional[Any] = None,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_ranks = len(input.shape) + (1 if network.has_implicit_batch_dimension else 0)  # type: ignore[union-attr]

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
    if network.has_implicit_batch_dimension:
        assert dim != 0, "Can't apply softmax on batch dimension when it's implicit."
        dim -= 1

    layer = network.add_softmax(input)
    layer.axes = 1 << dim
    set_layer_name(layer, target, name)
    return layer.get_output(0)
