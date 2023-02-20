import operator
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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

def add_activation_layer(
    network: TRTNetwork,
    input_val: TRTTensor,
    operation_type: trt.ActivationType,
    target: Target,
    name: str,
    alpha: Optional[Any] = None,
    beta: Optional[Any] = None,
    dyn_range_fn: Optional[Callable[Tuple[float, float]]] = None
) -> TRTTensor:
    """
    Add a TensorRT Activation layer to `network`.

    Args:
        network (TRTNetwork): TensorRT network object.
        input_val (TRTTensor): Input to the activation op.
            Must be a TensorRT tensor.
        op_type (trt.ElementWiseOperation): Type of the TensorRT activation
            operation.
        target (Target): Target of fx node.
        name (str): The name we want to assign to the created TensorRT layer.
        alpha (Optional[Any]): If not None, we will use it to set the alpha
            attribute of the created TensorRT activation layer.
        beta (Optional[Any]): If not None, we will use it to set the beta
            attribute of the created TensorRT activation layer.
        dyn_range_fn: Optional[Callable[Tuple[float, float]]]: A function which takes the dynamic range of a TensorRT Tensor and returns the output dynamic range


    Returns:
        The output of TensorRT Activation layer.
    """
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"{operation_type} received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    layer = network.add_activation(input_val, operation_type)
    if alpha is not None:
        layer.alpha = alpha
    if beta is not None:
        layer.beta = beta
    set_layer_name(layer, target, name)

    if input_val.dynamic_range is not None:
        dyn_range = dyn_range_fn(input_val.dynamic_range)
        mark_as_int8_layer(layer, dyn_range)

    return layer.get_output(0)

def add_elu(
    network: TRTNetwork,
    target: Target,
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    input_val = kwargs["input"]
    alpha = kwargs["alpha"]
    operation_type = trt.ActivationType.ELU
    return add_activation_layer(network, input_val, operation_type, target, name, alpha)

def add_gelu(
    network: TRTNetwork,
    target: Target,
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    input_val = kwargs["input"]
    approximate = kwargs["approximate"]
    if approximate != "none":
        raise RuntimeError("GeLU converter currently doesn't support fast gelu compute")
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"GELU received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "GeLU converter currently doesn't support implicit batch dimension"
        )

    plugin_name = "CustomGeluPluginDynamic"
    # type_id 0 for float32, 1 for  float16
    type_id = trt.PluginField(
        "type_id", np.array(0, dtype=np.int32), trt.PluginFieldType.INT32
    )
    field_collection = TRTPluginFieldCollection([type_id])
    plugin_version = "1"

    plugin = get_trt_plugin(plugin_name, field_collection, plugin_version)

    layer = network.add_plugin_v2([input_val], plugin)
    set_layer_name(layer, target, name)
    return layer.get_output(0)

def add_hard_sigmoid(
    network: TRTNetwork,
    target: Target,
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    input_val = kwargs["input"]
    return add_activation_layer(
        network,
        input_val,
        trt.ActivationType.HARD_SIGMOID,
        target,
        name,
        alpha=1 / 6,
        beta=0.5,
    )

def add_hardtanh(
    network: TRTNetwork,
    target: Target,
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    input_val = kwargs["input"]

    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"hardtanh received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    return add_activation_layer(
        network,
        input_val,
        trt.ActivationType.CLIP,
        target,
        name,
        alpha=kwargs["min_val"],
        beta=kwargs["max_val"],
    )


def add_leaky_relu(
    network: TRTNetwork,
    target: Target,
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    input_val = kwargs["input"]
    negative_slope = kwargs["negative_slope"]
    operation_type = trt.ActivationType.LEAKY_RELU
    return add_activation_layer(
        network, input_val, operation_type, target, name, negative_slope
    )

def add_relu(
    network: TRTNetwork,
    target: Target,
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    input_val = kwargs["input"]
    operation_type = trt.ActivationType.RELU

    def activation_dyn_range_fn(dyn_range):
        return max(0, dyn_range[0]), max(0, dyn_range[1])

    return add_activation_layer(network, input_val, operation_type, target, name, dyn_range_fn=activation_dyn_range_fn)

def add_selu(
    network: TRTNetwork,
    target: Target,
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    input_val = kwargs["input"]
    operation_type = trt.ActivationType.SELU
    return add_activation_layer(network, input_val, operation_type, target, name)

def add_sigmoid(
    network: TRTNetwork,
    target: Target,
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    input_val = kwargs["input"]

    def activation_dyn_range_fn(dyn_range):
        def sigmoid_fn(x):
            return 1 / (1 + np.exp(-x))

        return sigmoid_fn(dyn_range[0]), sigmoid_fn(dyn_range[1])

    return add_activation_layer(
        network, input_val, trt.ActivationType.SIGMOID, target, name, dyn_range_fn=activation_dyn_range_fn
    )


def add_softsign(
    network: TRTNetwork,
    target: Target,
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    input_val = kwargs["input"]
    operation_type = trt.ActivationType.SOFTSIGN
    return add_activation_layer(network, input_val, operation_type, target, name)

def add_tanh(
    network: TRTNetwork,
    target: Target,
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    input_val = kwargs["input"]
    operation_type = trt.ActivationType.TANH
    return add_activation_layer(network, input_val, operation_type, target, name)
