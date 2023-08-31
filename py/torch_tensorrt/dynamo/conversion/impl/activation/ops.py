from typing import Any, Optional

import numpy as np
import tensorrt as trt
import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion.impl.activation.base import convert_activation
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor


def relu(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
):
    operation_type = trt.ActivationType.RELU

    def relu_dyn_range_fn(dyn_range):
        return max(0, dyn_range[0]), max(0, dyn_range[1])

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        dyn_range_fn=relu_dyn_range_fn,
    )


def sigmoid(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
):
    operation_type = trt.ActivationType.SIGMOID

    def sigmoid_dyn_range_fn(dyn_range):
        def sigmoid_fn(x):
            return 1 / (1 + np.exp(-x))

        return sigmoid_fn(dyn_range[0]), sigmoid_fn(dyn_range[1])

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        dyn_range_fn=sigmoid_dyn_range_fn,
    )


def tanh(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
):
    operation_type = trt.ActivationType.TANH

    def tanh_dyn_range_fn(dyn_range):
        def tanh_fn(x):
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

        return tanh_fn(dyn_range[0]), tanh_fn(dyn_range[1])

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        dyn_range_fn=tanh_dyn_range_fn,
    )


def leaky_relu(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    alpha: Optional[Any] = 0.01,
):
    operation_type = trt.ActivationType.LEAKY_RELU

    def leaky_relu_dyn_range_fn(dyn_range):
        def leaky_relu_fn(x):
            return max(0, x) + alpha * min(0, x)

        return leaky_relu_fn(dyn_range[0]), leaky_relu_fn(dyn_range[1])

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        alpha,
        dyn_range_fn=leaky_relu_dyn_range_fn,
    )


def elu(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    alpha: Optional[Any] = 1.0,
    beta: Optional[Any] = None,
):
    EPS = 1e-4
    # actually call selu()
    if (
        abs(alpha - 1.6732632423543772) < EPS
        and beta is not None
        and abs(beta - 1.0507009873554805) < EPS
    ):
        print("Selu is called but re-uses elu function!")
        return selu(network, target, source_ir, name, input_val)

    else:
        operation_type = trt.ActivationType.ELU

        def elu_dyn_range_fn(dyn_range):
            return (
                torch.nn.functional.elu(dyn_range[0], alpha),
                torch.nn.functional.elu(dyn_range[1], alpha),
            )

        return convert_activation(
            network,
            target,
            source_ir,
            name,
            operation_type,
            input_val,
            alpha,
            dyn_range_fn=elu_dyn_range_fn,
        )


def selu(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
):
    operation_type = trt.ActivationType.SELU

    def selu_dyn_range_fn(dyn_range):
        return (
            torch.nn.functional.selu(dyn_range[0]),
            torch.nn.functional.selu(dyn_range[1]),
        )

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        dyn_range_fn=selu_dyn_range_fn,
    )


# no corresponding function in aten/native_functions
def softsign(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
):
    operation_type = trt.ActivationType.SOFTSIGN

    def softsign_dyn_range_fn(dyn_range):
        return (
            torch.nn.functional.softsign(dyn_range[0]),
            torch.nn.functional.softsign(dyn_range[1]),
        )

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        dyn_range_fn=softsign_dyn_range_fn,
    )


def softplus(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    beta: Optional[Any] = 1,
):
    operation_type = trt.ActivationType.SOFTPLUS

    def softplus_dyn_range_fn(dyn_range):
        return (
            torch.nn.functional.softplus(dyn_range[0], beta),
            torch.nn.functional.softplus(dyn_range[1], beta),
        )

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        alpha=1 / beta,
        beta=beta,
        dyn_range_fn=softplus_dyn_range_fn,
    )


def clip(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    alpha: Optional[Any],
    beta: Optional[Any],
):
    operation_type = trt.ActivationType.CLIP

    def clip_dyn_range_fn(dyn_range):
        def clip_fn(x):
            return max(alpha, min(beta, x))

        return clip_fn(dyn_range[0]), clip_fn(dyn_range[1])

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        alpha=alpha,
        beta=beta,
        dyn_range_fn=clip_dyn_range_fn,
    )


def hard_sigmoid(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    alpha: Optional[Any],
    beta: Optional[Any],
):
    operation_type = trt.ActivationType.HARD_SIGMOID

    def hard_sigmoid_dyn_range_fn(dyn_range):
        def hard_sigmoid_fn(x):
            return max(0, min(1, alpha * x + beta))

        return hard_sigmoid_fn(dyn_range[0]), hard_sigmoid_fn(dyn_range[1])

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        alpha=alpha,
        beta=beta,
        dyn_range_fn=hard_sigmoid_dyn_range_fn,
    )


# no corresponding function in aten/native_functions
def scaled_tanh(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    alpha: Optional[Any],
    beta: Optional[Any],
):
    operation_type = trt.ActivationType.SCALED_TANH

    def scaled_tanh_dyn_range_fn(dyn_range):
        def scaled_tanh_fn(x):
            return alpha * torch.nn.functional.tanh(beta * x)

        return scaled_tanh_fn(dyn_range[0]), scaled_tanh_fn(dyn_range[1])

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        alpha=alpha,
        beta=beta,
        dyn_range_fn=scaled_tanh_dyn_range_fn,
    )


# no corresponding function in aten/native_functions
def thresholded_relu(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    alpha: Optional[Any],
):
    operation_type = trt.ActivationType.THRESHOLDED_RELU

    def thresholded_relu_dyn_range_fn(dyn_range):
        def thresholded_relu_fn(x):
            return x if x > alpha else 0

        return thresholded_relu_fn(dyn_range[0]), thresholded_relu_fn(dyn_range[1])

    return convert_activation(
        network,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        alpha=alpha,
        dyn_range_fn=thresholded_relu_dyn_range_fn,
    )
