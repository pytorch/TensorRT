# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Union

import numpy as np
import tensorrt as trt
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    broadcast,
    get_transposed_trt_tensor,
    get_trt_tensor,
)


def linear(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: Union[TRTTensor, torch.Tensor, np.ndarray],
    bias: Optional[Union[TRTTensor, torch.Tensor, np.ndarray]],
) -> TRTTensor:
    # Process weight terms
    weight_matrix_op = trt.MatrixOperation.TRANSPOSE
    if not isinstance(weight, (TRTTensor, torch.Tensor, np.ndarray)):
        raise RuntimeError(
            f"Linear layer {name} has weight of type {type(weight)}, Expect Union[TRTTensor, torch.Tensor, np.ndarray],"
        )
    elif isinstance(weight, (torch.Tensor, np.ndarray)) and _transpose_once(ctx):
        # TensorRT can keep a host copy of a weight for every layer that reads it
        # transposed, so a weight that several linear layers read is transposed
        # once for all of them.
        weight = get_transposed_trt_tensor(
            ctx, weight, f"{name}_weight", len(input.shape)
        )
        weight_matrix_op = trt.MatrixOperation.NONE
    elif isinstance(weight, (torch.Tensor, np.ndarray)):
        weight = get_trt_tensor(ctx, weight, f"{name}_weight")

    # Process bias terms
    if bias is not None and not isinstance(bias, (TRTTensor, torch.Tensor, np.ndarray)):
        raise RuntimeError(
            f"Linear layer {name} has bias of type {type(bias)}, Expect Union[TRTTensor, torch.Tensor, np.ndarray],"
        )
    elif isinstance(bias, (torch.Tensor, np.ndarray)):
        bias = get_trt_tensor(ctx, bias, f"{name}_bias")

    # add IMatrixMultiplyLayer
    out = impl.matmul.matrix_multiply(
        ctx,
        target,
        source_ir,
        f"{name}_matrix_multiply",
        input,
        weight,
        input_matrix_op=trt.MatrixOperation.NONE,
        other_matrix_op=weight_matrix_op,
    )

    if bias is not None:
        # Rank-align a 1-D linear bias with the matmul output while preserving
        # singleton dimensions. TensorRT can broadcast those dimensions in the
        # elementwise add without materializing the bias to the full output
        # shape (for example, [O] -> [1, 1, O] for a [B, S, O] output).
        out, bias = broadcast(
            ctx,
            out,
            bias,
            f"{name}_add_bias_lhs",
            f"{name}_add_bias_rhs",
        )

        # add bias
        out = impl.elementwise.add(
            ctx, target, source_ir, f"{name}_add_bias", out, bias
        )

    return out


def _transpose_once(ctx: ConversionContext) -> bool:
    """Whether to transpose the weight of the linear node being converted once for
    all the linear nodes that read it (see the deduplicate_weights setting)."""
    node = ctx.current_node
    if (
        not ctx.compilation_settings.deduplicate_weights
        or node is None
        or len(node.args) < 2
        or not isinstance(node.args[1], torch.fx.Node)
    ):
        return False
    readers = [user for user in node.args[1].users if user.target == node.target]
    return len(readers) > 1
