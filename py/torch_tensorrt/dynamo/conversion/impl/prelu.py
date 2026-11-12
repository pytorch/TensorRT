# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    prepend_ones,
    set_layer_name,
)


def prelu(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: TRTTensor,
) -> TRTTensor:
    # TensorRT broadcasts only the slope to the input, but PyTorch broadcasts both, as
    # for a 0-d input with a slope of shape (1,).
    if len(weight.shape) > len(input.shape):
        input = prepend_ones(
            ctx, input, f"{name}_broadcast", len(weight.shape) - len(input.shape)
        )
    layer = ctx.net.add_parametric_relu(input, weight)
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)
