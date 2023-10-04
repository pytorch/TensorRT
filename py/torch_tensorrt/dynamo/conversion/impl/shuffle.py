from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import SourceIR, get_trt_tensor
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTTensor


def reshape(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: Sequence[Union[TRTTensor, torch.Tensor, np.ndarray]],
    shape: Sequence[int],
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        input = get_trt_tensor(ctx, input, f"{name}_input")
    layer = ctx.net.add_shuffle(input)
    layer.reshape_dims = tuple(shape)
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)
