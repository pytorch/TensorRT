from typing import Dict, Optional, Sequence, Union

import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor


def cat(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: Union[TRTTensor, Sequence[TRTTensor]],
    dim: int,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    for each_input in input:
        if(not isinstance(each_input, TRTTensor)):
            each_input = get_trt_tensor(each_input)
    concat_layer = ctx.net.add_concatenation(input)
    if dim < 0:
        dim = len(input[0].shape) + dim

    concat_layer.axis = dim
    set_layer_name(concat_layer, target, name + "_gather", source_ir)
    return concat_layer.get_output(0)
