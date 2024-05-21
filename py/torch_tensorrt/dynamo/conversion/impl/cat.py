from typing import Optional, Sequence, Union

import numpy as np
import tensorrt as trt
import torch
from torch.fx.node import Target
from torch_tensorrt import _enums
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_positive_dim,
    get_trt_tensor,
)
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTTensor


def cat(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: Sequence[Union[TRTTensor, torch.Tensor, np.ndarray]],
    dim: int,
    cast_dtype: Union[_enums.dtype, trt.DataType, np.dtype] = None,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    trt_inputs = []
    for i, each_input in enumerate(input):
        if not isinstance(each_input, TRTTensor):
            each_input = get_trt_tensor(ctx, each_input, f"{name}_tensor_{i}")
        if cast_dtype:
            each_input = cast_trt_tensor(
                ctx, each_input, cast_dtype, f"{name}_tensor_int32_cast_{i}"
            )
        trt_inputs.append(each_input)
    concat_layer = ctx.net.add_concatenation(trt_inputs)
    dim = get_positive_dim(dim, len(trt_inputs[0].shape))
    concat_layer.axis = dim
    set_layer_name(concat_layer, target, f"{name}_gather", source_ir)
    return concat_layer.get_output(0)
