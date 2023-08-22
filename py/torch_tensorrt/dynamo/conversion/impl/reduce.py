from typing import Optional, Tuple, Union

import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_axes_for_reduce_op,
)
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor


def amax(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    dim: Union[int, Tuple[int]],
    keepdim: bool = False,
) -> TRTTensor:
    if (isinstance(input_val, TRTTensor)) and (
        input_val.dtype == trt.int8 or input_val.dtype == trt.int32
    ):
        input_val = cast_trt_tensor(network, input_val, trt.float32, name)

    if dim is None:
        raise ValueError("amax requires specifying dimension(s) (dim).")

    layer = network.add_reduce(
        input_val,
        trt.ReduceOperation.MAX,
        axes=get_axes_for_reduce_op(dim),
        keep_dims=keepdim,
    )
    set_layer_name(layer, target, name)
    return layer.get_output(0)
