from typing import Optional, Union, cast, Any, Tuple

import tensorrt as trt

from torch.fx.node import Target

from torch_tensorrt.fx.types import TRTNetwork, TRTTensor
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.fx.converters.converter_utils import (
    get_axes_for_reduce_op,
    set_layer_name,
)

def amax(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Union[int, Tuple[int]],
    keep_dims: Optional[bool] = False,
    out: Optional[Any] = None
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(f"amax received input {input} that is not part of the TensorRT region!"
    )

    if dim is None:
        raise ValueError("amax requires specifying dimension(s) (dim).")

    layer = network.add_reduce(
        input, 
        trt.ReduceOperation.MAX, 
        axes=get_axes_for_reduce_op(dim, network.has_implicit_batch_dimension),
        keep_dims=keep_dims
    )
    set_layer_name(layer, target, name)
    
    layer_out = layer.get_output(0)
    if out is not None:
        if out.shape != layer_out.shape:
            raise RuntimeError(f"The shape of argument `out` is {out.shape}, which is not same as the layer output shape {layer_out.shape}!")
        else:
            out = layer_out

    return layer_out
