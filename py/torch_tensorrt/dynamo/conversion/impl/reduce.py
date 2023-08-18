from typing import Any, Optional, Tuple, Union

import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion.converter_utils import get_axes_for_reduce_op
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor


def amax(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Union[int, Tuple[int]],
    keep_dims: Optional[bool] = False,
    out: Optional[Any] = None,
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"amax received input {input} that is not part of the TensorRT region!"
        )

    if dim is None:
        raise ValueError("amax requires specifying dimension(s) (dim).")

    layer = network.add_reduce(
        input,
        trt.ReduceOperation.MAX,
        axes=get_axes_for_reduce_op(dim),
        keep_dims=keep_dims,
    )
    set_layer_name(layer, target, name)
    return layer.get_output(0)
