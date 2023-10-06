from typing import Optional

import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_axes_for_reduce_op,
)
from torch_tensorrt.fx.converters.converter_utils import (
    get_positive_dim,
    set_layer_name,
)
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor

from . import squeeze


def argmax(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int = 0,
    keep_dim: bool = False,
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"argmax received input {input} that is not part " "of the TensorRT region!"
        )
    if input.dtype == trt.int32:
        input = cast_trt_tensor(network, input, trt.float32, name)
    if dim < 0:
        dim = len(tuple(input.shape)) + dim
    reduce_mask = get_axes_for_reduce_op(get_positive_dim(dim, len(input.shape)))
    topk_layer = network.add_topk(input, trt.TopKOperation.MAX, 1, reduce_mask)
    set_layer_name(topk_layer, target, name)

    out = topk_layer.get_output(1)

    if not keep_dim:
        out = squeeze.squeeze(
            network, target, SourceIR.ATEN, name + "_squeeze", out, dim
        )

    return out
