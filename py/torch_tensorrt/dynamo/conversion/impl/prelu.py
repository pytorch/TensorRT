from typing import Optional

from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import set_layer_name
from torch_tensorrt.dynamo.types import TRTTensor


def prelu(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: TRTTensor,
) -> TRTTensor:
    # TRT requires that the slopes tensor must be unidirectional broadcastable to the input tensor:
    # the rank of the two tensors must be the same, and all dimensions of the slopes tensor must
    # either equal the input tensor or be 1. The output tensor has the same shape as the input tensor.
    input, weight = impl.elementwise.broadcast(
        ctx, input, weight, f"{name}_broadcast_input", f"{name}_broadcast_weight"
    )
    layer = ctx.net.add_parametric_relu(input, slopes=weight)
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)
