from typing import Optional

from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTTensor


def index_select(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
    index: TRTTensor,
) -> TRTTensor:
    input_tensor = get_trt_tensor(ctx, input, f"{name}_input")
    index_tensor = get_trt_tensor(ctx, index, f"{name}_index")

    # The axis parameter specifies the dimension along which to index.
    gather_layer = ctx.net.add_gather(input_tensor, index_tensor, axis=dim)

    set_layer_name(gather_layer, target, f"{name}_gather", source_ir)

    return gather_layer.get_output(0)
