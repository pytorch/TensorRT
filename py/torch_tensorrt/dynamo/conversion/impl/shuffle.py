from typing import Optional, Sequence, Union

from torch.fx.node import Target
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import SourceIR
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTTensor


def reshape(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    shape: Sequence[int],
) -> TRTTensor:
    layer = ctx.net.add_shuffle(input)
    layer.reshape_dims = tuple(shape)
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)
