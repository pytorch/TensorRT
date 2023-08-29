from typing import Optional, Sequence

from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.fx.converters.converter_utils import (
    get_positive_dim,
    set_layer_name,
)
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor


def permute(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    permutation: Sequence[int],
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"permute received input {input} that is not a TensorRT ITensor"
        )

    permutation = [get_positive_dim(i, len(input.shape)) for i in permutation]

    layer = network.add_shuffle(input)
    layer.second_transpose = tuple(permutation)
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)
