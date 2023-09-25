from typing import Optional, Union, Sequence, Dict

import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor

def cat(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTNetwork,
    dim: int,
    
) -> Union[TRTTensor, Sequence[TRTTensor]]:

    if any(not isinstance(t, TRTTensor) for t in input):  # type: ignore[union-attr]
        raise RuntimeError(
            f"cat received inputs {input} that is not part " "of the TensorRT region!"
        )
    concat_layer = network.add_concatenation(input)
    if dim < 0:
        dim = len(input[0].shape) + dim

    concat_layer.axis = dim
    set_layer_name(concat_layer, target, name + "_gather", source_ir)
    return concat_layer.get_output(0)