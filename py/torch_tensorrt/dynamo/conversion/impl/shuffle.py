from typing import List, Optional, Union

from torch.fx.node import Target
from torch_tensorrt.dynamo.conversion.converter_utils import SourceIR
from torch_tensorrt.fx.converters.converter_utils import (
    get_positive_dim,
    set_layer_name,
)
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor


def reshape(
    network: TRTNetwork,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    shape: List[int],
) -> TRTTensor:
    layer = network.add_shuffle(input)
    layer.reshape_dims = tuple(shape)
    set_layer_name(layer, target, f"{name}_reshape", source_ir)
    return layer.get_output(0)


def flatten(
    network: TRTNetwork,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    start_dim: int,
    end_dim: int,
) -> TRTTensor:
    shape = input.shape
    dim_size = len(shape)
    start_dim = get_positive_dim(start_dim, dim_size)
    end_dim = get_positive_dim(end_dim, dim_size)

    num_elements = 1
    for i in range(start_dim, end_dim + 1):
        num_elements *= shape[i]

    new_shape = (
        tuple(shape[:start_dim])
        + (num_elements,)
        + (tuple(shape[end_dim + 1 :]) if end_dim + 1 < dim_size else tuple())
    )
    layer = network.add_shuffle(input)
    layer.reshape_dims = new_shape
    set_layer_name(layer, target, f"{name}_flatten", source_ir)
    return layer.get_output(0)
