from typing import Optional, Sequence, Union

from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import get_positive_dim
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTTensor
from torch_tensorrt.fx.utils import get_dynamic_dims


def squeeze(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Optional[Union[int, Sequence[int]]] = None,
) -> TRTTensor:
    # Squeeze with dim=None would only work in explicit batch dim mode without any dynamic
    # dim, which is a very rare case. For now we just claim not supporting dim=None.
    assert dim is not None, "We don't support dim=None right now for squeeze."
    dims = []

    if isinstance(dim, int):
        dims.append(dim)
    else:
        for dim in dim:
            dims.append(dim)

    new_dims = []
    for dim in dims:
        dim = get_positive_dim(
            dim,
            len(input.shape) + (1 if ctx.net.has_implicit_batch_dimension else 0),
        )
        if ctx.net.has_implicit_batch_dimension:
            assert dim != 0, "We don't support squeeze batch dim when it's implicit."
            dim -= 1

        assert input.shape[dim] != -1, "We don't support squeeze dynamic dim."
        assert (
            len(get_dynamic_dims(input.shape)) <= 1
        ), "Currently more than one dynamic dim for input to squeeze is not supported."
        new_dims.append(dim)

    output_shape = []
    for i, s in enumerate(input.shape):
        if (i in new_dims) and s == 1:
            continue
        output_shape.append(s)
    layer = ctx.net.add_shuffle(input)
    layer.reshape_dims = tuple(output_shape)
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)
