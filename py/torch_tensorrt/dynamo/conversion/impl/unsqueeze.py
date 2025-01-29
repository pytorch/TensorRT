from typing import List, Optional, Sequence

from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_trt_tensor,
    set_layer_name,
)
from torch_tensorrt.dynamo.types import TRTTensor


def unsqueeze(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
) -> TRTTensor:
    axes = get_trt_tensor(ctx, dim, f"{name}_axes")
    layer = ctx.net.add_unsqueeze(input, axes)
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def broadcast_in_dim(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_t: TRTTensor,
    shape: Sequence[int],
    broadcast_dimensions: Sequence[int],
) -> TRTTensor:
    augmented_shape_list: List[Optional[int]] = list(shape)

    # For each dimension being broadcasted, set the augmented shape to None
    for broadcast_dim in broadcast_dimensions:
        augmented_shape_list[broadcast_dim] = None

    # TODO: Expand support to arbitrary broadcasts
    assert all(
        dim in (1, None) for dim in augmented_shape_list
    ), "broadcast_in_dim currently only supports unsqueeze broadcasting"

    # Unsqueeze the shape repeatedly to broadcast
    output = input_t
    for idx, x in enumerate(augmented_shape_list):
        # If the value is not None, that dimension is to be broadcasted
        if x is not None:
            output = unsqueeze(
                ctx,
                target,
                source_ir,
                name + f"_unsqueeze_for_broadcast_{idx}",
                output,
                idx,
            )

    assert tuple(output.shape) == tuple(shape), "broadcast_in_dim shapes don't match"

    return output
