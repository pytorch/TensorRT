from typing import List, Optional, Sequence, cast

from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_positive_dim,
    get_trt_tensor,
)
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import Shape, TRTTensor
from torch_tensorrt.fx.utils import get_dynamic_dims


def unsqueeze(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_t: TRTTensor,
    dim: Shape,
) -> TRTTensor:
    input_val = get_trt_tensor(ctx, input_t, f"{name}_input_t")
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"unsqueeze received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    dim = cast(int, dim)

    input_shape_size = (
        len(input_val.shape) + 1
        if ctx.net.has_implicit_batch_dimension
        else len(input_val.shape)
    )
    dim = get_positive_dim(dim, input_shape_size + 1)

    if ctx.net.has_implicit_batch_dimension:
        assert dim != 0
        dim -= 1

    assert (
        len(get_dynamic_dims(input_val.shape)) <= 1
    ), "Currently we don't support unsqueeze with more than one dynamic dims."
    layer = ctx.net.add_shuffle(input_val)
    layer.reshape_dims = (
        tuple(input_val.shape)[:dim] + (1,) + tuple(input_val.shape)[dim:]
    )
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
