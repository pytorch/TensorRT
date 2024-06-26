from typing import Optional, Sequence, Union

import numpy as np
import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_positive_dim,
    set_layer_name,
)
from torch_tensorrt.dynamo.conversion.impl.elementwise import ne
from torch_tensorrt.fx.types import TRTTensor


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
    dim_has_dynamic_shape = False
    for dim in dims:
        dim = get_positive_dim(
            dim,
            len(input.shape),
        )

        if input.shape[dim] == -1:
            dim_has_dynamic_shape = True
        new_dims.append(dim)

    layer = ctx.net.add_shuffle(input)
    set_layer_name(layer, target, name, source_ir)
    if dim_has_dynamic_shape:
        num_shape = len(input.shape)

        tensor_shape_layer = ctx.net.add_shape(input)
        tensor_shape = tensor_shape_layer.get_output(0)
        tensor_shape = cast_trt_tensor(
            ctx, tensor_shape, trt.int32, name + "shape_casted", "shape"
        )

        # change it to get_trt_tensor
        one_layer = ctx.net.add_constant(
            (num_shape,),
            np.ascontiguousarray([1] * num_shape, np.int32),
        )
        set_layer_name(one_layer, target, name + "_one", source_ir)

        zero_layer = ctx.net.add_constant(
            (num_shape,),
            np.zeros((num_shape,), dtype=np.int32),
        )
        set_layer_name(zero_layer, target, name + "_zero", source_ir)

        # append last element value
        num_append = num_shape - len(new_dims)
        if num_append > 0:
            new_dims += [new_dims[-1]] * num_append

        index_value = np.array(new_dims, dtype=np.int32)
        index_layer = ctx.net.add_constant(index_value.shape, index_value)
        set_layer_name(index_layer, target, name + "_index", source_ir)

        scatter_layer = ctx.net.add_scatter(
            zero_layer.get_output(0),
            index_layer.get_output(0),
            one_layer.get_output(0),
            trt.ScatterMode.ELEMENT,
        )
        set_layer_name(scatter_layer, target, name + "_scatter", source_ir)

        #  [1, 2, 1, 3, 1]
        #  [0, 0, 1, 1, 1]
        #  [t, t, f, t, f]
        ne_tensor = ne(
            ctx,
            target,
            source_ir,
            name + "_ne",
            tensor_shape,
            scatter_layer.get_output(0),
        )

        #  [t, t, f, t, f] -> [0, 1, 3]
        non_zero_layer = ctx.net.add_non_zero(ne_tensor)
        set_layer_name(non_zero_layer, target, name + "_non_zero", source_ir)

        non_zero_shuffle_layer = ctx.net.add_shuffle(non_zero_layer.get_output(0))
        set_layer_name(non_zero_shuffle_layer, target, name + "_shuffle", source_ir)
        non_zero_shuffle_layer.second_transpose = (1, 0)

        #  (1,2,1,3,1) + [0, 1, 3 ,4] -> [1, 2, 3, 1]
        gather_layer = ctx.net.add_gather_v2(
            tensor_shape, non_zero_shuffle_layer.get_output(0), mode=trt.GatherMode.ND
        )
        set_layer_name(gather_layer, target, name + "_gather", source_ir)

        layer.set_input(1, gather_layer.get_output(0))
    else:
        output_shape = []
        for i, s in enumerate(input.shape):
            if (i in new_dims) and s == 1:
                continue
            output_shape.append(s)
        layer.reshape_dims = tuple(output_shape)
    return layer.get_output(0)
