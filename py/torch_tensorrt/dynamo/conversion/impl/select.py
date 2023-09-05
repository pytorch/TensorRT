from typing import Optional, Sequence, Union, cast

import numpy as np
import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.impl.elementwise import convert_binary_elementwise
from torch_tensorrt.dynamo.conversion.impl.shape import get_shape_with_dynamic_shape
from torch_tensorrt.fx.converters.converter_utils import (
    get_positive_dim,
    get_trt_tensor,
    has_dynamic_shape,
    set_layer_name,
    to_numpy,
)
from torch_tensorrt.fx.types import Shape, TRTNetwork, TRTTensor


def select(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Shape,
    index: Shape,
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"slice_tensor received input {input} that is not part "
            "of the TensorRT region!"
        )

    ranks = len(input.shape) + (1 if ctx.net.has_implicit_batch_dimension else 0)
    dim = get_positive_dim(cast(int, dim), ranks)
    dynamic_shape = has_dynamic_shape(input.shape)
    if ctx.net.has_implicit_batch_dimension:
        if dim == 0:
            raise RuntimeError(
                f"We do not support slice_tensor at batch dim when it's implicit, got {dim}!"
            )
        dim = dim - 1
    else:
        if dynamic_shape:
            # Check whether slice target dim is dynamic shape dim
            assert input.shape[dim] != -1, "Can't select on negative shape dimension!"
    index = index

    if index >= input.shape[dim]:
        raise RuntimeError(
            f"cannot have index greater than the dimension length! {input.shape[dim]}"
        )
    output_shape = list(input.shape)
    output_shape[dim] = 1
    if dynamic_shape > 0:
        output_shape = get_shape_with_dynamic_shape(
            ctx, target, source_ir, name, output_shape, input
        )
    index_value = np.array(index, dtype=np.int32)
    indices_tensor = ctx.net.add_constant(
        index_value.shape, to_numpy(index_value)
    ).get_output(0)
    layer = ctx.net.add_gather(input, indices_tensor, dim)
    out = layer.get_output(0)
    if len(out.shape) != 1:
        layer = ctx.net.add_shuffle(out)
    return layer.get_output(0)


def index(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    index: Union[TRTTensor, Sequence[TRTTensor]],
) -> TRTTensor:
    adv_indx_indices = []
    tensor_indices = []

    for i in len(index):
        ind = index[i]
        # FIXME: check if the datatype for the indices needs to be casted to INT32
        # TRTInterpretor should take care
        adv_indx_indices.append(i)
        tensor_indices.append(ind)

    if not tensor_indices:
        identity_layer = network.add_identity(input)
        identity_layer.set_output_type(0, trt.int32)
        set_layer_name(identity_layer, target, name + "_index_identity", source_ir)
        return identity_layer.get_output(0)
    elif len(tensor_indices) == 1:
        indices_tensor = tensor_indices[0]
        gather_layer = network.add_gather(input, indices_tensor, adv_indx_indices[0])
        set_layer_name(gather_layer, target, name + "_index_gather", source_ir)
        return gather_layer.get_output(0)
    else:
        input_shape = input.shape
        rank = len(input_shape)
        adv_indx_count = len(adv_indx_indices)
        input_shape_layer = network.add_shape(input)
        set_layer_name(input_shape_layer, target, name + "_index_shape", source_ir)
        input_shape_tensor = input_shape_layer.get_output(0)
        dim_tensor_list = []
        for i in range(rank):
            # check this
            dim_tensor_layer = network.add_gather(input_shape_tensor, i, 0)
            set_layer_name(
                input_shape_layer, target, name + "_index_gather_rank", source_ir
            )
            dim_tensor = dim_tensor_layer.get_output(0)
            dim_tensor_list.append(dim_tensor)

        # for cases like
        # t: [x_1, y_1, y_2, ..., x_m, ..., y_n] -> t: [x_1, x_2, ..., x_m, y_1, y_2, ..., y_n],
        # where t is a tensor of rank m+n, {x_i} are axes where tensor index is provided, and {y_i} are axes
        # for ":"
        # Examples: x.shape = (10,20,30,40,50)
        # ind_1, ind_2 broadcasted to (2,3,4)
        # x[:, ind_1, ind_2] = 10, 2, 3, 4, 40, 50
        # x[:,ind_1, :, ind_2] = 2, 3, 4, 10, 30, 50
        transpose_layer = network.add_shuffle(input)
        new_order = []
        for i in range(adv_indx_count):
            new_order.append(adv_indx_indices[i])
        for i in range(rank):
            if i not in adv_indx_indices:
                new_order.append(i)

        permute_order = trt.Permutation()
        permute_order(new_order)
        transpose_layer.set_second_transpose(permute_order)
        set_layer_name(transpose_layer, target, name + "_index_transpose", source_ir)
        transpose_tensor = transpose_layer.get_output(0)

        # Flatten [x_1, x_2,.......x_m, y_1, y_2,.....y_m]
        transpose_tensor_shape = network.add_shape(transpose_tensor)
        d0 = 1
        d0 = get_trt_tensor(network, d0, "d0_initial")
        for i in range(adv_indx_count):
            dim_tensor_layer = network.add_gather(transpose_tensor_shape, i, 0)
            set_layer_name(
                dim_tensor_layer, target, name + "_index_gather_concatOne", source_ir
            )
            d0_gather = gather_layer.get_output(0)
            mult_d0 = convert_binary_elementwise(
                network,
                target,
                source_ir,
                name + "index_concatOne_shape",
                trt.ElementWisePROD,
                mult_d0,
                d0_gather,
            )

        d1 = 1
        d1 = get_trt_tensor(network, d0, "d0_initial")
        for i in range(adv_indx_count, rank):
            dim_tensor_layer = network.add_gather(transpose_tensor_shape, i, 0)
            set_layer_name(
                dim_tensor_layer, target, name + "_index_gather_concatTwo", source_ir
            )
            d1_gather = gather_layer.get_output(0)
            mult_d1 = convert_binary_elementwise(
                network,
                target,
                source_ir,
                name + "index_concatTwo_shape",
                trt.ElementWisePROD,
                mult_d1,
                d1_gather,
            )
        concat_tensor_layer = network.add_concatenation([mult_d0, mult_d1])
        set_layer_name(concat_tensor_layer, target, name + "_index_Concat", source_ir)
        concat_tensor = concat_tensor_layer.get_output(0)

        reshape_layer = network.add_shuffle(transpose_tensor)
        # check this
        reshape_layer.set_input(1, concat_tensor)
        flatten_tensor = reshape_layer.get_output(0)

        # tensor index = \sum_{i=1}^m (ind_i * \prod_{j=i+1}^m (x_j)),  ind_i is input indices[i], x_j is the
        # // j dimension of input x.
        multiplier = get_trt_tensor(
            network, dim_tensor_list[adv_indx_indices[adv_indx_count - 1]], "dim_last"
        )
        cum_adv_index = tensor_indices[adv_indx_count - 1]
        for i in range(adv_indx_count - 2, 0):
            adv_index = convert_binary_elementwise(
                network,
                target,
                source_ir,
                name + "index_intermediate",
                trt.ElementWisePROD,
                multiplier,
                tensor_indices[i],
            )
            cum_adv_index = convert_binary_elementwise(
                network,
                target,
                source_ir,
                name + "index_sum_intermediate",
                trt.ElementWiseSUM,
                cum_adv_index,
                adv_index,
            )
            multiplier = convert_binary_elementwise(
                network,
                target,
                source_ir,
                name + "index_intermediate",
                trt.ElementWisePROD,
                multiplier,
                dim_tensor_list[adv_indx_count[i]],
            )

        gather_layer_element = network.add_gather(flatten_tensor, cum_adv_index, 0)
        set_layer_name(
            gather_layer_element, target, name + "_index_gather_element", source_ir
        )
        gather_out = gather_layer.get_output(0)

        cum_adv_index_shape_tensor = cum_adv_index.add_shape(cum_adv_index_shape_tensor)
        # check if all advanced indices are consecutive
        concat_tensor_reshape = []
        if (
            adv_indx_count
            == adv_indx_indices[adv_indx_count - 1] - adv_indx_indices[0] + 1
        ):
            # concat_tensor_reshape_initial = -1
            # concat_tensor_reshape_initial_tensor = get_trt_tensor(network, concat_tensor_reshape_initial, "concat_tensor_reshape_initial")
            concat_tensor_reshape.append(-1)
            for i in range(0, rank):
                if i not in adv_indx_indices:
                    curr_dim = dim_tensor_list[i]
                    concat_tensor_reshape.append(curr_dim)

            concat_tensor_layer = network.add_concatenation(concat_tensor_reshape)
            set_layer_name(
                concat_tensor_layer, target, name + "_index_Concat_reshape", source_ir
            )
            concat_tensor = concat_tensor_layer.get_output(0)

            regular_index_shuffle_layer = network.add_shuffle(gather_out)
            set_layer_name(
                regular_index_shuffle_layer,
                target,
                name + "_index_regular_index",
                source_ir,
            )
            unfold_tensor = regular_index_shuffle_layer.get_output(0)

            transpose_advanced_shuffle_layer = network.add_shuffle(unfold_tensor)
            new_order = []
            for i in range(1, adv_indx_count[0] + 1):
                new_order.append(i)
            new_order.append(0)
            for i in range(adv_indx_indices[0] + 1, rank - adv_indx_count):
                new_order.append(i)

            permute_order = trt.Permutation()
            permute_order(new_order)
            transpose_advanced_shuffle_layer.set_second_transpose(permute_order)
            set_layer_name(
                transpose_advanced_shuffle_layer,
                target,
                name + "_index_advanced_shuffle_transpose",
                source_ir,
            )
            transpose_tensor = transpose_advanced_shuffle_layer.get_output(0)

            # unfold advanced layer
            concat_final_tensor = []
            for i in range(0, adv_indx_indices[0]):
                current_dim = dim_tensor_list[i]
                concat_final_tensor.push_back(curr_dim)

            concat_final_tensor.push_back(cum_adv_index_shape_tensor)
            for i in range(adv_indx_indices[0], rank):
                if i not in (adv_indx_indices):
                    current_dim = dim_tensor_list[i]
                    concat_final_tensor.append(current_dim)

            concat_final_shape_layer = network.add_concatenation(concat_final_tensor)
            set_layer_name(
                concat_final_shape_layer,
                target,
                name + "_index_concat_final_shape_layer",
                source_ir,
            )
            concat_final_tensor = concat_final_shape_layer.get_output(0)

            unfold_advanced_shuffle_layer = network.add_shuffle(transpose_tensor)
            # check this
            reshape_layer.set_input(1, concat_final_tensor)
            reshape_output = reshape_layer.get_output(0)

        else:
            concat_tensor = []
            for i in range(0, rank):
                if i not in adv_indx_indices:
                    curr_dim = dim_tensor_list[i]
                    concat_tensor.append(curr_dim)

            concat_layer = network.add_concatenation(concat_tensor)
            set_layer_name(
                concat_layer,
                target,
                name + "_index_concat_final_shape_layer",
                source_ir,
            )
            concat_final_tensor = concat_final_shape_layer.get_output(0)

            reshape_layer = network.add_shuffle(gather_out)
            reshape_layer.setInput(1, concat_final_tensor)
            set_layer_name(
                reshape_layer,
                target,
                name + "_index_shuffle_final_shape_layer",
                source_ir,
            )
            reshape_output = reshape_layer.get_output(0)

    return reshape_output
