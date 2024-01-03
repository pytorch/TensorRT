import logging
from typing import Optional, Sequence, Union, cast

import numpy as np
import tensorrt as trt
import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    broadcastable,
    get_positive_dim,
    get_trt_tensor,
    to_numpy,
)
from torch_tensorrt.dynamo.conversion.impl.elementwise import convert_binary_elementwise
from torch_tensorrt.dynamo.conversion.impl.shape import get_shape_with_dynamic_shape
from torch_tensorrt.fx.converters.converter_utils import (
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.fx.types import Shape, TRTTensor

_LOGGER: logging.Logger = logging.getLogger(__name__)


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
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    index: Sequence[Union[TRTTensor, np.ndarray, torch.Tensor]],
) -> TRTTensor:
    adv_indx_indices = []
    tensor_indices = []
    # check if the input is dynamic
    dynamic_shape = has_dynamic_shape(input.shape)
    # is_numpy is a flag to specify if all the indices are numpy or torchTensor.
    # If any is not this flag will be set to False
    _LOGGER.debug(
        "Determining whether aten.index constant-index optimization can be invoked"
    )
    is_numpy = all(
        isinstance(ind, (torch.Tensor, np.ndarray)) for ind in index if ind is not None
    )
    # here we need to check if all the index are broadcastable
    # if no, then we need to broadcast
    last_index = None
    for i, ind in enumerate(index):
        if ind is not None:
            _LOGGER.debug(f"Shape of {i} index is {ind.shape}")
            adv_indx_indices.append(i)
            # torch.nn.parameter.Parameter=> numpy array
            # numpy array is kept as numpy
            # other cases are kept as TRTTensor
            if is_numpy:
                ind = to_numpy(ind)
            else:
                ind = get_trt_tensor(ctx, ind, name + f"_parameter_to_fp32_tensor_{i}")
            if last_index is not None:
                assert broadcastable(
                    ind, last_index
                ), "The indices should be broadcastable!"
            last_index = ind
            tensor_indices.append(ind)

    if not tensor_indices:
        identity_layer = ctx.net.add_identity(input)
        identity_layer.set_output_type(0, trt.int32)
        set_layer_name(identity_layer, target, name + "_index_identity", source_ir)
        return identity_layer.get_output(0)
    elif len(tensor_indices) == 1:
        indices_tensor = get_trt_tensor(
            ctx, tensor_indices[0], name + "_parameter_to_fp32_tensor"
        )
        index = adv_indx_indices[0]
        _LOGGER.debug(f"The advanced index indices is {adv_indx_indices}")
        gather_layer = ctx.net.add_gather(input, indices_tensor, index)
        set_layer_name(gather_layer, target, name + "_index_gather", source_ir)
        return gather_layer.get_output(0)
    else:
        input_shape = input.shape
        _LOGGER.debug(f"The input shape is {input.shape}")
        if dynamic_shape:
            input_shape = get_shape_with_dynamic_shape(
                ctx.net, target, source_ir, name, input_shape, input
            )
        rank = len(input_shape)
        adv_indx_count = len(adv_indx_indices)
        dim_tensor_list = []

        for i in range(rank):
            dim = input_shape[i]
            dim_tensor = get_trt_tensor(ctx, dim, name + f"_individual_dim_{i}")
            # dim_tensor_list is a list of tensors
            dim_tensor_list.append(dim_tensor)

        # for cases like
        # t: [x_1, y_1, y_2, ..., x_m, ..., y_n] -> t: [x_1, x_2, ..., x_m, y_1, y_2, ..., y_n],
        # where t is a tensor of rank m+n, {x_i} are axes where tensor index is provided, and {y_i} are axes
        # for ":"
        # Examples: x.shape = (10,20,30,40,50)
        # ind_1, ind_2 broadcasted to (2,3,4)
        # x[:, ind_1, ind_2] = 10, 2, 3, 4, 40, 50
        # x[:,ind_1, :, ind_2] = 2, 3, 4, 10, 30, 50
        transpose_layer = ctx.net.add_shuffle(input)
        new_order = []
        for i in range(adv_indx_count):
            new_order.append(adv_indx_indices[i])
        for i in range(rank):
            if i not in adv_indx_indices:
                new_order.append(i)
        _LOGGER.debug(f"The new transpose order is {new_order}")

        transpose_layer.second_transpose = tuple(new_order)
        set_layer_name(transpose_layer, target, name + "_index_transpose", source_ir)
        transpose_tensor = transpose_layer.get_output(0)

        # Flatten [x_1, x_2,.......x_m, y_1, y_2,.....y_n]
        # transpose_tensor_shape = ctx.net.add_shape(transpose_tensor)
        transpose_tensor_shape = transpose_tensor.shape
        _LOGGER.debug(f"The shape of transpose tensor is {transpose_tensor_shape}")
        mult_d0 = 1
        for i in range(adv_indx_count):
            mult_d0 = mult_d0 * transpose_tensor_shape[i]
        mult_d1 = 1
        for i in range(adv_indx_count, rank):
            mult_d1 = mult_d1 * transpose_tensor_shape[i]

        concat_tensor_layer = ctx.net.add_concatenation(
            [
                get_trt_tensor(ctx, mult_d0, name + "_d0_shape"),
                get_trt_tensor(ctx, mult_d1, name + "_d1_shape"),
            ]
        )
        set_layer_name(concat_tensor_layer, target, name + "_index_Concat", source_ir)
        concat_tensor = concat_tensor_layer.get_output(0)

        reshape_layer = ctx.net.add_shuffle(transpose_tensor)
        reshape_layer.set_input(1, concat_tensor)
        flatten_tensor = reshape_layer.get_output(0)

        _LOGGER.debug(f"The flatten tensor shape is {flatten_tensor.shape}")

        # tensor index = \sum_{i=1}^m (ind_i * \prod_{j=i+1}^m (x_j)),  ind_i is input indices[i], x_j is the
        # // j dimension of input x.
        if is_numpy:
            multiplier = input_shape[adv_indx_indices[adv_indx_count - 1]]
            cum_adv_index = tensor_indices[adv_indx_count - 1]
            for i in range(adv_indx_count - 2, -1, -1):
                adv_index = multiplier * tensor_indices[i]
                cum_adv_index = cum_adv_index + adv_index
                multiplier = multiplier * input_shape[adv_indx_indices[i]]
            cum_adv_index = get_trt_tensor(
                ctx, cum_adv_index, name + "_index_sum_intermediate"
            )
        else:
            multiplier = get_trt_tensor(
                ctx,
                dim_tensor_list[adv_indx_indices[adv_indx_count - 1]],
                name + "_dim_last",
            )
            cum_adv_index = tensor_indices[adv_indx_count - 1]
            for i in range(adv_indx_count - 2, -1, -1):
                adv_index = convert_binary_elementwise(
                    ctx,
                    target,
                    source_ir,
                    name + f"_index_intermediate_{i}",
                    trt.ElementWiseOperation.PROD,
                    multiplier,
                    tensor_indices[i],
                )
                cum_adv_index = convert_binary_elementwise(
                    ctx,
                    target,
                    source_ir,
                    name + f"_index_sum_intermediate_{i}",
                    trt.ElementWiseOperation.SUM,
                    cum_adv_index,
                    adv_index,
                )
                multiplier = convert_binary_elementwise(
                    ctx,
                    target,
                    source_ir,
                    name + f"_index_intermediate_xj_{i}",
                    trt.ElementWiseOperation.PROD,
                    multiplier,
                    dim_tensor_list[adv_indx_indices[i]],
                )

        gather_layer_element = ctx.net.add_gather(flatten_tensor, cum_adv_index, 0)
        set_layer_name(
            gather_layer_element, target, name + "_index_gather_element", source_ir
        )
        gather_out = gather_layer_element.get_output(0)
        _LOGGER.debug(f"The shape after cumultative gather is {gather_out.shape}")
        _LOGGER.debug(f"The shape for cumulative adv index is {cum_adv_index}")

        cum_adv_index_shape_layer = ctx.net.add_shape(cum_adv_index)
        set_layer_name(
            cum_adv_index_shape_layer, target, name + "_cum_adv_index_shape", source_ir
        )
        cum_adv_index_shape_tensor = cum_adv_index_shape_layer.get_output(0)
        cum_adv_index_shape = cum_adv_index.shape
        _LOGGER.debug(f"The shape for cumulative adv index is {cum_adv_index_shape}")
        # check if all advanced indices are consecutive
        concat_tensor_reshape = []
        if (
            adv_indx_count
            == adv_indx_indices[adv_indx_count - 1] - adv_indx_indices[0] + 1
        ):
            _LOGGER.debug("The indices are continuous in this case")
            concat_tensor_reshape.append(
                get_trt_tensor(ctx, -1, name + "_dynamic_concat")
            )
            for i in range(0, rank):
                if i not in adv_indx_indices:
                    curr_dim = dim_tensor_list[i]
                    concat_tensor_reshape.append(curr_dim)

            concat_tensor_layer = ctx.net.add_concatenation(concat_tensor_reshape)
            set_layer_name(
                concat_tensor_layer, target, name + "_index_Concat_reshape", source_ir
            )
            concat_tensor = concat_tensor_layer.get_output(0)

            regular_index_shuffle_layer = ctx.net.add_shuffle(gather_out)
            regular_index_shuffle_layer.set_input(1, concat_tensor)
            set_layer_name(
                regular_index_shuffle_layer,
                target,
                name + "_index_regular_index",
                source_ir,
            )
            unfold_tensor = regular_index_shuffle_layer.get_output(0)
            _LOGGER.debug("The tensor is unfolded now")
            _LOGGER.debug(f"The unfolded tensor shape is {unfold_tensor.shape}")

            # Transpose folded advanced indexed axis to its original location.
            transpose_advanced_shuffle_layer = ctx.net.add_shuffle(unfold_tensor)
            new_order = []
            for i in range(1, adv_indx_indices[0] + 1):
                new_order.append(i)
            new_order.append(0)
            for i in range(adv_indx_indices[0] + 1, rank - adv_indx_count + 1):
                new_order.append(i)
            _LOGGER.debug(f"Transposing the indices to correct position {new_order}")

            transpose_advanced_shuffle_layer.second_transpose = tuple(new_order)
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
                concat_final_tensor.append(current_dim)

            concat_final_tensor.append(cum_adv_index_shape_tensor)
            for i in range(adv_indx_indices[0], rank):
                if i not in (adv_indx_indices):
                    current_dim = dim_tensor_list[i]
                    concat_final_tensor.append(current_dim)

            concat_final_shape_layer = ctx.net.add_concatenation(concat_final_tensor)
            set_layer_name(
                concat_final_shape_layer,
                target,
                name + "_index_continuous_concat_final_shape_layer",
                source_ir,
            )
            concat_final_tensor = concat_final_shape_layer.get_output(0)

            unfold_advanced_shuffle_layer = ctx.net.add_shuffle(transpose_tensor)
            # check this
            unfold_advanced_shuffle_layer.set_input(1, concat_final_tensor)
            set_layer_name(
                unfold_advanced_shuffle_layer,
                target,
                name + "_unfold_advanced_index",
                source_ir,
            )
            reshape_output = unfold_advanced_shuffle_layer.get_output(0)

        else:
            _LOGGER.debug("The indices are not continuous in this case")
            concat_final_tensor = []
            concat_final_tensor.append(cum_adv_index_shape_tensor)
            for i in range(0, rank):
                if i not in adv_indx_indices:
                    curr_dim = dim_tensor_list[i]
                    concat_final_tensor.append(curr_dim)

            concat_final_shape_layer = ctx.net.add_concatenation(concat_final_tensor)
            set_layer_name(
                concat_final_shape_layer,
                target,
                name + "_index_non_continuous_concat_final_shape_layer",
                source_ir,
            )
            concat_final_tensor = concat_final_shape_layer.get_output(0)

            reshape_layer = ctx.net.add_shuffle(gather_out)
            reshape_layer.set_input(1, concat_final_tensor)
            set_layer_name(
                reshape_layer,
                target,
                name + "_index_non_continuous_shuffle_final_shape_layer",
                source_ir,
            )
            reshape_output = reshape_layer.get_output(0)

    return reshape_output
