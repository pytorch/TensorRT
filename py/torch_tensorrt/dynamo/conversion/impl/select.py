import logging
from typing import Optional, Sequence, Union

import numpy as np
import tensorrt as trt
import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    broadcastable,
    cast_trt_tensor,
    get_positive_dim,
    get_trt_tensor,
    to_numpy,
)
from torch_tensorrt.dynamo.conversion.impl.elementwise import convert_binary_elementwise
from torch_tensorrt.dynamo.conversion.impl.shape import shape as get_shape
from torch_tensorrt.dynamo.utils import DYNAMIC_DIM
from torch_tensorrt.fx.converters.converter_utils import (
    has_dynamic_shape,
    set_layer_name,
)
from torch_tensorrt.fx.types import TRTTensor

_LOGGER: logging.Logger = logging.getLogger(__name__)


def select(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
    index: int,
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"slice_tensor received input {input} that is not part "
            "of the TensorRT region!"
        )

    ranks = len(input.shape)
    dim = get_positive_dim(dim, ranks)

    indices_tensor = get_trt_tensor(
        ctx, np.array(index, dtype=np.int32), f"{name}_indices_tensor"
    )
    layer = ctx.net.add_gather(input, indices_tensor, dim)

    return layer.get_output(0)


def index(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    indices: Sequence[Union[TRTTensor, np.ndarray, torch.Tensor]],
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
        isinstance(ind, (torch.Tensor, np.ndarray))
        for ind in indices
        if ind is not None
    )
    # here we need to check if all the index are broadcastable
    # if no, then we need to broadcast
    last_index = None
    for i, ind in enumerate(indices):
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
        cast_layer = ctx.net.add_cast(input, trt.int32)
        set_layer_name(cast_layer, target, name + "_index_casted", source_ir)
        return cast_layer.get_output(0)
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
        rank = len(input_shape)
        adv_indx_count = len(adv_indx_indices)
        dim_tensor_list = []

        for i in range(rank):
            if input_shape[i] != DYNAMIC_DIM:
                dim = input_shape[i]
                dim_tensor = get_trt_tensor(ctx, dim, name + f"_individual_dim_{i}")
            else:
                dim_tensor = get_shape(
                    ctx, target, source_ir, name + f"_individual_dim_dyn_{i}", input, i
                )
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
        dim_tensor_shape_mult_d0 = 1
        for i in range(adv_indx_count):
            if transpose_tensor_shape[i] == DYNAMIC_DIM:
                dim_tensor_shape_mult_d0 = get_shape(
                    ctx,
                    target,
                    source_ir,
                    name + f"_transpose_tensor_shape_mult_d0_{i}",
                    transpose_tensor,
                    i,
                )
            else:
                dim_tensor_shape_mult_d0 = transpose_tensor_shape[i]
            mult_d0 = convert_binary_elementwise(
                ctx,
                target,
                source_ir,
                name + f"_shape_{i}",
                trt.ElementWiseOperation.PROD,
                mult_d0,
                dim_tensor_shape_mult_d0,
            )
        mult_d1 = 1
        dim_tensor_shape_mult_d1 = 1
        for i in range(adv_indx_count, rank):
            if transpose_tensor_shape[i] == DYNAMIC_DIM:
                dim_tensor_shape_mult_d1 = get_shape(
                    ctx,
                    target,
                    source_ir,
                    name + f"_transpose_tensor_shape_mult_d0_{i}",
                    transpose_tensor,
                    i,
                )
            else:
                dim_tensor_shape_mult_d1 = transpose_tensor_shape[i]
            mult_d1 = convert_binary_elementwise(
                ctx,
                target,
                source_ir,
                name + f"_shape_{i}",
                trt.ElementWiseOperation.PROD,
                mult_d1,
                dim_tensor_shape_mult_d1,
            )

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
            multiplier = dim_tensor_list[adv_indx_indices[adv_indx_count - 1]]
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
        cum_adv_index_shape_tensor = cast_trt_tensor(
            ctx,
            cum_adv_index_shape_tensor,
            trt.int32,
            name + "_cum_adv_index_shape_casted",
        )
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


def index_select(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
    index: TRTTensor,
) -> TRTTensor:
    # The axis parameter specifies the dimension along which to index.
    dim = get_positive_dim(dim, len(input.shape))
    gather_layer = ctx.net.add_gather(input, index, axis=dim)

    set_layer_name(gather_layer, target, f"{name}_gather_layer_default", source_ir)

    return gather_layer.get_output(0)


def scatter(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
    index: Union[TRTTensor, np.ndarray, torch.Tensor],
    src: Union[TRTTensor, int, float],
) -> TRTTensor:
    input_shape = input.shape
    index_shape = index.shape
    index_shape_list = list(index_shape)
    if index.dtype == trt.int64:
        index = cast_trt_tensor(ctx, index, trt.int32, name + "_cast_index_tensor")
    dim = get_positive_dim(dim, len(input_shape))
    src_tensor = src
    # scatter.value
    if isinstance(src, (int, float)):
        src_tensor = get_trt_tensor(
            ctx, src * np.ones(index_shape_list), name + "_value_tensor"
        )
        src_tensor = cast_trt_tensor(
            ctx, src_tensor, input.dtype, name + "_cast_value_tensor"
        )
    # scatter.src
    elif not (isinstance(src, TRTTensor)):
        src_tensor = get_trt_tensor(ctx, src, name + "_src_tensor")

    scatter_layer = ctx.net.add_scatter(
        input, index, src_tensor, trt.ScatterMode.ELEMENT
    )
    scatter_layer.axis = dim
    set_layer_name(scatter_layer, target, name + "_scatter_layer", source_ir)
    out = scatter_layer.get_output(0)
    return out


def gather(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int,
    index: Union[TRTTensor, np.ndarray, torch.Tensor],
) -> TRTTensor:
    input_shape = input.shape
    dim = get_positive_dim(dim, len(input_shape))
    index = cast_trt_tensor(ctx, index, trt.int32, name + "_cast_index_tensor")
    gather_layer = ctx.net.add_gather(input, index, axis=dim)
    gather_layer.mode = trt.GatherMode.ELEMENT
    set_layer_name(gather_layer, target, name + "_gather_layer_element", source_ir)
    out = gather_layer.get_output(0)
    return out


def index_put_converter(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_tensor: TRTTensor,
    indices: Sequence[Union[TRTTensor, np.ndarray, torch.Tensor]],
    values: TRTTensor,
    accumulate: bool = False,
) -> TRTTensor:
    # Reshape indices to add an extra dimension if necessary (indices is a Tuple of ITensors)
    reshaped_indices = []
    for i, each_input in enumerate(indices):
        if not isinstance(each_input, TRTTensor):
            each_input = get_trt_tensor(ctx, each_input, f"{name}_tensor_{i}")
        each_input = impl.shuffle.reshape(
            ctx,
            target,
            source_ir,
            f"{name}_reshape_{i}",
            each_input,
            (-1, 1),  # Reshape to (N, 1)
        )
        reshaped_indices.append(each_input)

    # Concatenate along the second dimension (columns)
    indices_cat = impl.cat.cat(
        ctx, target, source_ir, f"{name}_cat", reshaped_indices, dim=1
    )

    scatter_layer = ctx.net.add_scatter(
        input_tensor, indices_cat, values, trt.ScatterMode.ND
    )
    scatter_layer.axis = 0
    set_layer_name(scatter_layer, target, f"{name}_scatter_layer", source_ir)
    return scatter_layer.get_output(0)
