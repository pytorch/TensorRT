import logging
from typing import List, Optional, Sequence, Union

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
    input_indices: Sequence[Union[TRTTensor, torch.Tensor, np.ndarray, int, None]],
    values: TRTTensor,
    accumulate: bool = False,
) -> TRTTensor:
    # Convert 'input_indices' to TRT tensors (or keep None as is)
    indices: List[Optional[Union[TRTTensor, None]]] = []
    for i, idx in enumerate(input_indices):
        if idx is None:
            indices.append(None)
        else:
            if not isinstance(idx, TRTTensor):
                idx = get_trt_tensor(ctx, idx, f"{name}_index_{i}", min_rank=1)
            if len(idx.shape) == 0 or not idx.shape:  # Reshape a scalar to (1,)
                idx = impl.shuffle.reshape(
                    ctx, target, source_ir, f"{name}_reshape_idx_{i}", idx, (1,)
                )
            indices.append(idx)

    rank = len(input_tensor.shape)
    # Pad the 'indices' list with None for remaining dimensions
    indices = list(indices) + [None] * (rank - len(indices))

    # Separate 'F' (Free) dimensions where None is used, and 'I' (Indexed) dimensions
    F = [i for i in range(rank) if indices[i] is None]  # Free dimensions
    I = [i for i in range(rank) if indices[i] is not None]  # Indexed dimensions
    K = len(I)

    # Determine the maximum size 'N' among the index tensors
    if K > 0:
        index_shapes = [tensor.shape[0] for tensor in indices if tensor is not None]
        N = max(index_shapes) if index_shapes else 1
    else:
        N = 1

    # Compute shapes and volume for the free dimensions
    F_shapes = [input_tensor.shape[i] for i in F]
    F_volume = trt.volume(F_shapes) if F_shapes else 1

    # Process indexed dimensions (I)
    I_tensors = []
    for i in I:
        idx = indices[i]
        assert idx is not None
        idx_reshaped = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_reshape_idx_I_{i}", idx, (idx.shape[0], 1)
        )
        expanded_idx = impl.slice.expand(
            ctx,
            target,
            source_ir,
            f"{name}_expand_idx_I_{i}",
            idx_reshaped,
            (N, F_volume),
        )
        I_tensors.append(expanded_idx)

    # Create a meshgrid for free dimensions (F)
    if len(F) > 0:
        arange_tensors = []
        for dim in F:
            dim_size = input_tensor.shape[dim]
            arange_tensor = impl.arange.arange(
                ctx, target, source_ir, f"{name}_arange_{dim}", 0, dim_size, 1
            )
            arange_tensors.append(arange_tensor)

        meshgrid_tensors = []
        for i, arange in enumerate(arange_tensors):
            reshape_shape = [1] * len(F)
            reshape_shape[i] = F_shapes[i]
            arange_reshaped = impl.shuffle.reshape(
                ctx,
                target,
                source_ir,
                f"{name}_reshape_arange_F_{F[i]}",
                arange,
                tuple(reshape_shape),
            )
            expanded_arange = impl.slice.expand(
                ctx,
                target,
                source_ir,
                f"{name}_expand_arange_F_{F[i]}",
                arange_reshaped,
                tuple(F_shapes),
            )
            meshgrid_tensors.append(expanded_arange)

        meshgrid_stacked = impl.cat.cat(
            ctx,
            target,
            source_ir,
            f"{name}_stack_meshgrid",
            [
                impl.shuffle.reshape(
                    ctx,
                    target,
                    source_ir,
                    f"{name}_reshape_mesh_{i}",
                    t,
                    (*F_shapes, 1),
                )
                for i, t in enumerate(meshgrid_tensors)
            ],
            dim=-1,
        )
        meshgrid_reshaped = impl.shuffle.reshape(
            ctx,
            target,
            source_ir,
            f"{name}_reshape_meshgrid",
            meshgrid_stacked,
            (F_volume, len(F)),
        )
        if K > 0:
            meshgrid_expanded = impl.slice.expand(
                ctx,
                target,
                source_ir,
                f"{name}_expand_meshgrid",
                meshgrid_reshaped,
                (N, F_volume, len(F)),
            )
        else:
            meshgrid_expanded = meshgrid_reshaped
    else:
        meshgrid_expanded = None

    # Combine all indexed dimensions (I)
    if K > 0:
        I_combined = impl.cat.cat(
            ctx,
            target,
            source_ir,
            f"{name}_cat_I",
            [
                impl.shuffle.reshape(
                    ctx, target, source_ir, f"{name}_reshape_I_{i}", t, (N, F_volume, 1)
                )
                for i, t in enumerate(I_tensors)
            ],
            dim=2,
        )
    else:
        I_combined = None

    # Build the final index list (ii_list) by slicing either I_combined or meshgrid_expanded
    ii_list = []
    i_idx = 0
    f_idx = 0
    for dim in range(rank):
        unique_suffix = f"{dim}_{i_idx if dim in I else f_idx}"
        if dim in I:
            start = [0, 0, i_idx]
            shape = [N, F_volume, 1]
            stride = [1, 1, 1]
            idx_tensor = impl.slice.slice(
                ctx,
                target,
                source_ir,
                f"{name}_slice_I_dim_{unique_suffix}",
                I_combined,
                start,
                shape,
                stride,
            )
            ii_list.append(idx_tensor)
            i_idx += 1
        else:
            start = [0, 0, f_idx]
            shape = [N, F_volume, 1]
            stride = [1, 1, 1]
            mesh_tensor = impl.slice.slice(
                ctx,
                target,
                source_ir,
                f"{name}_slice_F_dim_{unique_suffix}",
                meshgrid_expanded,
                start,
                shape,
                stride,
            )
            ii_list.append(mesh_tensor)
            f_idx += 1

    # Concatenate the final indices and reshape to (N * F_volume, rank)
    indices_cat = impl.cat.cat(
        ctx, target, source_ir, f"{name}_cat_indices", ii_list, dim=2
    )
    indices_cat = impl.shuffle.reshape(
        ctx,
        target,
        source_ir,
        f"{name}_reshape_indices_cat",
        indices_cat,
        (N * F_volume, rank),
    )

    if not isinstance(values, TRTTensor):
        values = get_trt_tensor(ctx, values, f"{name}_values", min_rank=0)

    # Define the expected shape based on (N,) + F_shapes
    expected_shape = (N,) + tuple(F_shapes)

    # Broadcast 'values' to match the expected shape
    if len(values.shape) == 0 or values.shape == (1,):  # Scalar case
        values_reshaped = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_reshape_scalar", values, (1,)
        )
        num_dims = len(expected_shape)
        ones_shape = tuple([1] * num_dims)
        values_reshaped = impl.shuffle.reshape(
            ctx,
            target,
            source_ir,
            f"{name}_reshape_to_ones",
            values_reshaped,
            ones_shape,
        )
        values_expanded = impl.slice.expand(
            ctx,
            target,
            source_ir,
            f"{name}_expand_values_scalar",
            values_reshaped,
            expected_shape,
        )
    else:  # Non-scalar case
        values_shape = list(values.shape)

        # Pad dimensions if necessary
        if len(values_shape) < len(expected_shape):
            values_shape = [1] * (
                len(expected_shape) - len(values_shape)
            ) + values_shape

        # Calculate a broadcastable shape
        broadcast_shape = []
        for exp_dim, val_dim in zip(expected_shape, values_shape):
            if val_dim == 1:
                broadcast_shape.append(exp_dim)
            elif val_dim == exp_dim:
                broadcast_shape.append(val_dim)
            else:
                raise ValueError(f"Cannot broadcast {values_shape} to {expected_shape}")

        # Reshape and then expand
        values_reshaped = impl.shuffle.reshape(
            ctx,
            target,
            source_ir,
            f"{name}_reshape_values",
            values,
            tuple(broadcast_shape),
        )
        values_expanded = impl.slice.expand(
            ctx,
            target,
            source_ir,
            f"{name}_expand_values",
            values_reshaped,
            expected_shape,
        )

    # Flatten values to (N * F_volume,)
    flattened_values = impl.shuffle.reshape(
        ctx,
        target,
        source_ir,
        f"{name}_flatten_values",
        values_expanded,
        (N * F_volume,),
    )

    # Perform Scatter ND operation
    scatter_layer = ctx.net.add_scatter(
        input_tensor,
        indices_cat,
        flattened_values,
        trt.ScatterMode.ND if not accumulate else trt.ScatterMode.ND_ELEMENTWISE_ADD,
    )
    set_layer_name(scatter_layer, target, f"{name}_scatter", source_ir)
    return scatter_layer.get_output(0)
