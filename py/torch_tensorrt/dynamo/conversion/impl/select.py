# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import List, Optional, Sequence, Union

import numpy as np
import tensorrt as trt
import torch
from tensorrt import ITensor
from torch.fx.node import Target
from torch_tensorrt._enums import dtype
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    broadcastable,
    cast_trt_tensor,
    get_positive_dim,
    get_trt_tensor,
    has_dynamic_shape,
    set_layer_name,
    to_numpy,
)
from torch_tensorrt.dynamo.conversion.impl.elementwise import convert_binary_elementwise
from torch_tensorrt.dynamo.conversion.impl.shape import shape as get_shape
from torch_tensorrt.dynamo.utils import DYNAMIC_DIM

_LOGGER: logging.Logger = logging.getLogger(__name__)


def select(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: ITensor,
    dim: int,
    index: Union[int, ITensor],
) -> ITensor:
    if not isinstance(input, ITensor):
        raise RuntimeError(
            f"slice_tensor received input {input} that is not part "
            "of the TensorRT region!"
        )

    ranks = len(input.shape)
    dim = get_positive_dim(dim, ranks)

    if isinstance(index, ITensor):
        # The index was computed at runtime, so it cannot be folded into a constant.
        indices_tensor = cast_trt_tensor(
            ctx, index, trt.int32, f"{name}_cast_index_tensor", target, source_ir
        )
        # Gather keeps every index dimension in its output, so eager's shape needs a rank 0
        # index. Only a shape made of static 1s is known to hold exactly one element.
        index_shape = tuple(indices_tensor.shape)
        if any(d != 1 for d in index_shape):
            raise RuntimeError(
                "select needs an index with exactly one element and a static "
                f"shape, got shape {index_shape}"
            )
        if index_shape:
            indices_tensor = impl.shuffle.reshape(
                ctx, target, source_ir, f"{name}_index_rank0", indices_tensor, ()
            )
    else:
        indices_tensor = get_trt_tensor(
            ctx, np.array(index, dtype=np.int32), f"{name}_indices_tensor"
        )
    layer = ctx.net.add_gather(input, indices_tensor, dim)
    set_layer_name(layer, target, f"{name}_gather", source_ir)

    return layer.get_output(0)


def is_boolean_tensor(
    tensor: Union[ITensor, np.ndarray, torch.Tensor, torch.fx.Node],
) -> bool:
    if isinstance(tensor, torch.Tensor):
        return bool(tensor.dtype == torch.bool)
    elif isinstance(tensor, np.ndarray):
        return bool(tensor.dtype == np.bool_)
    elif isinstance(tensor, ITensor):
        return bool(tensor.dtype == trt.DataType.BOOL)
    # when index is a node
    else:
        val = tensor.meta.get("val")
        if val is not None and val.dtype is torch.bool:
            return True

    return False


def expand_boolean_indices(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: ITensor,
    indices: Sequence[Union[ITensor, np.ndarray, torch.Tensor]],
) -> Sequence[Union[ITensor, np.ndarray, torch.Tensor]]:
    new_indices = []
    for i, ind in enumerate(indices):
        if ind is not None and is_boolean_tensor(ind):
            _LOGGER.debug(
                f"Boolean index detected at position {i}, converting with nonzero()"
            )
            mask_tensor = get_trt_tensor(ctx, ind, name + f"_bool_mask_{i}")

            nonzero_layer = ctx.net.add_non_zero(mask_tensor)
            set_layer_name(
                nonzero_layer, target, name + f"_bool_nonzero_{i}", source_ir
            )
            # TRT add_non_zero returns shape (ndim, N): row d holds the d-th
            # coordinate of every nonzero element.  This is the transpose of
            # PyTorch's nonzero() which returns (N, ndim).
            # Ref: https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/python-api/infer/Graph/Layers.html#tensorrt.INetworkDefinition.add_non_zero
            nonzero_indices = nonzero_layer.get_output(0)  # (mask_ndim, N)

            mask_ndim = len(ind.shape) if hasattr(ind, "shape") else 1

            if mask_ndim > 1:
                # An N-D mask stands for N index tensors, alone or among other indices.
                # Extract row d (axis=0) from (mask_ndim, N) → (N,) per dim.
                for d in range(mask_ndim):
                    gather_layer = ctx.net.add_gather(
                        nonzero_indices,
                        get_trt_tensor(ctx, d, name + f"_bool_nz_dim_{i}_{d}"),
                        axis=0,
                    )
                    set_layer_name(
                        gather_layer,
                        target,
                        name + f"_bool_nonzero_row_{i}_{d}",
                        source_ir,
                    )
                    row = gather_layer.get_output(0)  # (N,)
                    sq = ctx.net.add_shuffle(row)
                    sq.reshape_dims = (-1,)
                    set_layer_name(
                        sq, target, name + f"_bool_row_sq_{i}_{d}", source_ir
                    )
                    new_indices.append(sq.get_output(0))
                continue  # already appended all per-dim indices; skip append below
            elif len(indices) == 1:
                # x[bool_1d] = v — 1D mask: nonzero → (1, N), flatten to (N,).
                to_squeeze = nonzero_indices
            else:
                # Multi-index bool (1-D bool at position i): extract row i from
                # (1, N) — i.e. gather row 0 along axis=0.
                gather_layer = ctx.net.add_gather(
                    nonzero_indices,
                    get_trt_tensor(ctx, 0, name + f"_dim_index_{i}"),
                    axis=0,
                )
                set_layer_name(
                    gather_layer, target, name + f"_bool_nonzero_extract_{i}", source_ir
                )
                to_squeeze = gather_layer.get_output(0)

            squeeze_layer = ctx.net.add_shuffle(to_squeeze)
            squeeze_layer.reshape_dims = (-1,)
            set_layer_name(
                squeeze_layer,
                target,
                name + f"_bool_mask_squeeze_{i}",
                source_ir,
            )
            squeezed_index = squeeze_layer.get_output(0)
            new_indices.append(squeezed_index)
        else:
            new_indices.append(ind)
    return new_indices


def index(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: ITensor,
    indices: Sequence[Union[ITensor, np.ndarray, torch.Tensor]],
) -> ITensor:
    adv_indx_indices = []
    tensor_indices = []
    # is_numpy is a flag to specify if all the indices are numpy or torchTensor.
    # If any is not this flag will be set to False
    _LOGGER.debug(
        "Determining whether aten.index constant-index optimization can be invoked"
    )
    indices = expand_boolean_indices(ctx, target, source_ir, name, input, indices)
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
            # other cases are kept as ITensor
            if is_numpy:
                ind = to_numpy(ind)
            else:
                ind = get_trt_tensor(ctx, ind, name + f"_parameter_to_fp32_tensor_{i}")
            if last_index is not None:
                assert broadcastable(ind, last_index), (
                    f"Index tensors must be broadcastable with each other, but index {i} "
                    f"has shape {tuple(ind.shape)} which is not broadcastable with the "
                    f"previous index shape {tuple(last_index.shape)}. "
                    "All advanced (integer/boolean) indices must follow NumPy style broadcasting rules. "
                    "See https://numpy.org/doc/stable/user/basics.broadcasting.html"
                )
            last_index = ind
            tensor_indices.append(ind)

    if not tensor_indices:
        cast_layer = ctx.net.add_cast(input, dtype.i32.to(trt.DataType))
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
    input: ITensor,
    dim: int,
    index: ITensor,
) -> ITensor:
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
    input: ITensor,
    dim: int,
    index: Union[ITensor, np.ndarray, torch.Tensor],
    src: Union[ITensor, int, float],
) -> ITensor:
    input_shape = input.shape
    index_shape = index.shape
    index_shape_list = list(index_shape)
    # TensorRT Scatter supports both INT32 and INT64 indices.
    dim = get_positive_dim(dim, len(input_shape))
    src_tensor = src
    # scatter.value - need to create a tensor filled with the scalar value
    if isinstance(src, (int, float)):
        if has_dynamic_shape(index_shape):
            # Dynamic shape: get index shape at runtime and use impl.full operation
            shape_layer = ctx.net.add_shape(index)
            set_layer_name(shape_layer, target, name + "_index_shape", source_ir)
            shape_tensor = shape_layer.get_output(0)
            src_tensor = impl.full.full(
                ctx,
                target,
                source_ir,
                name + "_fill_value",
                shape_tensor,
                src,
                input.dtype,
            )
        else:
            # Static shape: use numpy to create the filled tensor.
            src_tensor = get_trt_tensor(
                ctx,
                np.full(index_shape_list, src),
                name + "_value_tensor",
                dtype=input.dtype,
            )
    # scatter.src
    elif not (isinstance(src, ITensor)):
        src_tensor = get_trt_tensor(ctx, src, name + "_src_tensor")

    if isinstance(src_tensor, ITensor) and src_tensor.dtype != input.dtype:
        src_tensor = cast_trt_tensor(
            ctx, src_tensor, input.dtype, name + "_cast_src_tensor"
        )

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
    input: ITensor,
    dim: int,
    index: Union[ITensor, np.ndarray, torch.Tensor],
) -> ITensor:
    input_shape = input.shape
    dim = get_positive_dim(dim, len(input_shape))
    # Note: TensorRT's Gather layer supports both int32 and int64 indices
    # https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/classnvinfer1_1_1_i_gather_layer.html
    index = get_trt_tensor(ctx, index, name + "_index_tensor")
    gather_layer = ctx.net.add_gather(input, index, axis=dim)
    gather_layer.mode = trt.GatherMode.ELEMENT
    set_layer_name(gather_layer, target, name + "_gather_layer_element", source_ir)
    out = gather_layer.get_output(0)
    return out


def index_put_scatter_add_plugin(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_tensor: ITensor,
    input_indices: Sequence[Union[ITensor, torch.Tensor, np.ndarray, int, None]],
    values: ITensor,
) -> ITensor:
    """Insert a ScatterAdd IPluginV3 layer for index_put(accumulate=True).

    Calls ``at::index_put(..., accumulate=True)`` directly inside the TRT
    engine via the C++ ScatterAddPlugin — same atomicAdd CUDA kernel as
    PyTorch eager, O(P) with no indicator-matrix overhead.

    Supports any number of non-None index tensors (N >= 1).  The plugin
    inputs are laid out as: [src, idx_0, ..., idx_{N-1}, values].
    """
    non_none_indices = [x for x in input_indices if x is not None]
    assert (
        len(non_none_indices) >= 1
    ), "ScatterAdd plugin requires at least one non-None index tensor"

    # Plugin supports float32/float16/bfloat16; cast other types through float32.
    _supported_float_dtypes = (trt.float32, trt.float16, trt.bfloat16)
    original_dtype = input_tensor.dtype
    if original_dtype not in _supported_float_dtypes:
        input_tensor = cast_trt_tensor(
            ctx, input_tensor, trt.float32, f"{name}_src_cast"
        )

    # Ensure index tensors are TRT tensors with int32 or int64 dtype.
    _supported_idx_dtypes = (trt.int32, trt.int64)
    idx_tensors = []
    for i, idx in enumerate(non_none_indices):
        if not isinstance(idx, ITensor):
            idx = get_trt_tensor(ctx, idx, f"{name}_idx_{i}", min_rank=1)
        if idx.dtype not in _supported_idx_dtypes:
            idx = cast_trt_tensor(ctx, idx, trt.int32, f"{name}_idx_{i}_cast")
        idx_tensors.append(idx)

    if not isinstance(values, ITensor):
        values = get_trt_tensor(ctx, values, f"{name}_values", min_rank=1)

    # Values must match src dtype after any cast above.
    if values.dtype != input_tensor.dtype:
        values = cast_trt_tensor(ctx, values, input_tensor.dtype, f"{name}_values_cast")

    creator = trt.get_plugin_registry().get_creator("ScatterAdd", "1", "torch_tensorrt")
    assert (
        creator is not None
    ), "ScatterAdd plugin creator not found — is torch_tensorrt_runtime loaded?"

    pfc = trt.PluginFieldCollection([])
    plugin = creator.create_plugin("ScatterAdd", pfc, trt.TensorRTPhase.BUILD)
    assert plugin is not None, "Failed to create ScatterAdd plugin instance"

    plugin_inputs = [input_tensor] + idx_tensors + [values]
    layer = ctx.net.add_plugin_v3(plugin_inputs, [], plugin)
    set_layer_name(layer, target, name, source_ir)
    out = layer.get_output(0)
    if original_dtype not in _supported_float_dtypes:
        out = cast_trt_tensor(ctx, out, original_dtype, f"{name}_out_cast")
    return out


def index_put_scatter_elements(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_tensor: ITensor,
    input_indices: Sequence[Union[ITensor, torch.Tensor, np.ndarray, None]],
    values: ITensor,
) -> ITensor:
    """``index_put`` with ``accumulate=True`` and one 1-D index, summing duplicate indices.

    The scatter layer leaves duplicate writes undefined, so this uses TensorRT's standard
    ScatterElements plugin with reduction "add", which runs without PyTorch.
    """
    input_indices = expand_boolean_indices(
        ctx, target, source_ir, name, input_tensor, input_indices
    )
    ((dim, index),) = [(d, i) for d, i in enumerate(input_indices) if i is not None]
    if not isinstance(index, ITensor):
        index = get_trt_tensor(ctx, index, f"{name}_index", min_rank=1)
    rank = len(input_tensor.shape)
    shape: List[Union[int, ITensor]] = [
        (
            get_shape(ctx, target, source_ir, f"{name}_dim_{d}", input_tensor, d)
            if input_tensor.shape[d] == DYNAMIC_DIM
            else input_tensor.shape[d]
        )
        for d in range(rank)
    ]
    # The plugin takes int64 indices only, and its TensorRT 10 builds do not wrap negative ones.
    index = cast_trt_tensor(ctx, index, trt.int64, f"{name}_index_int64")
    index = impl.condition.select(
        ctx,
        target,
        source_ir,
        f"{name}_wrap",
        impl.elementwise.add(
            ctx, target, source_ir, f"{name}_wrapped", index, shape[dim]
        ),
        index,
        impl.elementwise.lt(ctx, target, source_ir, f"{name}_negative", index, 0),
    )

    # The plugin wants indices and updates with the input's rank, so give both the shape of
    # input[..., index, ...], which the values broadcast to as in PyTorch.
    updates_shape = list(shape)
    updates_shape[dim] = (
        get_shape(ctx, target, source_ir, f"{name}_index_len", index, 0)
        if index.shape[0] == DYNAMIC_DIM
        else index.shape[0]
    )
    index = impl.shuffle.reshape(
        ctx,
        target,
        source_ir,
        f"{name}_index_axis",
        index,
        tuple(-1 if d == dim else 1 for d in range(rank)),
    )
    index = impl.slice.expand(
        ctx, target, source_ir, f"{name}_index_expand", index, tuple(updates_shape)
    )
    values = impl.slice.expand(
        ctx, target, source_ir, f"{name}_values_expand", values, tuple(updates_shape)
    )

    # 16-bit atomics would round after every duplicate, so sum in float32 and round once.
    compute_dtype = (
        trt.float32
        if input_tensor.dtype in (trt.float16, trt.bfloat16)
        else input_tensor.dtype
    )
    data = cast_trt_tensor(ctx, input_tensor, compute_dtype, f"{name}_input_cast")
    values = cast_trt_tensor(ctx, values, compute_dtype, f"{name}_values_cast")

    creator = trt.get_plugin_registry().get_creator("ScatterElements", "2", "")
    assert creator is not None, "TensorRT's ScatterElements plugin is not registered"
    fields = trt.PluginFieldCollection(
        [
            trt.PluginField(
                "axis", np.array([dim], dtype=np.int32), trt.PluginFieldType.INT32
            ),
            trt.PluginField(
                "reduction",
                np.frombuffer(b"add", dtype=np.int8),
                trt.PluginFieldType.CHAR,
            ),
        ]
    )
    plugin = creator.create_plugin(name, fields, trt.TensorRTPhase.BUILD)
    layer = ctx.net.add_plugin_v3([data, index, values], [], plugin)
    set_layer_name(layer, target, f"{name}_scatter_add", source_ir)
    return cast_trt_tensor(
        ctx, layer.get_output(0), input_tensor.dtype, f"{name}_output_cast"
    )


def _broadcast_shape(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    tensors: Sequence[ITensor],
) -> List[Union[int, ITensor]]:
    """The shape that ``tensors`` broadcast to, with an ITensor for each runtime size.

    In each dim the sizes other than 1 agree, so a static one is the answer whenever there
    is one, and otherwise the first runtime size that is not 1.
    """
    rank = max((len(t.shape) for t in tensors), default=0)
    shape: List[Union[int, ITensor]] = []
    for d in range(rank):
        dims = [(t, len(t.shape) - rank + d) for t in tensors]
        dims = [(t, k) for t, k in dims if k >= 0]
        static = [t.shape[k] for t, k in dims if t.shape[k] not in (1, DYNAMIC_DIM)]
        if static:
            shape.append(static[0])
            continue
        size: Union[int, ITensor] = 1
        for j, (t, k) in enumerate(dims):
            if t.shape[k] != DYNAMIC_DIM:
                continue
            runtime = get_shape(ctx, target, source_ir, f"{name}_dim_{d}_{j}", t, k)
            if isinstance(size, int):
                size = runtime
            else:
                keep = impl.elementwise.ne(
                    ctx, target, source_ir, f"{name}_keep_{d}_{j}", size, 1
                )
                size = impl.condition.select(
                    ctx, target, source_ir, f"{name}_pick_{d}_{j}", size, runtime, keep
                )
        shape.append(size)
    return shape


def _volume(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    shape: Sequence[Union[int, ITensor]],
) -> Union[int, ITensor]:
    """The product of ``shape``, an ITensor when any size is an ITensor."""
    volume: Union[int, ITensor] = 1
    for i, size in enumerate(shape):
        if isinstance(volume, int) and isinstance(size, int):
            volume *= size
        else:
            volume = impl.elementwise.mul(
                ctx, target, source_ir, f"{name}_{i}", volume, size
            )
    return volume


def _broadcast_index_put_values(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    values: ITensor,
    positions: Sequence[int],
    index_shape: Sequence[Union[int, ITensor]],
    free_shape: Sequence[Union[int, ITensor]],
) -> ITensor:
    """``values`` broadcast to ``input[indices]`` as in PyTorch, flat with the index dims first.

    In ``input[indices]``, the broadcast index shape takes the place of the indexed dims at
    ``positions`` when they are adjacent, and comes before the free dims otherwise.
    """
    adjacent = bool(positions) and positions[-1] - positions[0] + 1 == len(positions)
    start = positions[0] if adjacent else 0
    result_shape = (*free_shape[:start], *index_shape, *free_shape[start:])
    # input[indices] can have more dims than TensorRT allows, so broadcast within the values'
    # own dims, then repeat the values over the leading dims they lack.
    lead = len(result_shape) - len(values.shape)
    if lead < 0:
        # TensorRT can hold a 0-d value, such as value + 1, as a 1-D tensor of one element,
        # which has a dim more than input[indices] when every index is 0-d.
        values = impl.shuffle.reshape(
            ctx,
            target,
            source_ir,
            f"{name}_squeeze",
            values,
            tuple(values.shape)[-lead:],
        )
        lead = 0
    # PyTorch rejects a runtime size that is neither 1 nor the size it broadcasts to, while
    # expand would drop the values past that size, so check such sizes at runtime.
    for k, (size, want) in enumerate(zip(tuple(values.shape), result_shape[lead:])):
        if size != DYNAMIC_DIM:
            continue
        got = get_shape(ctx, target, source_ir, f"{name}_size_{k}", values, k)
        fits = impl.elementwise.logical_or(
            ctx,
            target,
            source_ir,
            f"{name}_fits_{k}",
            impl.elementwise.eq(ctx, target, source_ir, f"{name}_one_{k}", got, 1),
            impl.elementwise.eq(ctx, target, source_ir, f"{name}_same_{k}", got, want),
        )
        check = ctx.net.add_assertion(
            fits, "index_put: values do not broadcast to input[indices]"
        )
        set_layer_name(check, target, f"{name}_check_{k}", source_ir)
    if tuple(values.shape) != result_shape[lead:]:
        values = impl.slice.expand(
            ctx, target, source_ir, f"{name}_expand", values, result_shape[lead:]
        )
    values = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_row", values, (1, -1)
    )
    if lead:
        repeats = _volume(
            ctx, target, source_ir, f"{name}_repeats", result_shape[:lead]
        )
        values = impl.slice.expand(
            ctx, target, source_ir, f"{name}_repeat", values, (repeats, values.shape[1])
        )
    # The scatter coordinates list the index dims first. The index can be empty at runtime,
    # so pass every size and keep a size of 0 as 0 rather than as "copy the input dim".
    if start and index_shape:
        values = impl.shuffle.reshape(
            ctx,
            target,
            source_ir,
            f"{name}_split",
            values,
            (
                _volume(ctx, target, source_ir, f"{name}_before", free_shape[:start]),
                _volume(ctx, target, source_ir, f"{name}_indexed", index_shape),
                _volume(ctx, target, source_ir, f"{name}_after", free_shape[start:]),
            ),
            zero_is_placeholder=False,
        )
        values = impl.permutation.permute(
            ctx, target, source_ir, f"{name}_index_dims_first", values, (1, 0, 2)
        )
    return impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_flatten", values, (-1,)
    )


def _index_ranks_in_graph(ctx: ConversionContext) -> List[Optional[int]]:
    """The rank of each index of the index_put being converted, None where unknown."""
    node = ctx.current_node
    indices = node.args[1] if node is not None and len(node.args) > 1 else None
    if not isinstance(indices, (list, tuple)):
        return []
    ranks: List[Optional[int]] = []
    for index in indices:
        val = index.meta.get("val") if isinstance(index, torch.fx.Node) else None
        ranks.append(val.dim() if isinstance(val, torch.Tensor) else None)
    return ranks


def index_put_converter(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_tensor: ITensor,
    input_indices: Sequence[Union[ITensor, torch.Tensor, np.ndarray, int, None]],
    values: ITensor,
) -> ITensor:
    # A converter can hold a 0-d tensor, such as i + 1, as shape (1,), and that extra dim
    # would move the values in input[indices]. Give such an index its rank in the graph.
    # Masks are left as they are, since expand_boolean_indices has no 0-d case.
    ranks = _index_ranks_in_graph(ctx)
    input_indices = [
        (
            impl.shuffle.reshape(
                ctx, target, source_ir, f"{name}_index_{i}_0d", idx, ()
            )
            if isinstance(idx, ITensor)
            and idx.dtype != trt.bool
            and tuple(idx.shape) == (1,)
            and i < len(ranks)
            and ranks[i] == 0
            else idx
        )
        for i, idx in enumerate(input_indices)
    ]
    input_indices = expand_boolean_indices(
        ctx, target, source_ir, name, input_tensor, input_indices
    )
    indices: List[Optional[ITensor]] = [
        None if idx is None else get_trt_tensor(ctx, idx, f"{name}_index_{i}")
        for i, idx in enumerate(input_indices)
    ]
    # The index tensors broadcast together, as in PyTorch. The scatter below wants 1-D
    # indices, so expand each one to that shape and flatten it, unless both are 1-D.
    index_shape = _broadcast_shape(
        ctx, target, source_ir, f"{name}_index", [i for i in indices if i is not None]
    )
    for i, idx in enumerate(indices):
        if idx is None or (len(idx.shape) == 1 and len(index_shape) == 1):
            continue
        if index_shape:
            idx = impl.slice.expand(
                ctx, target, source_ir, f"{name}_index_expand_{i}", idx, index_shape
            )
        indices[i] = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_index_flatten_{i}", idx, (-1,)
        )

    rank = len(input_tensor.shape)
    # Pad the 'indices' list with None for remaining dimensions
    indices = list(indices) + [None] * (rank - len(indices))

    # Separate 'F' (Free) dimensions where None is used, and 'I' (Indexed) dimensions
    F = [i for i in range(rank) if indices[i] is None]  # Free dimensions
    I = [i for i in range(rank) if indices[i] is not None]  # Indexed dimensions
    K = len(I)
    N = _volume(ctx, target, source_ir, f"{name}_index_volume", index_shape)

    if not isinstance(values, ITensor):
        values = get_trt_tensor(ctx, values, f"{name}_values", min_rank=0)
    if isinstance(N, int) and N == 0:
        # Empty index -> no-op. Building the scatter path anyway trips a
        # TensorRT squeezeDims assertion on DGX Spark (GB10, SM 12.1).
        identity_layer = ctx.net.add_identity(input_tensor)
        set_layer_name(identity_layer, target, f"{name}_empty_index_noop", source_ir)
        return identity_layer.get_output(0)

    # Compute shapes and volume for the free dimensions.
    # F_shape_values: per-free-dim size as int (static) or ITensor (dynamic).
    # F_volume: product of F_shape_values, int if all static else ITensor.
    F_shape_values: List[Union[int, ITensor]] = []
    for _fi, _fdim in enumerate(F):
        _s = input_tensor.shape[_fdim]
        if _s == DYNAMIC_DIM:
            F_shape_values.append(
                get_shape(
                    ctx,
                    target,
                    source_ir,
                    f"{name}_fshape_{_fdim}",
                    input_tensor,
                    _fdim,
                )
            )
        else:
            F_shape_values.append(_s)
    F_volume = _volume(ctx, target, source_ir, f"{name}_fvol", F_shape_values)

    # Process indexed dimensions (I)
    I_tensors = []
    for i in I:
        idx = indices[i]
        assert idx is not None
        idx_reshaped = impl.unsqueeze.unsqueeze(
            ctx, target, source_ir, f"{name}_unsqueeze_idx_I_{i}", idx, 1
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
        for _fi2, dim in enumerate(F):
            dim_size = F_shape_values[_fi2]  # int or ITensor
            arange_tensor = impl.arange.arange(
                ctx, target, source_ir, f"{name}_arange_{dim}", 0, dim_size, 1
            )
            arange_tensors.append(arange_tensor)

        if len(arange_tensors) == 1:
            # No need to stack
            meshgrid_stacked = arange_tensors[0]
        else:
            meshgrid_tensors = []
            for i, arange in enumerate(arange_tensors):
                reshape_shape: List[Union[int, ITensor]] = [1] * len(F)
                reshape_shape[i] = F_shape_values[i]
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
                    tuple(F_shape_values),
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
                        (*F_shape_values, 1),
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
        I_combined = [
            impl.shuffle.reshape(
                ctx, target, source_ir, f"{name}_reshape_I_{i}", t, (N, F_volume, 1)
            )
            for i, t in enumerate(I_tensors)
        ]
    else:
        I_combined = []

    # Build the final index list (ii_list) by slicing either I_combined or meshgrid_expanded
    ii_list = []
    i_idx = 0
    f_idx = 0
    for dim in range(rank):
        unique_suffix = f"{dim}_{i_idx if dim in I else f_idx}"
        if dim in I:
            idx_tensor = I_combined[i_idx]
            ii_list.append(idx_tensor)
            i_idx += 1
        else:
            # Extract the f_idx-th column along the last dim (static len(F)) of
            # meshgrid_expanded (shape: N×F_volume×len(F)).  Using gather+unsqueeze
            # avoids passing F_volume (potentially a ITensor) as a slice shape.
            f_idx_t = get_trt_tensor(
                ctx,
                np.array(f_idx, dtype=np.int32),
                f"{name}_f_idx_t_{unique_suffix}",
            )
            gather_l = ctx.net.add_gather(meshgrid_expanded, f_idx_t, axis=2)
            set_layer_name(
                gather_l, target, f"{name}_gather_mesh_{unique_suffix}", source_ir
            )
            mesh_tensor = gather_l.get_output(0)  # (N, F_volume)
            mesh_tensor = impl.unsqueeze.unsqueeze(
                ctx,
                target,
                source_ir,
                f"{name}_unsq_mesh_{unique_suffix}",
                mesh_tensor,
                2,
            )  # (N, F_volume, 1)
            ii_list.append(mesh_tensor)
            f_idx += 1

    # Concatenate the final indices and reshape to (N * F_volume, rank)
    indices_cat = impl.cat.cat(
        ctx, target, source_ir, f"{name}_cat_indices", ii_list, dim=2
    )

    # Flatten the indices_cat to (N * F_volume, rank)
    indices_cat = impl.shuffle.reshape(
        ctx,
        target,
        source_ir,
        f"{name}_reshape_indices_cat",
        indices_cat,
        (-1, rank),
    )

    if 0 in tuple(values.shape):
        # Values with no elements only fit an empty input[indices], which the scatter checks
        # at runtime. Laying them out trips TensorRT's engine build.
        flattened_values = impl.shuffle.reshape(
            ctx, target, source_ir, f"{name}_values_flatten", values, (-1,)
        )
    else:
        flattened_values = _broadcast_index_put_values(
            ctx,
            target,
            source_ir,
            f"{name}_values",
            values,
            I,
            index_shape,
            F_shape_values,
        )
    if flattened_values.dtype != input_tensor.dtype:
        flattened_values = cast_trt_tensor(
            ctx,
            flattened_values,
            input_tensor.dtype,
            f"{name}_values_cast",
        )
    scatter_layer = ctx.net.add_scatter(
        input_tensor,
        indices_cat,
        flattened_values,
        trt.ScatterMode.ND,
    )
    set_layer_name(scatter_layer, target, f"{name}_scatter", source_ir)
    return scatter_layer.get_output(0)
