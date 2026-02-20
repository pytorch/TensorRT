from typing import List, Optional, Sequence, Union

import numpy as np
import tensorrt as trt
from tensorrt import ITensor as TRTTensor
import torch
from torch.fx.node import Target

from torch_tensorrt import _enums
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_positive_dim,
    get_trt_tensor,
    set_layer_name,
)


def _lc_input_dtype(
    tensors: Sequence[TRTTensor],
    cast_dtype: Optional[Union[_enums.dtype, trt.DataType, np.dtype]] = None,
) -> Optional[trt.DataType]:
    if cast_dtype is not None:
        if isinstance(cast_dtype, _enums.dtype):
            return cast_dtype.to(trt.DataType)
        elif isinstance(cast_dtype, (np.dtype, torch.dtype)):
            return _enums.dtype._from(cast_dtype).to(trt.DataType)
        else:
            return cast_dtype  # Already trt.DataType

    if len(tensors) == 0:
        return None

    first_dtype = tensors[0].dtype
    if all(t.dtype == first_dtype for t in tensors):
        return None

    result_dtype = None
    for tensor in tensors:
        if result_dtype is None:
            result_dtype = _enums.dtype._from(tensor.dtype).to(torch.dtype)
        else:
            torch_dtype2 = _enums.dtype._from(tensor.dtype).to(torch.dtype)
            result_dtype = torch.promote_types(result_dtype, torch_dtype2)

    return _enums.dtype._from(result_dtype).to(trt.DataType)


def cat_itensors(
    ctx: ConversionContext,
    target: Target,
    name: str,
    tensors: Sequence[TRTTensor],
    dim: int,
    cast_dtype: Optional[Union[_enums.dtype, trt.DataType, np.dtype]] = None,
) -> TRTTensor:
    common_dtype = _lc_input_dtype(tensors, cast_dtype)
    if common_dtype is not None:
        tensors = [
            cast_trt_tensor(ctx, t, common_dtype, f"{name}_input{i}_cast")
            for i, t in enumerate(tensors)
        ]

    concat_layer = ctx.net.add_concatenation(tensors)
    concat_layer.axis = dim
    set_layer_name(concat_layer, target, f"{name}_concat")

    return concat_layer.get_output(0)


def cat(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: Sequence[Union[TRTTensor, torch.Tensor, np.ndarray]],
    dim: int,
    cast_dtype: Optional[Union[_enums.dtype, trt.DataType, np.dtype]] = None,
) -> TRTTensor:
    if len(input) > 0 and hasattr(input[0], "shape"):
        dim = get_positive_dim(dim, len(input[0].shape))

    _get_trt_tensor = lambda i, t: get_trt_tensor(
        ctx, t, f"{name}_input{i}_const", dtype=None, min_rank=1
    )

    itensors = [_get_trt_tensor(i, t) for i, t in enumerate(input)]

    return cat_itensors(ctx, target, name, itensors, dim, cast_dtype)


def _convert_to_shape_tensor(
    ctx: ConversionContext,
    target: Target,
    name: str,
    value: Union[torch.Tensor, np.ndarray, TRTTensor, int, float],
    index: int,
) -> TRTTensor:
    if isinstance(value, TRTTensor):
        return value

    # Convert to int32 for shape operations (shapes must be integers)
    if isinstance(value, int):
        const_arr = np.array([value], dtype=np.int32)
        shape = (1,)
    else:
        # For tensors, flatten to 1D with int32 dtype
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach()
        const_arr = np.array(value, dtype=np.int32)
        if hasattr(value, "numel"):
            shape = (value.numel(),)
        else:
            shape = (1,)

    layer = ctx.net.add_constant(shape, const_arr)
    set_layer_name(layer, target, f"{name}_input{index}_const")
    return layer.get_output(0)


def unify_and_concat_trt_tensors(
    ctx: ConversionContext,
    target: Target,
    name: str,
    inputs: Sequence[Union[int, np.ndarray, torch.Tensor, TRTTensor]],
    concat_axis: int,
    cast_dtype: Optional[Union[_enums.dtype, trt.DataType, np.dtype]] = None,
    force_trt_output: bool = False,
) -> Union[TRTTensor, List[int]]:
    all_ints = all(isinstance(x, int) for x in inputs)
    if all_ints and not force_trt_output:
        return list(inputs)
    else:
        shape_tensors = [
            _convert_to_shape_tensor(ctx, target, name, inp, i)
            for i, inp in enumerate(inputs)
        ]

        return cat_itensors(ctx, target, name, shape_tensors, concat_axis, cast_dtype)
