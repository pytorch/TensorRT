from typing import List, Optional, Sequence, Union

import numpy as np
import tensorrt as trt
import torch
from tensorrt import ITensor as TRTTensor
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


def unify_trt_tensors(
    ctx: ConversionContext,
    target: Target,
    name: str,
    inputs: Sequence[Union[int, np.ndarray, torch.Tensor, TRTTensor]],
    concat_axis: int,
    cast_dtype: Union[_enums.dtype, trt.DataType, np.dtype] = None,
    force_trt_output: bool = False,
) -> Union[TRTTensor, List[int]]:
    """
    Normalize all inputs to TRT tensors if needed, optionally cast, and concat if any dynamic.

    Args:
        ctx: TensorRT conversion context.
        target: FX target for naming.
        name: Base name for layers.
        inputs: Sequence of ints / numpy arrays / torch tensors / TRT tensors.
        concat_axis: Axis along which to concatenate tensors if dynamic.
        cast_dtype: Optional target dtype for casting TRT tensors.
        force_trt_output: If True, return TRT tensor even if all inputs are static ints.
    """
    has_dynamic = any(not isinstance(x, int) for x in inputs)
    trt_tensors = []

    for i, x in enumerate(inputs):
        # convert to TRTTensor
        if isinstance(x, TRTTensor):
            t = x
        elif isinstance(x, int) and not has_dynamic and not force_trt_output:
            t = x  # pure static path
        else:
            t = ctx.net.add_constant((1,), np.array([x], dtype=np.int32))
            set_layer_name(t, target, f"{name}_dim{i}_const")
            t = t.get_output(0)

        # optional cast
        if cast_dtype and isinstance(t, TRTTensor):
            t = cast_trt_tensor(ctx, t, cast_dtype, f"{name}_cast_{i}")

        trt_tensors.append(t)

    if not has_dynamic and not force_trt_output:
        return trt_tensors  # all ints

    # promote remaining ints to TRT consts before concat
    for i, t in enumerate(trt_tensors):
        if isinstance(t, int):
            const = ctx.net.add_constant((1,), np.array([t], dtype=np.int32))
            set_layer_name(const, target, f"{name}_static_{i}_const")
            trt_tensors[i] = const.get_output(0)

    concat = ctx.net.add_concatenation(trt_tensors)
    concat.axis = concat_axis
    set_layer_name(concat, target, f"{name}_concat")
    return concat.get_output(0)


def cat(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: Sequence[Union[TRTTensor, torch.Tensor, np.ndarray]],
    dim: int,
    cast_dtype: Union[_enums.dtype, trt.DataType, np.dtype] = None,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    trt_inputs = []
    for i, each_input in enumerate(input):
        if not isinstance(each_input, TRTTensor):
            each_input = get_trt_tensor(ctx, each_input, f"{name}_tensor_{i}")
        if cast_dtype:
            each_input = cast_trt_tensor(
                ctx, each_input, cast_dtype, f"{name}_tensor_int32_cast_{i}"
            )
        trt_inputs.append(each_input)

    if len(trt_inputs) > 1:
        # Cast to promoted type for all inputs
        promoted_type = trt_inputs[0].dtype
        for each_input in trt_inputs[1:]:
            promoted_type = _enums.dtype._from(
                torch.promote_types(
                    _enums.dtype._from(promoted_type).to(torch.dtype),
                    _enums.dtype._from(each_input.dtype).to(torch.dtype),
                )
            )
        trt_promoted_type = promoted_type.to(trt.DataType)

        trt_casted_inputs = []
        for i, each_input in enumerate(trt_inputs):
            casted_input = cast_trt_tensor(
                ctx, each_input, trt_promoted_type, f"{name}_input_casted_{i}"
            )
            trt_casted_inputs.append(casted_input)
        trt_inputs = trt_casted_inputs
    else:
        trt_promoted_type = None

    dim = get_positive_dim(dim, len(trt_inputs[0].shape))
    return unify_trt_tensors(
        ctx,
        target,
        name,
        trt_inputs,
        concat_axis=dim,
        cast_dtype=trt_promoted_type,
        force_trt_output=True,
    )
