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


def unify_and_concat_trt_tensors(
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
        target: Operation Target.
        name: Operation Name.
        inputs: Sequence of ints / numpy arrays / torch tensors / TRT tensors.
        concat_axis: Axis along which to concatenate tensors if dynamic.
        cast_dtype: Optional target dtype for casting TRT tensors.
        force_trt_output: If True, return TRT tensor even if all inputs are static ints. (True for concat operations)
    """
    # Normalize scalar tensors (0D) to Python values to avoid 0D vs 1D shape issues.
    #
    # eg case:
    # torch.tensor(3) is a 0D tensor (shape=[])
    # get_trt_tensor creates a 0D TRT constant for it (shape=trt.Dims())
    # Python int 3 via get_trt_tensor creates a 1D TRT constant (shape=(1,))
    # because to_torch(3) returns torch.tensor([3]) with shape (1,)
    #
    # By normalizing torch.tensor(3) -> 3, we ensure:
    # 1. Pure static case: all ints -> returns list directly, no TRT ops needed (eg:upsample)
    # 2. Mixed case: Python ints become 1D constants, compatible with other 1D tensors
    normalized_inputs = []
    for x in inputs:
        if isinstance(x, (torch.Tensor, np.ndarray)) and x.ndim == 0:
            normalized_inputs.append(x.item())
        else:
            normalized_inputs.append(x)
    has_dynamic = any(not isinstance(x, int) for x in normalized_inputs)
    trt_tensors = []

    for i, x in enumerate(normalized_inputs):
        # convert to TRTTensor
        if isinstance(x, TRTTensor):
            t = x
        elif isinstance(x, int) and not has_dynamic and not force_trt_output:
            t = x  # pure static path
        else:
            # Use get_trt_tensor which handles empty tensors properly via create_constant
            t = get_trt_tensor(ctx, x, f"{name}_input_{i}")
        trt_tensors.append(t)

    if not has_dynamic and not force_trt_output:
        return trt_tensors  # all ints

    final_dtype = None
    if cast_dtype:
        # Explicit cast requested
        if isinstance(cast_dtype, _enums.dtype):
            final_dtype = cast_dtype.to(trt.DataType)
        elif isinstance(cast_dtype, (np.dtype, torch.dtype)):
            final_dtype = _enums.dtype._from(cast_dtype).to(trt.DataType)
        else:
            final_dtype = cast_dtype  # already trt.DataType
    else:
        # Automatic promotion
        promoted_type = None
        for t in trt_tensors:
            if isinstance(t, TRTTensor):
                if promoted_type is None:
                    promoted_type = t.dtype
                else:
                    promoted_type = _enums.dtype._from(
                        torch.promote_types(
                            _enums.dtype._from(promoted_type).to(torch.dtype),
                            _enums.dtype._from(t.dtype).to(torch.dtype),
                        )
                    ).to(trt.DataType)
        final_dtype = promoted_type

    # final cast
    if final_dtype is not None:
        casted = []
        for i, t in enumerate(trt_tensors):
            if isinstance(t, TRTTensor):
                t = cast_trt_tensor(ctx, t, final_dtype, f"{name}_cast_{i}")
            casted.append(t)
        trt_tensors = casted

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
    # int is only when cat called in other ops like pad
    if not isinstance(input[0], int):
        dim = get_positive_dim(dim, len(input[0].shape))
    else:
        dim = 0
    return unify_and_concat_trt_tensors(
        ctx,
        target,
        name,
        input,
        concat_axis=dim,
        cast_dtype=cast_dtype,
        force_trt_output=True,
    )
