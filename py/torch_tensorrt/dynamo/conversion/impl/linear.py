from typing import Optional, Sequence, Union

import numpy as np
import tensorrt as trt
import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    SourceIR,
    cast_trt_tensor,
    get_trt_tensor,
)
from torch_tensorrt.fx.types import TRTTensor


def linear(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: Union[TRTTensor, torch.Tensor, np.ndarray],
    bias: Optional[Union[TRTTensor, torch.Tensor, np.ndarray]],
) -> TRTTensor:
    # Process weight terms
    if not isinstance(weight, (TRTTensor, torch.Tensor, np.ndarray)):
        raise RuntimeError(
            f"Linear layer {name} has weight of type {type(weight)}, Expect Union[TRTTensor, torch.Tensor, np.ndarray],"
        )
    elif isinstance(weight, (torch.Tensor, np.ndarray)):
        weight = get_trt_tensor(ctx, weight, f"{name}_weight")

    # Process bias terms
    if bias is not None and not isinstance(bias, (TRTTensor, torch.Tensor, np.ndarray)):
        raise RuntimeError(
            f"Linear layer {name} has bias of type {type(bias)}, Expect Union[TRTTensor, torch.Tensor, np.ndarray],"
        )
    elif isinstance(bias, (torch.Tensor, np.ndarray)):
        bias = get_trt_tensor(ctx, bias, f"{name}_bias")

    # add IMatrixMultiplyLayer
    out = impl.matmul.matrix_multiply(
        ctx,
        target,
        source_ir,
        name,
        input,
        weight,
        input_matrix_op=trt.MatrixOperation.NONE,
        other_matrix_op=trt.MatrixOperation.TRANSPOSE,
    )

    if bias is not None:
        # add bias
        out = impl.elementwise.add(ctx, target, source_ir, name, out, bias)

    return out


def cdist_forward(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    x1: TRTTensor,
    x2: TRTTensor,
    p: float,
    compute_mode: Optional[int] = None,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    """
    Computes pairwise distances between sets of vectors in tensors x1 and x2 using the p-norm. The function treats the last dimension
    of x1 and x2 as feature dimensions, which must be identical for both inputs. The second-to-last dimensions can differ, reflecting
    the number of vectors in each tensor. The dimensions preceding the last are considered as batch dimensions, and pairwise distances
    are computed for each matching set in these dimensions.

    The output tensor's shape is derived by matching the batch dimensions of x1 and x2, where the mismatched batch dimensions are
    merged, and the resulting shape reflects the computed distances for each pair of vectors. It's crucial that the batch dimensions
    (except for the size of sets of vectors to compare) of x1 and x2 either match or one of them is 1 (broadcasting).

    Example:
    - If x1.shape = [2, 3, 10, 5] and x2.shape = [2, 3, 20, 5], both having the same batch dimensions [2, 3], the output shape will be [2, 3, 10, 20].
      This represents computing distances in two batches of three groups, each comparing 10 vectors from x1 with 20 vectors from x2.
    - For x1.shape = [10, 5] (10 vectors, each of 5 features) and x2.shape = [20, 5] (20 vectors, each of 5 features),
      since there are no batch dimensions to match, the output shape is simply [10, 20], comparing all vectors from x1 against all vectors from x2.
    """
    x1_expand_shape = list(x1.shape[:-1]) + [1, x1.shape[-1]]
    x2_expand_shape = list(x2.shape[:-2]) + [1] + list(x2.shape[-2:])

    # Reshape x1 and x2 for broadcasting
    x1_expanded = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_x1_expand", x1, x1_expand_shape
    )
    x2_expanded = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_x2_expand", x2, x2_expand_shape
    )

    diff = impl.elementwise.sub(
        ctx, target, source_ir, f"{name}_diff", x1_expanded, x2_expanded
    )

    if p == 0:
        diff_non_zero = impl.elementwise.ne(
            ctx, target, source_ir, f"{name}_diff_non_zero", diff, 0
        )
        diff_non_zero = cast_trt_tensor(
            ctx, diff_non_zero, torch.float32, f"{name}_cast", target, source_ir
        )
        dist = impl.reduce.sum(
            ctx,
            target,
            source_ir,
            f"{name}_sum",
            diff_non_zero,
            dim=-1,
            keepdim=False,
        )
    elif p == 1:
        abs_val = impl.unary.abs(ctx, target, source_ir, f"{name}_abs_val", diff)
        dist = impl.reduce.sum(
            ctx, target, source_ir, f"{name}_sum", abs_val, dim=-1, keepdim=False
        )
    elif p == 2:
        diff_squared = impl.elementwise.pow(
            ctx, target, source_ir, f"{name}_diff_squared", diff, 2
        )
        dist_squared = impl.reduce.sum(
            ctx,
            target,
            source_ir,
            f"{name}_dist_sq_sum",
            diff_squared,
            dim=-1,
            keepdim=False,
        )
        dist = impl.unary.sqrt(ctx, target, source_ir, f"{name}_sqrt", dist_squared)
    elif 0 < p < 1 or 1 < p < 2 or 2 < p < float("inf"):
        abs_val = impl.unary.abs(ctx, target, source_ir, f"{name}_abs_val", diff)
        pow_val = impl.elementwise.pow(
            ctx, target, source_ir, f"{name}_pow_val_1", abs_val, p
        )
        sum_val = impl.reduce.sum(
            ctx, target, source_ir, f"{name}_sum", pow_val, dim=-1, keepdim=False
        )
        dist = impl.elementwise.pow(
            ctx, target, source_ir, f"{name}_pow_val_2", sum_val, 1 / p
        )
    elif p == float("inf"):
        abs_val = impl.unary.abs(ctx, target, source_ir, f"{name}_abs_val", diff)
        dist = impl.reduce.max(
            ctx,
            target,
            source_ir,
            f"{name}_max",
            abs_val,
            dim=-1,
            keepdim=False,
            return_indices=False,
        )
    else:
        raise NotImplementedError(f"Currently, p={p} is not implemented.")
    return dist
