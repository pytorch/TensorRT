import logging
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import tensorrt as trt
import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    create_constant,
    get_axes_for_reduce_op,
    get_positive_dim,
    get_trt_tensor,
    has_dynamic_shape,
    set_layer_name,
    to_numpy,
)
from torch_tensorrt.dynamo.conversion.impl.cat import cat
from torch_tensorrt.dynamo.conversion.impl.elementwise.ops import ge
from torch_tensorrt.dynamo.conversion.impl.shape import shape as get_shape
from torch_tensorrt.dynamo.types import TRTTensor
from torch_tensorrt.dynamo.utils import DYNAMIC_DIM

_LOGGER: logging.Logger = logging.getLogger(__name__)


def batch_norm(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: Optional[Union[TRTTensor, torch.Tensor, np.ndarray]],
    bias: Optional[Union[TRTTensor, torch.Tensor, np.ndarray]],
    running_mean: Optional[Union[TRTTensor, torch.Tensor, np.ndarray]],
    running_var: Optional[Union[TRTTensor, torch.Tensor, np.ndarray]],
    training: bool,
    momentum: float,
    eps: float,
    cudnn_enabled: bool,
    return_mean_rstd: bool,
) -> Union[TRTTensor, Tuple[TRTTensor, torch.Tensor, torch.Tensor]]:

    if has_dynamic_shape(input.shape):
        assert input.shape[1] != -1, "Channel dim can't be dynamic for batch norm."

    # Save the original output shape for later use
    output_shape = input.shape

    if weight is None:
        weight = get_trt_tensor(ctx, 1.0, f"{name}_weight")
    if bias is None:
        bias = get_trt_tensor(ctx, 0.0, f"{name}_bias")
    if running_mean is None:
        running_mean = get_trt_tensor(ctx, 0.0, f"{name}_running_mean")
    if running_var is None:
        running_var = get_trt_tensor(ctx, 1.0, f"{name}_running_var")

    # eps_tensor for numerical stability
    eps_tensor = get_trt_tensor(ctx, eps, f"{name}_eps")

    # adjusted_var = running_var + eps
    adjusted_var = impl.elementwise.add(
        ctx, target, source_ir, f"{name}_adjusted_var", running_var, eps_tensor
    )

    # sqrt_adjusted_var = sqrt(adjusted_var)
    sqrt_adjusted_var = impl.unary.sqrt(
        ctx, target, source_ir, f"{name}_sqrt", adjusted_var
    )

    # scale = weight / sqrt_adjusted_var
    scale = impl.elementwise.div(
        ctx, target, source_ir, f"{name}_scale", weight, sqrt_adjusted_var
    )

    # scaled_running_mean = running_mean * scale
    scaled_running_mean = impl.elementwise.mul(
        ctx, target, source_ir, f"{name}_scaled_running_mean", running_mean, scale
    )

    # bias_adjusted = bias - scaled_running_mean
    bias_adjusted = impl.elementwise.sub(
        ctx, target, source_ir, f"{name}_bias_adjusted", bias, scaled_running_mean
    )

    # Reshape scale and bias_adjusted to match input shape for broadcasting
    expanded_shape = [1] * len(output_shape)
    expanded_shape[1] = output_shape[1]  # Set channel dimension

    scale_reshape = impl.shuffle.reshape(
        ctx,
        target,
        source_ir,
        f"{name}_reshape_scale",
        scale,
        tuple(expanded_shape),
    )
    bias_adjusted_reshape = impl.shuffle.reshape(
        ctx,
        target,
        source_ir,
        f"{name}_reshape_bias",
        bias_adjusted,
        tuple(expanded_shape),
    )

    # Apply the scale and bias to the input
    scaled_input = impl.elementwise.mul(
        ctx, target, source_ir, f"{name}_scaled_input", input, scale_reshape
    )
    output = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_output",
        scaled_input,
        bias_adjusted_reshape,
    )

    # For BatchNorm1d, reshape output back to original shape if necessary
    if len(output_shape) < 4:
        output = impl.shuffle.reshape(
            ctx,
            target,
            source_ir,
            f"{name}_reshape_1d",
            output,
            output_shape,
        )

    if return_mean_rstd:
        # return fake mean and rstd for now
        return output, None, None

    return output


def layer_norm(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    normalized_shape: List[int],
    weight: Optional[Union[torch.Tensor, np.ndarray]],
    bias: Optional[Union[torch.Tensor, np.ndarray]],
    eps: float,
    cudnn_enable: bool,
    return_mean_rstd: bool,
) -> Union[TRTTensor, Tuple[TRTTensor, torch.Tensor, torch.Tensor]]:
    dims = list(range(len(input.shape) - len(normalized_shape), len(input.shape)))
    axes = get_axes_for_reduce_op(dims)
    weight = get_trt_tensor(ctx, weight, f"{name}_weight")
    bias = get_trt_tensor(ctx, bias, f"{name}_bias")
    # Cast weight and bias to have same dtype as input
    weight = cast_trt_tensor(
        ctx, weight, input.dtype, f"{name}_weight_cast", target, source_ir
    )
    bias = cast_trt_tensor(
        ctx, bias, input.dtype, f"{name}_bias_cast", target, source_ir
    )
    if tuple(input.shape) != tuple(weight.shape):
        weight = impl.slice.expand(
            ctx, target, source_ir, f"{name}_expand_weight", weight, input.shape
        )
    if tuple(input.shape) != tuple(bias.shape):
        bias = impl.slice.expand(
            ctx, target, source_ir, f"{name}_expand_bias", bias, input.shape
        )

    layer_norm = ctx.net.add_normalization(input, weight, bias, axes)
    layer_norm.epsilon = eps
    layer_norm.compute_precision = input.dtype
    set_layer_name(layer_norm, target, f"{name}_layer_norm", source_ir)

    if return_mean_rstd:
        # return fake mean and rstd for now
        return layer_norm.get_output(0), None, None

    return layer_norm.get_output(0)


def native_group_norm(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: Optional[Union[torch.Tensor, np.ndarray]],
    bias: Optional[Union[torch.Tensor, np.ndarray]],
    N: int,
    C: int,
    HxW: int,
    group: int,
    eps: float,
    return_mean_rstd: bool = True,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    # TODO: Ask TRT team about the usage of INormalization Layer usage with num_groups and update the implementation
    #       with INormalization Layer
    assert (
        len(input.shape) >= 3
    ), f"The input dimension should not be less than 3, got {len(input.shape)}!"

    B = input.shape[0]
    # if C is provided, it must be as same as the channel from the input shape,
    # else if C is zero, we should get the channel from the input shape
    if C == 0:
        C = input.shape[1]
    assert (
        C == input.shape[1]
    ), f"The number of Channel={C} must be equal to the number of channels in the input shape={input.shape[1]}"
    # Groups are a subdivision of the channel dimension.
    assert (
        C % group == 0
    ), f"The num of channels ({C}) should be divisible by num_groups ({group})!"
    input = get_trt_tensor(ctx, input, f"{name}_input")

    shape = list(input.shape)

    for i, s in enumerate(shape):
        if i == 0 and s > 0:
            shape[i] = B * group
        elif i == 1:
            shape[i] = C // group
        elif i > 1 and s == -1:
            shape[i] = 0

    # Normalize every group.
    reshaped_input = impl.shuffle.reshape(
        ctx,
        target,
        source_ir,
        f"{name}_reshape_input",
        input,
        shape,
    )

    if weight is None:
        weight = to_numpy(1.0)

    if bias is None:
        bias = to_numpy(0.0)

    weight = get_trt_tensor(ctx, weight, f"{name}_weight")
    bias = get_trt_tensor(ctx, bias, f"{name}_bias")
    weight_bias_shape = (1, C) + (1,) * (len(input.shape) - 2)

    dims = list(range(1, len(input.shape)))

    # E[X]
    mean_trt = impl.reduce.mean(
        ctx,
        target,
        source_ir,
        f"{name}_mean",
        reshaped_input,
        dims,
        True,
    )

    mean_trt = impl.slice.expand(
        ctx,
        target,
        source_ir,
        f"{name}_expand_mean_trt",
        mean_trt,
        reshaped_input.shape,
    )

    # X - E[X]
    sub_trt = impl.elementwise.sub(
        ctx,
        target,
        source_ir,
        f"{name}_sub",
        reshaped_input,
        mean_trt,
    )

    # variance
    pow_trt = get_trt_tensor(ctx, 2, f"{name}_power", np.float32)
    pow_var = impl.elementwise.pow(
        ctx,
        target,
        source_ir,
        f"{name}_pow",
        sub_trt,
        pow_trt,
    )

    var_trt = impl.reduce.mean(
        ctx,
        target,
        source_ir,
        f"{name}_mean_var",
        pow_var,
        dims,
        True,
    )

    var_trt = impl.slice.expand(
        ctx,
        target,
        source_ir,
        f"{name}_expand_var_trt",
        var_trt,
        reshaped_input.shape,
    )

    eps_trt = get_trt_tensor(ctx, eps, f"{name}_eps", np.float32)
    add_trt = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_add",
        var_trt,
        eps_trt,
    )

    sqrt_trt = impl.unary.sqrt(
        ctx,
        target,
        source_ir,
        f"{name}_sqrt",
        add_trt,
    )

    # y = (X - E[X]) / sqrt((var + eps))
    output = impl.elementwise.div(
        ctx,
        target,
        source_ir,
        f"{name}_div",
        sub_trt,
        sqrt_trt,
    )

    shape = list(output.shape)
    for i, s in enumerate(shape):
        if i == 0 and s > 0:
            shape[i] = B
        elif i == 1:
            shape[i] = C
        elif i > 1 and s == -1:
            shape[i] = 0

    reshaped_output = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_reshape_output", output, shape
    )
    reshaped_gamma = impl.shuffle.reshape(
        ctx,
        target,
        source_ir,
        f"{name}_reshape_gamma",
        weight,
        weight_bias_shape,
    )

    reshaped_output = impl.elementwise.mul(
        ctx,
        target,
        source_ir,
        f"{name}_mul_gamma",
        reshaped_output,
        reshaped_gamma,
    )

    reshaped_bias = impl.shuffle.reshape(
        ctx,
        target,
        source_ir,
        f"{name}_reshape_beta",
        bias,
        weight_bias_shape,
    )
    reshaped_output = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_add_beta",
        reshaped_output,
        reshaped_bias,
    )
    if return_mean_rstd:
        # return fake mean and rstd for now
        return reshaped_output, None, None
    return reshaped_output


def group_norm(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    num_groups: int,
    weight: Optional[Union[torch.Tensor, np.ndarray]],
    bias: Optional[Union[torch.Tensor, np.ndarray]],
    eps: float,
    cudnn_enabled: bool,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return native_group_norm(
        ctx,
        target,
        source_ir,
        name,
        input,
        weight,
        bias,
        0,
        0,
        0,
        num_groups,
        eps,
        return_mean_rstd=False,
    )


def softmax(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: Optional[Any] = None,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    input_ranks = len(input.shape)

    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"softmax received input {input} that is not part "
            "of the TensorRT region!"
        )

    # Used to get dim when dim is None. Copied from PyTorch softmax implementation.
    def get_softmax_dim(ndim: int) -> int:
        if ndim == 0 or ndim == 1 or ndim == 3:
            ret = 0
        else:
            ret = 1
        return ret

    if dim is None:
        dim = get_softmax_dim(input_ranks)
    else:
        dim = cast(int, dim)

    dim = get_positive_dim(dim, input_ranks)

    layer = ctx.net.add_softmax(input)
    layer.axes = 1 << dim
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def pdist(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    p: float = 2,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    shape = input.shape
    # Extend input from shape [N, D] to [N, 1, D]
    extend_input = impl.unsqueeze.unsqueeze(
        ctx,
        target,
        source_ir,
        f"{name}_unsqueeze",
        input,
        1,
    )

    # Expand the input from [N, 1, D] to [N, N, D]
    x = impl.slice.expand(
        ctx,
        target,
        source_ir,
        f"{name}_expand",
        extend_input,
        (shape[0], shape[0]) + shape[1:],
    )
    # Subtract the expanded input from original input. Result shape = [N, N, D]
    # This matrix has the distance of each sample to every other sample and hence the shape is [N, N, D]
    x = impl.elementwise.sub(ctx, target, source_ir, f"{name}_sub", x, input)

    if p == 0:
        # norm = torch.sum(x!=0, dim=2)
        nonzero_val = impl.elementwise.ne(ctx, target, source_ir, f"{name}_ne", x, 0)
        norm = impl.reduce.sum(
            ctx, target, source_ir, f"{name}_sum", nonzero_val, dim=2, keepdim=False
        )
        norm = cast_trt_tensor(
            ctx, norm, torch.float32, f"{name}_cast", target, source_ir
        )
    elif p == 1:
        # norm = torch.sum(torch.abs(x), dim=2)
        abs_val = impl.unary.abs(ctx, target, source_ir, f"{name}_abs", x)
        norm = impl.reduce.sum(
            ctx, target, source_ir, f"{name}_sum", abs_val, dim=2, keepdim=False
        )
    elif 0 < p < 1 or 1 < p < float("inf"):
        # norm = torch.pow(torch.sum(torch.pow(torch.abs(x), p), dim=2), 1/p)
        abs_val = impl.unary.abs(ctx, target, source_ir, f"{name}_abs", x)
        pow_val = impl.elementwise.pow(
            ctx, target, source_ir, f"{name}_pow1", abs_val, p
        )
        sum_val = impl.reduce.sum(
            ctx, target, source_ir, f"{name}_sum", pow_val, dim=2, keepdim=False
        )
        norm = impl.elementwise.pow(
            ctx, target, source_ir, f"{name}_pow2", sum_val, 1 / p
        )
    elif p == float("inf"):
        # norm = torch.max(torch.abs(x))
        abs_val = impl.unary.abs(ctx, target, source_ir, f"{name}_abs", x)
        norm = impl.reduce.max(
            ctx,
            target,
            source_ir,
            f"{name}_max",
            abs_val,
            dim=2,
            keepdim=False,
            return_indices=False,
        )
    else:
        raise RuntimeError(
            f"p should between [0, inf], currently p={p} is not supported!"
        )
    if shape[0] == DYNAMIC_DIM:
        dim = get_shape(ctx, target, source_ir, f"{name}_get_shape", input, 0)
        shuffle_layer = ctx.net.add_shuffle(dim)
        shuffle_layer.reshape_dims = trt.Dims()
        set_layer_name(shuffle_layer, target, f"{name}_shuffle", source_ir)
        dim_tensor = shuffle_layer.get_output(0)
        indices_tensor = tri_upper_indices(
            ctx, target, source_ir, f"{name}_triu_indices", dim_tensor
        )
        gather_layer = ctx.net.add_gather_v2(
            norm, indices_tensor, mode=trt.GatherMode.ND
        )
        set_layer_name(gather_layer, target, f"{name}_gather_layer", source_ir)
        gather_layer.axis = 2
        return gather_layer.get_output(0)
    else:
        indices = np.triu_indices(shape[0], k=1)
        return impl.select.index(ctx, target, source_ir, f"{name}_index", norm, indices)


def tri_upper_indices(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    size_tensor: TRTTensor,
) -> TRTTensor:
    """
    Return the indices for the upper-triangle part of a square size of matrix in a N-by-2 Tensor,
    where the diagonal offset = 1. One loop is used to calculate the indices like below.
    x = 0, y = 0, y_start = 1
    out_size = size * (size - 1) // 2
    for _ in range(out_size):
        y_out.append(y_start + y)
        x_out.append(x)
        y += 1
        if (y_start + y) >= size:
            x += 1
            y_start += 1
            y = 0
    Args:
        ctx (ConversionContext): A ConversionContext containing the TensorRT network.
        target (Target): Target of calling node.
        source_ir (Optional[SourceIR]): SourceIR of calling converter.
        name (str): Name of the calling layer.
        size_tensor (TRTTensor): number of rows in the 2-D square matrix. scalar tensor.

    Example:
        if size_tensor is 4, it will return [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    """
    constant_0 = create_constant(ctx, 0, f"{name}_zero", np.int32, 0)
    constant_1 = create_constant(ctx, 1, f"{name}_one", np.int32, 0)
    constant_2 = create_constant(ctx, 2, f"{name}_two", np.int32, 0)

    size_minus_one = impl.elementwise.sub(
        ctx, target, source_ir, f"{name}_size_minus_one", size_tensor, constant_1
    )

    size_mult_prev = impl.elementwise.mul(
        ctx, target, source_ir, f"{name}_size_mult_prev", size_tensor, size_minus_one
    )

    num_loop = impl.elementwise.floor_divide(
        ctx, target, source_ir, f"{name}_num_loop", size_mult_prev, constant_2
    )

    loop = ctx.net.add_loop()
    loop.add_trip_limit(num_loop, trt.TripLimit.COUNT)

    x_recurrence = loop.add_recurrence(constant_0)
    set_layer_name(x_recurrence, target, f"{name}_x_recurrence", source_ir)
    x_tensor = x_recurrence.get_output(0)

    y_recurrence = loop.add_recurrence(constant_0)
    set_layer_name(y_recurrence, target, f"{name}_y_recurrence", source_ir)
    y_tensor = y_recurrence.get_output(0)

    y_start_recurrence = loop.add_recurrence(constant_1)
    set_layer_name(y_start_recurrence, target, f"{name}_y_start_recurrence", source_ir)
    y_start_tensor = y_start_recurrence.get_output(0)

    x_inc = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_x_inc",
        x_tensor,
        constant_1,
    )

    y_current = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_y_current",
        y_start_tensor,
        y_tensor,
    )

    y_inc = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_y_inc",
        y_tensor,
        constant_1,
    )

    next_y = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_next_y",
        y_start_tensor,
        y_inc,
    )

    y_start_inc = impl.elementwise.add(
        ctx,
        target,
        source_ir,
        f"{name}_y_start_inc",
        y_start_tensor,
        constant_1,
    )
    cond = ge(ctx, target, source_ir, f"{name}_cond", next_y, size_tensor)
    x_output = impl.condition.select(
        ctx,
        target,
        source_ir,
        f"{name}_x_output",
        x_inc,
        x_tensor,
        cond,
    )
    x_recurrence.set_input(1, x_output)

    y_start_current = impl.condition.select(
        ctx,
        target,
        source_ir,
        f"{name}_y_start_current",
        y_start_inc,
        y_start_tensor,
        cond,
    )
    y_start_recurrence.set_input(1, y_start_current)

    y_val = impl.condition.select(
        ctx,
        target,
        source_ir,
        f"{name}_y_val",
        constant_0,
        y_inc,
        cond,
    )
    y_recurrence.set_input(1, y_val)

    loop_output_x = loop.add_loop_output(x_tensor, trt.LoopOutput.CONCATENATE)
    loop_output_y = loop.add_loop_output(y_current, trt.LoopOutput.CONCATENATE)
    loop_output_x.set_input(1, num_loop)
    loop_output_y.set_input(1, num_loop)

    # Cat two N tensors into 2 x N. [0, 0, 0], [1, 2, 3] -> [[0, 0, 0], [1, 2, 3]]
    x_index = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_x_index", loop_output_x.get_output(0), (1, -1)
    )
    y_index = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_y_index", loop_output_y.get_output(0), (1, -1)
    )

    x_y_tensor = cat(
        ctx,
        target,
        source_ir,
        f"{name}_x_y_tensor",
        [x_index, y_index],
        0,
    )

    # Reshape 2 x N output to N x 2. [[0, 0, 0], [1, 2, 3]] -> [[0, 1], [0, 2], [0, 3]]
    indices_tensor = ctx.net.add_shuffle(x_y_tensor)
    set_layer_name(indices_tensor, target, f"{name}_indices_tensor", source_ir)
    indices_tensor.first_transpose = trt.Permutation([1, 0])
    indices_tensor.reshape_dims = (-1, 2)

    return indices_tensor.get_output(0)


def cdist_forward(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    x1: TRTTensor,
    x2: TRTTensor,
    p: float,
    compute_mode: Optional[int],
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    """
    Computes pairwise distances between sets of vectors in tensors x1 and x2 using the p-norm. The function treats the last dimension
    of x1 and x2 as feature dimensions, which must be identical for both inputs. The second-to-last dimensions can differ, reflecting
    the number of vectors in each tensor. The dimensions preceding the last are considered as batch dimensions, and pairwise distances
    are computed for each matching set in these dimensions.

    The output tensor's shape is derived by matching the batch dimensions of x1 and x2, where the mismatched batch dimensions are
    merged, and the resulting shape reflects the computed distances for each pair of vectors. It's crucial that the batch dimensions
    (except for the size of sets of vectors to compare) of x1 and x2 either match or one of them is 1 (broadcasting).

    Args:
        x1 (Tensor): input tensor of shape B x P x M.
        x2 (Tensor): input tensor of shape B x R x M.
        p (float): p value for the p-norm distance to calculate between each vector pair
        compute_mode (int): Controls the computation method based on the size of the input sets:
            - None ('use_mm_for_euclid_dist_if_necessary'): Default mode. Uses matrix multiplication to calculate
              Euclidean distance (p=2) if either the number of vectors in x1 or x2 exceeds 25 (P > 25 or R > 25).
            - 1 ('use_mm_for_euclid_dist'): Always use matrix multiplication approach to calculate
            euclidean distance (p = 2)
            - 2 ('donot_use_mm_for_euclid_dist'): Never use matrix multiplication approach to calculate
            euclidean distance (p = 2)

    Example:
    - If x1.shape = [2, 3, 10, 5] and x2.shape = [2, 3, 20, 5], both having the same batch dimensions [2, 3], the output shape will be [2, 3, 10, 20].
      This represents computing distances in two batches of three groups, each comparing 10 vectors from x1 with 20 vectors from x2.
    - For x1.shape = [10, 5] (10 vectors, each of 5 features) and x2.shape = [20, 5] (20 vectors, each of 5 features),
      since there are no batch dimensions to match, the output shape is simply [10, 20], comparing all vectors from x1 against all vectors from x2.

    Note: The `compute_mode` parameter is designed to optimize the performance of the Euclidean distance calculation,
    especially useful when working with large datasets. This parameter allows you to control how the distances are computed,
    with different modes available to leverage matrix multiplication for speed improvements.
    """
    if compute_mode is None:
        compute_mode = 0

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
        if (
            compute_mode == 0 and (x1.shape[-2] > 25 or x2.shape[-2] > 25)
        ) or compute_mode == 1:
            # Compute squared elements
            x1_squared = impl.elementwise.pow(
                ctx, target, source_ir, f"{name}_x1_squared", x1, 2
            )
            x2_squared = impl.elementwise.pow(
                ctx, target, source_ir, f"{name}_x2_squared", x2, 2
            )

            # Sum squares along the last dimension
            x1_sum_squared = impl.reduce.sum(
                ctx,
                target,
                source_ir,
                f"{name}_x1_sum",
                x1_squared,
                dim=-1,
                keepdim=True,
            )
            x2_sum_squared = impl.reduce.sum(
                ctx,
                target,
                source_ir,
                f"{name}_x2_sum",
                x2_squared,
                dim=-1,
                keepdim=True,
            )

            # Reshape sums for broadcasting
            rank = len(x2.shape)
            permute_shape = list(range(rank - 2)) + [rank - 1, rank - 2]
            x1_sum_expanded = x1_sum_squared
            x2_sum_expanded = impl.permutation.permute(
                ctx, target, source_ir, f"{name}_permute", x2_sum_squared, permute_shape
            )

            # Compute dot product of x1 and transposed x2
            x2_tr = impl.permutation.permute(
                ctx, target, source_ir, f"{name}_permute_mm", x2, permute_shape
            )
            dot_product = impl.matmul.matrix_multiply(
                ctx,
                target,
                source_ir,
                f"{name}_dot_product",
                x1,
                x2_tr,
                input_matrix_op=trt.MatrixOperation.NONE,
                other_matrix_op=trt.MatrixOperation.NONE,
            )

            # Combine results to get squared distances
            dist_squared = impl.elementwise.add(
                ctx,
                target,
                source_ir,
                f"{name}_dist_squared_initial",
                x1_sum_expanded,
                x2_sum_expanded,
            )
            dist_squared = impl.elementwise.sub(
                ctx,
                target,
                source_ir,
                f"{name}_dist_squared",
                dist_squared,
                impl.elementwise.mul(
                    ctx, target, source_ir, f"{name}_dot_product_scaled", dot_product, 2
                ),
            )

            # Compute the Euclidean distances
            dist = impl.unary.sqrt(ctx, target, source_ir, f"{name}_dist", dist_squared)
        else:
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
    return dist
