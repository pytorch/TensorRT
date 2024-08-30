import logging
from typing import Any, Callable, Dict, List, Optional

import torch
from torch._decomp import register_decomposition
from torch._ops import OpOverload
from torch_tensorrt.dynamo._defaults import default_device
from torch_tensorrt.dynamo.conversion.converter_utils import get_positive_dim
from torch_tensorrt.dynamo.utils import to_torch_device

from ._decomposition_groups import (
    ENABLED_TORCH_DECOMPOSITIONS,
    TORCH_TRT_DECOMPOSITIONS,
    _core_aten_decompositions,
    aten,
    torch_disabled_decompositions,
    torch_enabled_decompositions,
)

logger = logging.getLogger(__name__)


def register_torch_trt_decomposition(
    aten_op: OpOverload, registry: Optional[Any] = None
) -> Callable[[Any], Any]:
    """Checks if the decomposition already exists in one of the sets
    Registers the decomposition via the Torch utility

    Alerts the user if the decomposition already exists, before registering
    Throws an AssertionError if the user attempts to register a decomposition
    which is present in the set of explicitly disabled decompositions
    """
    if aten_op in torch_enabled_decompositions:
        logger.warning(
            f"Detected custom decomposition for {aten_op}, which conflicts "
            "with an existing Torch decomposition in torch_enabled_decompositions. "
            "The custom implementation will take precedence."
        )
    elif aten_op in torch_disabled_decompositions:
        logger.info(
            f"Detected custom decomposition for {aten_op}, which is present "
            "in torch_disabled_decompositions."
        )

    # Conflicts with _core_aten_decompositions will only occur if
    # enable_experimental_decompositions is True in get_decompositions
    if aten_op in _core_aten_decompositions:
        logger.debug(
            f"Detected custom decomposition for {aten_op}, which conflicts "
            "with an existing Torch decomposition in core_aten_decompositions. "
            "The custom implementation will take precedence."
        )

    def register(fn: Callable[[Any], Any]) -> Any:
        return register_decomposition(aten_op=aten_op, registry=registry)(fn)

    return register


def replace_inplace_op(aten_op: OpOverload, outplace_op: OpOverload) -> Any:
    """Replace inplace operation with functional equivalent
    Adapted from:
    https://github.com/pytorch/pytorch/blob/3344d79e3f732dadd5c85b99a7aa1a022f187929/torch/_decomp/decompositions.py#L3355-L3361
    """

    @register_torch_trt_decomposition(aten_op, registry=TORCH_TRT_DECOMPOSITIONS)
    def inplace_op(*args, **kwargs):  # type: ignore
        out = outplace_op(*args, **kwargs)
        return args[0].copy_(out)

    return inplace_op


replace_inplace_op(aten.add_, aten.add)
replace_inplace_op(aten.addbmm_, aten.addbmm)
replace_inplace_op(aten.addmm_, aten.addmm)
replace_inplace_op(aten.addmv_, aten.addmv)
replace_inplace_op(aten.baddbmm_, aten.baddbmm)
replace_inplace_op(aten.cumprod_, aten.cumprod)
replace_inplace_op(aten.index_put_, aten.index_put)
replace_inplace_op(aten.index_reduce_, aten.index_reduce)
replace_inplace_op(aten.relu_, aten.relu)
replace_inplace_op(aten.round_, aten.round)
replace_inplace_op(aten.scatter_, aten.scatter)
replace_inplace_op(aten.scatter_add_, aten.scatter_add)
replace_inplace_op(aten.scatter_reduce_, aten.scatter_reduce)


@register_torch_trt_decomposition(aten.rsqrt, registry=TORCH_TRT_DECOMPOSITIONS)
def rsqrt_replacement(*args, **kwargs) -> torch.Tensor:  # type: ignore
    return torch.reciprocal(torch.sqrt(*args, **kwargs))


@register_torch_trt_decomposition(aten._unsafe_view, registry=TORCH_TRT_DECOMPOSITIONS)
def unsafe_view_replacement(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # type: ignore
    return torch.reshape(x, *args, **kwargs)


@register_torch_trt_decomposition(
    torch.ops.aten.lift_fresh_copy, registry=TORCH_TRT_DECOMPOSITIONS
)
def lift_fresh_copy_replacement(x: torch.Tensor) -> torch.Tensor:
    return x


@register_torch_trt_decomposition(aten.alias, registry=TORCH_TRT_DECOMPOSITIONS)
def alias_replacement(x: torch.Tensor) -> torch.Tensor:
    return x


@register_torch_trt_decomposition(
    torch.ops.aten.reciprocal.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def reciprocal_replacement(
    input_: torch.Tensor,
) -> torch.Tensor:
    return torch.div(1, input_)


@register_torch_trt_decomposition(
    torch.ops.prims.var.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def var_decomposition(
    input_tensor: torch.Tensor,
    dims: Optional[List[int]],
    correction: int,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if dims is None:
        dims = []

    # If the dimensions are empty, variance is taken over all dimensions
    if isinstance(dims, (tuple, list)) and len(dims) == 0:
        N = input_tensor.numel()
    # Otherwise, the number of samples is the product of the dimensions reduced over
    else:
        N = 1
        for dim_i in dims:
            N *= input_tensor.shape[dim_i]

    # Compute the mean, difference, and correction term as per the formula:
    # https://pytorch.org/docs/stable/generated/torch.var.html

    # Additionally, prims does not support keepdim, and so we only keep dimensions
    # on the first reduction, then remove it for the second
    sample_mean = torch.mean(input_tensor, dims, keepdim=True)
    diff = input_tensor - sample_mean
    squared_diff = diff * diff
    variance_unnormalized = torch.sum(squared_diff, dims, keepdim=False)

    if correction is None:
        correction_term = float(N - 1)
    elif isinstance(correction, int):
        correction_term = float(N - correction)
    elif isinstance(correction, float):
        correction_term = float(N) - correction
    else:
        raise RuntimeError("correction must be int or float")

    if correction_term <= 0:
        raise RuntimeError(f"correction term was non-positive, got: {correction_term}")

    variance = variance_unnormalized / correction_term

    return variance


@register_torch_trt_decomposition(
    torch.ops.aten.empty_permuted.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def empty_permuted_decomposition(*args, **kwargs) -> torch.Tensor:  # type: ignore
    empty_size = args[0]
    empty_permute = args[1]
    perm = [0] * len(empty_size)
    for permute_index, permute_element in enumerate(empty_permute):
        perm[permute_element] = permute_index
    kwargs["device"] = to_torch_device(default_device())
    return torch.empty([empty_size[l] for l in empty_permute], **kwargs).permute(perm)


@register_torch_trt_decomposition(
    torch.ops.aten.slice_scatter.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def slice_scatter_decomposition(
    input_tensor: torch.Tensor,
    src_tensor: torch.Tensor,
    dim: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: Optional[int] = None,
) -> torch.Tensor:
    dim_size = input_tensor.shape[dim]
    device_input_tensor = input_tensor.device
    start = get_positive_dim(start, input_tensor.shape[dim])
    if end is None:
        end = dim_size
    end = get_positive_dim(end, input_tensor.shape[dim])
    if step is None:
        step = 1

    # Ensure start, end, and step are all integers
    assert isinstance(start, (int, torch.SymInt)), "start must be an integer"
    assert isinstance(end, (int, torch.SymInt)), "end must be an integer"
    assert isinstance(step, (int, torch.SymInt)), "step must be an integer"

    src_dim = src_tensor.shape
    # step == 0 is not a valid torch case
    # also src_dim should be equal to slice dimension

    if start == 0 and end == dim_size and step == 1:
        return src_tensor

    cat_tensors = []
    index_tensor_shape = []
    for i, src_each_dim in enumerate(list(src_dim)):
        if i != dim:
            index_tensor_shape.append(src_each_dim)
    for index in range(start, end, step):
        cat_tensors.append(index * torch.ones(index_tensor_shape, dtype=torch.int64))
    index_tensor = torch.stack(cat_tensors, dim)
    index_tensor = index_tensor.to(device_input_tensor)
    index_tensor_64 = index_tensor.to(torch.int64)
    output_tensor = torch.scatter(input_tensor, dim, index_tensor_64, src_tensor)
    return output_tensor


@register_torch_trt_decomposition(
    torch.ops.aten.select_scatter.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def select_scatter_decomposition(
    input_tensor: torch.Tensor,
    src_tensor: torch.Tensor,
    dim: int,
    index: int,
) -> torch.Tensor:
    src_tensor = torch.unsqueeze(src_tensor, dim)
    return torch.slice_scatter(input_tensor, src_tensor, dim, index, index + 1, 1)


@register_torch_trt_decomposition(
    torch.ops.aten.empty_strided.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def empty_strided_decomposition(*args, **kwargs) -> torch.Tensor:  # type: ignore
    empty_size = args[0]
    empty_stride = args[1]
    return torch.as_strided(
        torch.empty(empty_size, device=to_torch_device(default_device())),
        empty_size,
        empty_stride,
    )


@register_torch_trt_decomposition(
    torch.ops.aten.scatter_add.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def scatter_add_decomposition(
    input_tensor: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src_tensor: torch.Tensor,
) -> torch.Tensor:
    scatter_add_tensor = input_tensor
    src_shape = list(src_tensor.shape)
    src_dim = src_shape[dim]
    for i in range(0, src_dim):
        to_scatter_tensor = torch.zeros(input_tensor.shape, dtype=input_tensor.dtype)
        # index and src slice
        src_slice = torch.select(src_tensor, dim, i)
        index_slice = torch.select(index, dim, i)

        # unsqueeze src and index in dim
        src_slice = torch.unsqueeze(src_slice, dim)
        index_slice = torch.unsqueeze(index_slice, dim)

        # moving tensor to default device
        device = input_tensor.device
        scatter_add_tensor = scatter_add_tensor.to(device)
        to_scatter_tensor = to_scatter_tensor.to(device)
        index_slice = index_slice.to(device)
        src_slice = src_slice.to(device)

        scatter_add_tensor = torch.add(
            scatter_add_tensor,
            torch.scatter(to_scatter_tensor, dim, index_slice, src_slice),
        )

    return scatter_add_tensor


def get_decompositions(
    enable_experimental_decompositions: bool = False,
) -> Dict[OpOverload, Callable[[Any], Any]]:
    if enable_experimental_decompositions:
        CORE_ATEN_DECOMPOSITIONS_FILTERED: Dict[OpOverload, Callable[[Any], Any]] = {
            decomp: _core_aten_decompositions[decomp]
            for decomp in _core_aten_decompositions
            if decomp not in torch_disabled_decompositions
        }
        return {**CORE_ATEN_DECOMPOSITIONS_FILTERED, **TORCH_TRT_DECOMPOSITIONS}
    else:
        return {**ENABLED_TORCH_DECOMPOSITIONS, **TORCH_TRT_DECOMPOSITIONS}
