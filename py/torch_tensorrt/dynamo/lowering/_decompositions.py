import logging
from typing import Any, Callable, Dict, List, Optional

import torch
from torch._decomp import register_decomposition
from torch._ops import OpOverload
from torch_tensorrt.dynamo.conversion.converter_utils import get_positive_dim

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
    torch.ops.aten.slice_scatter.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def slice_scatter_decomposition(
    input_tensor: torch.Tensor,
    src_tensor: torch.Tensor,
    dim: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: Optional[int] = None,
):
    dim_size = input_tensor.shape[dim]
    start = get_positive_dim(start, input_tensor.shape[dim])
    if end is None:
        end = dim_size
    end = get_positive_dim(end, input_tensor.shape[dim])
    if step is None:
        step = 1

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
    indices = torch.arange(start, end, step)
    cat_tensors = [(indices * torch.ones(index_tensor_shape))]
    index_tensor = torch.stack(cat_tensors, dim)
    index_tensor = index_tensor.to(torch.int64).cuda()
    output_tensor = torch.scatter(input_tensor, dim, index_tensor, src_tensor)
    return output_tensor


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
