from typing import Any, Callable, Dict

import torch
from torch._decomp import OpOverload, core_aten_decompositions, register_decomposition

DECOMPOSITIONS: Dict[OpOverload, Callable[..., Any]] = {**core_aten_decompositions()}

aten = torch.ops.aten


def replace_inplace_op(aten_op: OpOverload, outplace_op: OpOverload) -> Any:
    """Replace inplace operation with functional equivalent
    Adapted from:
    https://github.com/pytorch/pytorch/blob/3344d79e3f732dadd5c85b99a7aa1a022f187929/torch/_decomp/decompositions.py#L3355-L3361
    """

    @register_decomposition(aten_op, registry=DECOMPOSITIONS)  # type: ignore[misc]
    def inplace_op(*args: Any, **kwargs: Any) -> Any:
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


@register_decomposition(aten.std, registry=DECOMPOSITIONS)  # type: ignore[misc]
def std_replacement(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.sqrt(torch.var(*args, **kwargs))


@register_decomposition(aten.rsqrt, registry=DECOMPOSITIONS)  # type: ignore[misc]
def rsqrt_replacement(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.reciprocal(torch.sqrt(*args, **kwargs))


@register_decomposition(aten._unsafe_view, registry=DECOMPOSITIONS)  # type: ignore[misc]
def unsafe_view_replacement(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.reshape(x, *args, **kwargs)


@register_decomposition(torch.ops.aten.lift_fresh_copy, registry=DECOMPOSITIONS)  # type: ignore[misc]
def lift_fresh_copy_replacement(x: torch.Tensor) -> torch.Tensor:
    return x


@register_decomposition(aten.alias, registry=DECOMPOSITIONS)  # type: ignore[misc]
def alias_replacement(x: torch.Tensor) -> torch.Tensor:
    return x


@register_decomposition(torch.ops.aten.addmm, registry=DECOMPOSITIONS)  # type: ignore[misc]
def addmm_replacement(
    input_: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: int = 1,
    alpha: int = 1,
) -> torch.Tensor:
    return torch.add(
        torch.mul(input_, beta), torch.mul(torch.matmul(mat1, mat2), alpha)
    )


@register_decomposition(torch.ops.aten.reciprocal.default, registry=DECOMPOSITIONS)  # type: ignore[misc]
def reciprocal_replacement(
    input_: torch.Tensor,
) -> torch.Tensor:
    return torch.div(1, input_)


def get_decompositions() -> Dict[OpOverload, Callable[..., Any]]:
    return DECOMPOSITIONS
