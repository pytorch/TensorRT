import torch
from torch._decomp import register_decomposition, core_aten_decompositions
from typing import Sequence, List


DECOMPOSITIONS = {**core_aten_decompositions()}

aten = torch.ops.aten


def replace_inplace_op(aten_op, outplace_op):
    """Replace inplace operation with functional equivalent
    Adapted from:
    https://github.com/pytorch/pytorch/blob/3344d79e3f732dadd5c85b99a7aa1a022f187929/torch/_decomp/decompositions.py#L3355-L3361
    """

    @register_decomposition(aten_op, registry=DECOMPOSITIONS)
    def inplace_op(*args, **kwargs):
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


@register_decomposition(aten.std, registry=DECOMPOSITIONS)
def std_replacement(*args, **kwargs) -> torch.Tensor:
    return torch.sqrt(torch.var(*args, **kwargs))


@register_decomposition(aten.rsqrt, registry=DECOMPOSITIONS)
def rsqrt_replacement(*args, **kwargs) -> torch.Tensor:
    return torch.reciprocal(torch.sqrt(*args, **kwargs))


@register_decomposition(aten._unsafe_view, registry=DECOMPOSITIONS)
def unsafe_view_replacement(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return torch.reshape(x, *args, **kwargs)


@register_decomposition(torch.ops.aten.lift_fresh_copy, registry=DECOMPOSITIONS)
def lift_fresh_copy_replacement(x: torch.Tensor) -> torch.Tensor:
    return x


@register_decomposition(aten.alias, registry=DECOMPOSITIONS)
def alias_replacement(x: torch.Tensor) -> torch.Tensor:
    return x


@register_decomposition(torch.ops.aten.addmm, registry=DECOMPOSITIONS)
def addmm_replacement(
    input_: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, *, beta=1, alpha=1
) -> torch.Tensor:
    return torch.add(
        torch.mul(input_, beta), torch.mul(torch.matmul(mat1, mat2), alpha)
    )


@register_decomposition(torch.ops.aten.reciprocal.default, registry=DECOMPOSITIONS)
def reciprocal_replacement(
    input_: torch.Tensor,
) -> torch.Tensor:
    return torch.div(1, input_)


@register_decomposition(torch.ops.aten.var_mean, registry=DECOMPOSITIONS)
def var_mean_replacement(
    input_: torch.Tensor,
    dim: int = None,
    correction: int = 1,
    keepdim: bool = False,
    unbiased: bool = True,
) -> List[torch.Tensor] :
    if (dim is not None):
        mean = torch.mean(input_)
        variance = torch.mean(torch.pow(torch.sub(input_, mean), 2))
    else:
        mean = torch.mean(input_, dim)
        variance = torch.mean(torch.pow(torch.sub(input_, mean), 2), dim)

    if (unbiased):
        N = torch.numel(input_)
        correction_term = N / (N - correction)
        variance = torch.mul(variance, correction_term)
    
    return [variance, mean]


@register_decomposition(torch.ops.aten.var_mean.dim, registry=DECOMPOSITIONS)
def var_mean_dim_replacement(
    input_: torch.Tensor,
) -> torch.Tensor:
    return torch.div(1, input_)


@register_decomposition(torch.ops.aten.var_mean, registry=DECOMPOSITIONS)
def var_mean_correction_replacement(
    input_: torch.Tensor,
) -> torch.Tensor:
    return torch.div(1, input_)



def get_decompositions():
    return DECOMPOSITIONS
