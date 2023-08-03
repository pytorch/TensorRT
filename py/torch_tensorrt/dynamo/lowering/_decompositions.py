from typing import Callable, Dict, Set
import torch
from torch._decomp import (
    register_decomposition,
    core_aten_decompositions,
    get_decompositions as get_torch_decompositions,
)

aten = torch.ops.aten

_core_aten_decompositions: Dict[
    torch._ops.OpOverload, Callable
] = core_aten_decompositions()
enabled_decompositions: Set[torch._ops.OpOverload] = {
    aten._adaptive_avg_pool2d_backward,
    aten.addcdiv,
    aten.addcdiv_,
    aten.addcmul,
    aten.addcmul_,
    aten.addr,
    aten.aminmax,
    aten.arange.default,
    aten.arange.start,
    aten.avg_pool2d_backward,
    aten.binary_cross_entropy,
    aten.binary_cross_entropy_backward,
    aten.binary_cross_entropy_with_logits,
    aten.celu,
    aten.col2im,
    aten.count_nonzero,
    aten.cudnn_batch_norm,
    aten.cudnn_batch_norm_backward,
    aten.deg2rad,
    aten.detach,
    aten.diag_embed,
    aten.diagonal_backward,
    aten.dot,
    aten.elu,
    aten.elu_backward,
    aten._embedding_bag,
    aten.embedding_dense_backward,
    aten._euclidean_dist.default,
    aten.expand_as,
    aten.eye,
    aten.fill,
    aten.frac,
    aten._fused_moving_avg_obs_fq_helper,
    aten.gelu,
    aten.gelu_backward,
    aten.glu_backward,
    aten.grid_sampler_2d,
    aten.hardshrink,
    aten.hardshrink_backward,
    aten.hardsigmoid,
    aten.hardsigmoid_backward,
    aten.hardswish,
    aten.hardswish_,
    aten.hardswish_backward,
    aten.hardtanh,
    aten.hardtanh_,
    aten.hardtanh_backward,
    aten.heaviside,
    aten.huber_loss,
    aten.huber_loss_backward,
    aten.im2col,
    aten.index_add,
    aten.index_add_,
    aten.index_copy,
    aten.index_copy_,
    aten.index_fill,
    aten.index_fill_,
    aten.index_select,
    aten.isneginf,
    aten.isposinf,
    aten.l1_loss,
    aten.leaky_relu,
    aten.leaky_relu_,
    aten.leaky_relu_backward,
    aten.lerp,
    aten.linspace,
    aten.logaddexp,
    aten.logaddexp2,
    aten.logit,
    aten.logit_backward,
    aten.log_sigmoid_backward,
    aten.log_sigmoid_forward,
    aten._log_softmax,
    aten._log_softmax_backward_data,
    aten.logspace,
    aten.logsumexp.default,
    aten.masked_fill,
    aten.masked_fill_,
    aten.max_pool2d_with_indices_backward,
    aten.mish,
    aten.mse_loss,
    aten.mse_loss_backward,
    aten.mv,
    aten.mvlgamma,
    aten.nansum,
    aten.nan_to_num,
    aten.narrow,
    # TODO: Disable the below operators once freezing is done
    aten.native_batch_norm,
    aten.native_batch_norm_backward,
    aten._native_batch_norm_legit,
    aten._native_batch_norm_legit_functional,
    aten._native_batch_norm_legit_no_training,
    aten.native_dropout_backward,
    aten.native_group_norm,
    aten.native_group_norm_backward,
    aten.native_layer_norm,
    aten.native_layer_norm_backward,
    aten.new_empty,
    aten.new_full,
    aten.new_ones,
    aten.new_zeros,
    aten.nll_loss_backward,
    aten.nll_loss_forward,
    aten.norm,
    aten.ones,
    aten.ones_like,
    aten._prelu_kernel,
    aten._prelu_kernel_backward,
    aten._reshape_alias,
    aten.rad2deg,
    aten.renorm,
    aten.renorm_,
    aten.rot90,
    aten.rsub.Scalar,
    aten.rsub.Tensor,
    aten.select_backward,
    aten.select_scatter,
    aten.sgn,
    aten.sigmoid_backward,
    aten.silu,
    aten.silu_,
    aten.silu_backward,
    aten.sinc,
    aten.slice_backward,
    aten.smooth_l1_loss,
    aten.smooth_l1_loss_backward,
    aten.soft_margin_loss,
    aten.soft_margin_loss_backward,
    aten._softmax,
    aten._softmax_backward_data,
    aten.softplus,
    aten.softplus_backward,
    aten.softshrink,
    aten.softshrink_backward,
    aten.special_entr,
    aten.special_log_ndtr,
    aten.special_xlog1py,
    aten.stack,
    aten.t,
    aten.tanh_backward,
    aten.threshold,
    aten.threshold_backward,
    aten.trace,
    aten.transpose.int,
    aten.tril.default,
    aten.triu.default,
    aten.unfold,
    aten.unfold_backward,
    aten.unfold_copy,
    aten.upsample_bilinear2d,
    aten.upsample_bilinear2d.vec,
    aten.upsample_nearest2d_backward,
    aten.xlogy,
    aten.zero,
    aten.zero_,
    aten.zeros,
    aten.zeros_like,
}
disabled_decompositions: Set[torch._ops.OpOverload] = {}

ENABLED_TORCH_DECOMPOSITIONS: Dict[
    torch._ops.OpOverload, Callable
] = get_torch_decompositions(enabled_decompositions)
TORCH_TRT_DECOMPOSITIONS: Dict[torch._ops.OpOverload, Callable] = {}


def replace_inplace_op(aten_op: OpOverload, outplace_op: OpOverload) -> Any:
    """Replace inplace operation with functional equivalent
    Adapted from:
    https://github.com/pytorch/pytorch/blob/3344d79e3f732dadd5c85b99a7aa1a022f187929/torch/_decomp/decompositions.py#L3355-L3361
    """

    @register_decomposition(aten_op, registry=TORCH_TRT_DECOMPOSITIONS)
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


@register_decomposition(aten.std, registry=TORCH_TRT_DECOMPOSITIONS)
def std_replacement(*args, **kwargs) -> torch.Tensor:
    return torch.sqrt(torch.var(*args, **kwargs))


@register_decomposition(aten.rsqrt, registry=TORCH_TRT_DECOMPOSITIONS)
def rsqrt_replacement(*args, **kwargs) -> torch.Tensor:
    return torch.reciprocal(torch.sqrt(*args, **kwargs))


@register_decomposition(aten._unsafe_view, registry=TORCH_TRT_DECOMPOSITIONS)
def unsafe_view_replacement(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return torch.reshape(x, *args, **kwargs)


@register_decomposition(
    torch.ops.aten.lift_fresh_copy, registry=TORCH_TRT_DECOMPOSITIONS
)
def lift_fresh_copy_replacement(x: torch.Tensor) -> torch.Tensor:
    return x


@register_decomposition(aten.alias, registry=TORCH_TRT_DECOMPOSITIONS)
def alias_replacement(x: torch.Tensor) -> torch.Tensor:
    return x


@register_decomposition(torch.ops.aten.addmm, registry=TORCH_TRT_DECOMPOSITIONS)
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


@register_decomposition(
    torch.ops.aten.reciprocal.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def reciprocal_replacement(
    input_: torch.Tensor,
) -> torch.Tensor:
    return torch.div(1, input_)


def get_decompositions(
    enable_experimental_decompositions: bool = False,
) -> Dict[torch._ops.OpOverload, Callable]:
    if enable_experimental_decompositions:
        CORE_ATEN_DECOMPOSITIONS_FILTERED: Dict[torch._ops.OpOverload, Callable] = {
            decomp: _core_aten_decompositions[decomp]
            for decomp in _core_aten_decompositions
            if (
                decomp not in TORCH_TRT_DECOMPOSITIONS
                and decomp not in disabled_decompositions
            )
        }
        return {**CORE_ATEN_DECOMPOSITIONS_FILTERED, **TORCH_TRT_DECOMPOSITIONS}
    else:
        duplicate_registrations = set(ENABLED_TORCH_DECOMPOSITIONS.keys()).intersection(
            set(TORCH_TRT_DECOMPOSITIONS.keys())
        )
        assert (
            not duplicate_registrations
        ), f"Detected duplicate decompositions on: {duplicate_registrations}"
        return {**ENABLED_TORCH_DECOMPOSITIONS, **TORCH_TRT_DECOMPOSITIONS}
