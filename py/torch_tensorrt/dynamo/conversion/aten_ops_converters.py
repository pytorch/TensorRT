import logging
import operator
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.fx.node import Argument, Node, Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.converter_utils import (
    dynamic_unsupported_with_args,
    enforce_tensor_types,
    is_only_operator_on_placeholder,
)
from torch_tensorrt.fx.types import TRTTensor

_LOGGER: logging.Logger = logging.getLogger(__name__)


def args_bounds_check(
    args: Tuple[Argument, ...], i: int, replacement: Optional[Any] = None
) -> Any:
    return args[i] if len(args) > i else replacement


def get_ir(target: Target) -> SourceIR:
    target_module = getattr(target, "__module__", "None")
    if any(
        target_module.startswith(prefix)
        for prefix in ("torch.ops.aten", "torch._ops.aten")
    ):
        return SourceIR.ATEN
    elif any(
        target_module.startswith(prefix)
        for prefix in ("torch.ops.prims", "torch._ops.prims")
    ):
        return SourceIR.PRIM
    elif target_module.startswith("torch.nn"):
        return SourceIR.NN

    return SourceIR.UNKNOWN


def one_user_validator(node: Node) -> bool:
    # Validate only one user, which is a getitem node that accesses the first element in the list
    return (
        len(node.users) == 1
        and list(node.users)[0].target == operator.getitem
        and list(node.users)[0].args[1] == 0
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.native_batch_norm.default, capability_validator=one_user_validator
)
@dynamo_tensorrt_converter(torch.ops.aten.batch_norm.default)
@dynamo_tensorrt_converter(torch.ops.aten.batch_norm)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_batch_norm(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.batch_norm(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        weight=args[1],
        bias=args[2],
        running_mean=args[3],
        running_var=args[4],
        training=args[5],
        momentum=args[6],
        eps=args[7],
        cudnn_enabled=args_bounds_check(args, 8, True),
        return_mean_rstd=(target == torch.ops.aten.native_batch_norm.default),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten._native_batch_norm_legit_no_training.default,
    capability_validator=one_user_validator,
)
def aten_ops_batch_norm_legit_no_training(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.batch_norm(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        weight=args[1],
        bias=args[2],
        running_mean=args[3],
        running_var=args[4],
        training=False,
        momentum=args[5],
        eps=args[6],
        cudnn_enabled=False,
        return_mean_rstd=(
            target == torch.ops.aten._native_batch_norm_legit_no_training.default
        ),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.native_layer_norm.default, capability_validator=one_user_validator
)
@dynamo_tensorrt_converter(torch.ops.aten.layer_norm.default)
@dynamo_tensorrt_converter(torch.ops.aten.layer_norm)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_layer_norm(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.layer_norm(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        normalized_shape=args[1],
        weight=args_bounds_check(args, 2),
        bias=args_bounds_check(args, 3),
        eps=args_bounds_check(args, 4, 1e-05),
        cudnn_enable=args_bounds_check(args, 5, True),
        return_mean_rstd=(target == torch.ops.aten.native_layer_norm.default),
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.native_group_norm.default, capability_validator=one_user_validator
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_native_group_norm(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.native_group_norm(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        weight=args[1],
        bias=args[2],
        N=args[3],
        C=args[4],
        HxW=args[5],
        group=args[6],
        eps=args[7],
    )


@dynamo_tensorrt_converter(torch.ops.aten.group_norm.default)
@dynamo_tensorrt_converter(torch.ops.aten.group_norm)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_group_norm(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.group_norm(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        num_groups=args[1],
        weight=args_bounds_check(args, 2, None),
        bias=args_bounds_check(args, 3, None),
        eps=args_bounds_check(args, 4, 1e-05),
        cudnn_enabled=args_bounds_check(args, 5, True),
    )


@dynamo_tensorrt_converter(torch.ops.aten.cat.default)
def aten_ops_cat(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.cat.cat(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        dim=args_bounds_check(args, 1, 0),
    )


def embedding_param_validator(embedding_node: Node) -> bool:
    scale_grad_by_freq = args_bounds_check(embedding_node.args, 3)
    sparse = args_bounds_check(embedding_node.args, 4)

    if scale_grad_by_freq is not None:
        _LOGGER.debug(
            f"Currently we don't support specifying scale gradient by word frequency, got {scale_grad_by_freq}."
        )
        return False

    if sparse is not None:
        _LOGGER.debug(f"Currently we don't support sparse gradient, got {sparse}.")
        return False

    return True


@dynamo_tensorrt_converter(
    torch.ops.aten.embedding.default, capability_validator=embedding_param_validator
)
def aten_ops_embedding(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.embedding.embedding(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[1],
        weight=args[0],
        # args[2] is the padding index, which is useful for training only
        scale_grad_by_freq=args_bounds_check(args, 3),
        sparse=args_bounds_check(args, 4),
    )


def embedding_bag_validator(node: Node) -> bool:
    mode = args_bounds_check(node.args, 4, 0)
    indices = node.args[1].meta.get("tensor_meta")
    if indices is None:
        return False
    return (
        bool(node.args[2].op == "get_attr")
        and (mode == 0 or mode == 1 or mode == 2)
        and len(indices.shape) == 1
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.embedding_bag.default, capability_validator=embedding_bag_validator
)
@dynamo_tensorrt_converter(
    torch.ops.aten._embedding_bag.default, capability_validator=embedding_bag_validator
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
        1: (TRTTensor,),
        2: (np.ndarray, torch.Tensor),
    }
)
def aten_ops_embedding_bag(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.embedding.embedding_bag(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        weight=args[0],
        indices=args[1],
        offsets=args[2],
        scale_grad_by_freq=args_bounds_check(args, 3, False),
        mode=args_bounds_check(args, 4, 0),
        sparse=args_bounds_check(args, 5, False),
        per_sample_weights=args_bounds_check(args, 6, None),
        include_last_offset=args_bounds_check(args, 7, False),
        # padding index is useful for training only
    )


@dynamo_tensorrt_converter(torch.ops.aten.fmod.Scalar)
@dynamo_tensorrt_converter(torch.ops.aten.fmod.Tensor)
def aten_ops_fmod(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.fmod(ctx, target, SourceIR.ATEN, name, args[0], args[1])


@dynamo_tensorrt_converter(torch.ops.aten.grid_sampler)
@dynamo_tensorrt_converter(torch.ops.aten.grid_sampler_2d)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
        1: (TRTTensor,),
    }
)
def aten_ops_grid(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.grid.grid(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        grid=args[1],
        interpolation_mode=args[2],
        padding_mode=args[3],
        align_corners=args[4],
    )


@dynamo_tensorrt_converter(torch.ops.aten.relu.default)
def aten_ops_relu(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.relu(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.sigmoid.default)
def aten_ops_sigmoid(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.sigmoid(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


def index_dtype_validator(node: Node) -> bool:
    index = node.args[1]
    for ind in index:
        if ind is not None:
            val = ind.meta.get("val")
            if val is not None and val.dtype not in (torch.int32, torch.int64):
                return False
    return True


@dynamo_tensorrt_converter(
    torch.ops.aten.index.Tensor, capability_validator=index_dtype_validator
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_index(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.select.index(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.tanh.default)
def aten_ops_tanh(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.tanh(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.leaky_relu.default)
def aten_ops_leaky_relu(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.leaky_relu(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, 0.01),
    )


@dynamo_tensorrt_converter(torch.ops.aten.elu.default)
def aten_ops_elu(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.elu(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        alpha=args_bounds_check(args, 1, 1.0),
        beta=args_bounds_check(args, 2, None),
    )


@dynamo_tensorrt_converter(torch.ops.aten.softplus.default)
def aten_ops_softplus(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.softplus(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        beta=args_bounds_check(args, 1, 1),
    )


@dynamo_tensorrt_converter(torch.ops.aten.hardsigmoid.default)
def aten_ops_hard_sigmoid(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.hard_sigmoid(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        alpha=args_bounds_check(args, 1, 1 / 6),
        beta=args_bounds_check(args, 2, 1 / 2),
    )


@dynamo_tensorrt_converter(torch.ops.aten.matmul)
@dynamo_tensorrt_converter(torch.ops.aten.mm.default)
@dynamo_tensorrt_converter(torch.ops.aten.mv.default)
@dynamo_tensorrt_converter(torch.ops.aten.bmm.default)
def aten_ops_matmul(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.matmul.matrix_multiply(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.rsqrt.default)
def aten_ops_rsqrt(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.rsqrt(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.neg.default)
def aten_ops_neg(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.neg(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.squeeze.dim)
@dynamo_tensorrt_converter(torch.ops.aten.squeeze.dims)
def aten_ops_squeeze(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.squeeze.squeeze(ctx, target, SourceIR.ATEN, name, args[0], args[1])


@dynamo_tensorrt_converter(torch.ops.aten.erf.default)
def aten_ops_erf(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.erf(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.unsqueeze.default)
def aten_ops_unsqueeze(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unsqueeze.unsqueeze(
        ctx, target, SourceIR.ATEN, name, input_t=args[0], dim=args[1]
    )


@dynamo_tensorrt_converter(torch.ops.aten._softmax.default)
def aten_ops_softmax(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.softmax(
        ctx, target, SourceIR.ATEN, name, args[0], args[1]
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.split.Tensor, capability_validator=dynamic_unsupported_with_args([1])
)
@dynamo_tensorrt_converter(
    torch.ops.aten.split.sizes, capability_validator=dynamic_unsupported_with_args([1])
)
@dynamo_tensorrt_converter(
    torch.ops.aten.split_with_sizes.default,
    capability_validator=dynamic_unsupported_with_args([1]),
)
def aten_ops_split(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.split.split(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        split_size_or_sections=args[1],
        dim=args_bounds_check(args, 2, 0),
    )


@dynamo_tensorrt_converter(torch.ops.aten.where.self)
def aten_ops_where(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.condition.where(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[1],
        args[2],
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.clamp.default)
@dynamo_tensorrt_converter(torch.ops.aten.clamp.Tensor)
@dynamo_tensorrt_converter(torch.ops.aten.clip.default)
@dynamo_tensorrt_converter(torch.ops.aten.clip.Tensor)
def aten_ops_clamp(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.clamp(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input_val=args[0],
        min_val=args_bounds_check(args, 1),
        max_val=args_bounds_check(args, 2),
    )


@dynamo_tensorrt_converter(torch.ops.aten.select.int)
def aten_ops_select(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.select.select(
        ctx, target, SourceIR.ATEN, name, args[0], args[1], args[2]
    )


@dynamo_tensorrt_converter(torch.ops.aten.slice.Tensor)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_slice(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.slice.slice_op(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, replacement=0),
        args_bounds_check(args, 2, replacement=None),
        args_bounds_check(args, 3, replacement=None),
        args_bounds_check(args, 4, replacement=1),
    )


@dynamo_tensorrt_converter(torch.ops.aten.chunk.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_chunk(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.slice.chunk(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
        args_bounds_check(args, 2, 0),
    )


@dynamo_tensorrt_converter(torch.ops.aten.cumsum.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_cumsum(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.slice.cumsum(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.tile.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_tile(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.slice.tile(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.permute.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_permute(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.permutation.permute(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


def to_copy_dtype_validator(placeholder_only: bool) -> Callable[[Node], bool]:
    """Return validator for to_copy node with placeholder restrictions"""

    def validate_dtype(to_copy_node: Node) -> bool:
        """Returns true if the to_copy node can be converted to TRT

        Based on data type being casted to
        """
        allowed_casts = {
            torch.float,
            torch.int32,
            torch.bool,
            torch.int8,
            torch.float16,
        }

        # Validate input node has convertible kwargs
        if "dtype" in to_copy_node.kwargs:
            if to_copy_node.kwargs["dtype"] in allowed_casts:
                return True
            else:
                _LOGGER.debug(
                    f"_to_copy converter rejected node {to_copy_node} with dtype {to_copy_node.kwargs['dtype']}"
                )
                return False
        else:
            _LOGGER.debug(
                f"_to_copy converter rejected node {to_copy_node} with kwargs {to_copy_node.kwargs}"
            )
            return False

    def validator(to_copy_node: Node) -> bool:
        """Returns true if the to_copy node can be converted to TRT
        and the placeholder restriction is satisfied
        """
        # The placeholder restriction is satsfied if placeholder_only is the same
        # truth value as is_only_operator_on_placeholder(to_copy_node)
        return validate_dtype(to_copy_node) and (
            (not placeholder_only) ^ is_only_operator_on_placeholder(to_copy_node)
        )

    return validator


@dynamo_tensorrt_converter(
    torch.ops.aten.clone.default,
    capability_validator=lambda node: not is_only_operator_on_placeholder(node),
)
@dynamo_tensorrt_converter(
    torch.ops.aten._to_copy.default,
    capability_validator=to_copy_dtype_validator(placeholder_only=False),
)
def aten_ops_clone_copy_dtype(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.cast.to_copy(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        kwargs.get("dtype", args[0].dtype),
        force_layer=True,
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.clone.default,
    capability_validator=is_only_operator_on_placeholder,
)
@dynamo_tensorrt_converter(
    torch.ops.aten._to_copy.default,
    capability_validator=to_copy_dtype_validator(placeholder_only=True),
)
def aten_ops_clone_copy_placeholder(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    # For clone or copy nodes where the input is also the output,
    # we need to force cast to ensure a layer is added to the TRT engine
    # since TRT engine inputs cannot also be TRT engine outputs
    return impl.cast.to_copy(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        kwargs.get("dtype", args[0].dtype),
        force_layer=True,
    )


@dynamo_tensorrt_converter(torch.ops.aten.expand.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_expand(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.slice.expand(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.amax.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_amax(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.reduce.amax(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, replacement=[]),
        args_bounds_check(args, 2, replacement=False),
    )


@dynamo_tensorrt_converter(torch.ops.aten.amin.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_amin(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.reduce.amin(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, replacement=[]),
        args_bounds_check(args, 2, replacement=False),
    )


@dynamo_tensorrt_converter(torch.ops.aten.sum.default)
@dynamo_tensorrt_converter(torch.ops.aten.sum.dim_IntList)
@dynamo_tensorrt_converter(torch.ops.prims.sum.default)
def aten_ops_sum(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    sum_ = impl.reduce.sum(
        ctx,
        target,
        get_ir(target),
        name,
        args[0],
        args_bounds_check(args, 1, replacement=None),
        args_bounds_check(args, 2, replacement=False),
    )

    if kwargs.get("output_dtype", None) is not None:
        return impl.cast.to_copy(
            ctx,
            target,
            SourceIR.ATEN,
            name,
            sum_,
            kwargs["output_dtype"],
            force_layer=True,
        )
    else:
        return sum_


@dynamo_tensorrt_converter(torch.ops.aten.prod.default)
@dynamo_tensorrt_converter(torch.ops.aten.prod.dim_int)
def aten_ops_prod(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.reduce.prod(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, replacement=None),
        args_bounds_check(args, 2, replacement=False),
    )


@dynamo_tensorrt_converter(torch.ops.aten.max.default)
@dynamo_tensorrt_converter(
    torch.ops.aten.max.dim, capability_validator=one_user_validator
)
def aten_ops_max(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.reduce.max(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        dim=args_bounds_check(args, 1, replacement=None),
        keepdim=args_bounds_check(args, 2, replacement=False),
        return_indices=(target == torch.ops.aten.max.dim),
    )


@dynamo_tensorrt_converter(torch.ops.aten.min.default)
@dynamo_tensorrt_converter(
    torch.ops.aten.min.dim, capability_validator=one_user_validator
)
def aten_ops_min(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.reduce.min(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        dim=args_bounds_check(args, 1, replacement=None),
        keepdim=args_bounds_check(args, 2, replacement=False),
        return_indices=(target == torch.ops.aten.min.dim),
    )


@dynamo_tensorrt_converter(torch.ops.aten.mean.default)
@dynamo_tensorrt_converter(torch.ops.aten.mean.dim)
def aten_ops_mean(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.reduce.mean(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, replacement=None),
        args_bounds_check(args, 2, replacement=False),
    )


@dynamo_tensorrt_converter(torch.ops.aten.exp.default)
def aten_ops_exp(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.exp(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.log.default)
def aten_ops_log(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.log(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.sqrt.default)
def aten_ops_sqrt(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.sqrt(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.reciprocal.default)
def aten_ops_recip(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.recip(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.abs.default)
def aten_ops_abs(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.abs(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.sin.default)
def aten_ops_sin(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.sin(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.cos.default)
def aten_ops_cos(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.cos(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.tan.default)
def aten_ops_tan(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.tan(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.sinh.default)
def aten_ops_sinh(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.sinh(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.cosh.default)
def aten_ops_cosh(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.cosh(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.asin.default)
def aten_ops_asin(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.asin(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.acos.default)
def aten_ops_acos(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.acos(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.atan.default)
def aten_ops_atan(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.atan(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.asinh.default)
def aten_ops_asinh(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.asinh(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.acosh.default)
def aten_ops_acosh(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.acosh(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.atanh.default)
def aten_ops_atanh(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.atanh(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.ceil.default)
def aten_ops_ceil(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.ceil(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.floor.default)
def aten_ops_floor(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.floor(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.logical_not.default)
def aten_ops_logical_not(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.logical_not(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.sign.default)
def aten_ops_sign(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.sign(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.round.default)
def aten_ops_round(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.round(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.isinf.default)
def aten_ops_isinf(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.isinf(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.add.Tensor)
@dynamo_tensorrt_converter(torch.ops.aten.add.Scalar)
def aten_ops_add(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    other = args[1]
    alpha = kwargs.get("alpha", 1)

    if alpha != 1:
        other = impl.elementwise.mul(
            ctx,
            target,
            SourceIR.ATEN,
            name,
            other,
            alpha,
        )

    return impl.elementwise.add(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        other,
    )


@dynamo_tensorrt_converter(torch.ops.aten.mul.Tensor)
@dynamo_tensorrt_converter(torch.ops.aten.mul.Scalar)
def aten_ops_mul(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.mul(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.maximum.default)
def aten_ops_maximum(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.max(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.minimum.default)
def aten_ops_minimum(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.min(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.sub.Tensor)
@dynamo_tensorrt_converter(torch.ops.aten.sub.Scalar)
def aten_ops_sub(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    other = args[1]
    alpha = kwargs.get("alpha", 1)

    if alpha != 1:
        other = impl.elementwise.mul(
            ctx,
            target,
            SourceIR.ATEN,
            name,
            other,
            alpha,
        )

    return impl.elementwise.sub(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        other,
    )


@dynamo_tensorrt_converter(torch.ops.aten.div.Tensor)
@dynamo_tensorrt_converter(torch.ops.aten.div.Tensor_mode)
@dynamo_tensorrt_converter(torch.ops.aten.div.Scalar)
@dynamo_tensorrt_converter(torch.ops.aten.div.Scalar_mode)
@dynamo_tensorrt_converter(torch.ops.prims.div.default)
def aten_ops_div(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    rounding_mode = kwargs.get("rounding_mode")

    if rounding_mode is None:
        return impl.elementwise.div(
            ctx,
            target,
            get_ir(target),
            name,
            args[0],
            args[1],
        )
    elif rounding_mode == "floor":
        return impl.elementwise.floor_divide(
            ctx,
            target,
            get_ir(target),
            name,
            args[0],
            args[1],
        )
    elif rounding_mode == "trunc":
        return impl.elementwise.trunc_div(
            ctx,
            target,
            get_ir(target),
            name,
            args[0],
            args[1],
        )
    else:
        raise RuntimeError(
            f"Target {target} does not support rounding mode {rounding_mode}"
        )


@dynamo_tensorrt_converter(torch.ops.aten.pow.Tensor_Tensor)
@dynamo_tensorrt_converter(torch.ops.aten.pow.Scalar)
@dynamo_tensorrt_converter(torch.ops.aten.pow.Tensor_Scalar)
def aten_ops_pow(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.pow(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.floor_divide.default)
@dynamo_tensorrt_converter(torch.ops.aten.floor_divide.Scalar)
def aten_ops_floor_div(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.floor_divide(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.logical_and.default)
def aten_ops_logical_and(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.logical_and(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.logical_or.default)
def aten_ops_logical_or(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.logical_or(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.logical_xor.default)
def aten_ops_logical_xor(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.logical_xor(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


def bitwise_type_validator(node: Node) -> bool:
    supported_type = [torch.bool, bool]

    tensor_targets = [
        torch.ops.aten.bitwise_and.Tensor,
        torch.ops.aten.bitwise_or.Tensor,
        torch.ops.aten.bitwise_xor.Tensor,
    ]
    scalar_targets = [
        torch.ops.aten.bitwise_and.Scalar,
        torch.ops.aten.bitwise_or.Scalar,
        torch.ops.aten.bitwise_xor.Scalar,
    ]
    scalar_tensor_targets = [
        torch.ops.aten.bitwise_and.Scalar_Tensor,
        torch.ops.aten.bitwise_or.Scalar_Tensor,
        torch.ops.aten.bitwise_xor.Scalar_Tensor,
    ]

    if node.target in tensor_targets:
        lhs_val = node.args[0]
        rhs_val = node.args[1]
        lhs_meta = lhs_val.meta.get("tensor_meta")
        rhs_meta = rhs_val.meta.get("tensor_meta")
        if lhs_meta is None or rhs_meta is None:
            return False
        return lhs_meta.dtype in supported_type and rhs_meta.dtype in supported_type

    elif node.target in scalar_targets:
        lhs_val = node.args[0]
        rhs_val = node.args[1]
        lhs_meta = lhs_val.meta.get("tensor_meta")
        if lhs_meta is None:
            return False
        return lhs_meta.dtype in supported_type and isinstance(rhs_val, bool)

    elif node.target in scalar_tensor_targets:
        lhs_val = node.args[0]
        rhs_val = node.args[1]
        rhs_meta = rhs_val.meta.get("tensor_meta")
        if rhs_meta is None:
            return False
        return isinstance(lhs_val, bool) and rhs_meta.dtype in supported_type

    else:
        return False


@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_and.Tensor, capability_validator=bitwise_type_validator
)
@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_and.Scalar, capability_validator=bitwise_type_validator
)
@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_and.Scalar_Tensor,
    capability_validator=bitwise_type_validator,
)
def aten_ops_bitwise_and(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.bitwise_and(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_or.Tensor, capability_validator=bitwise_type_validator
)
@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_or.Scalar, capability_validator=bitwise_type_validator
)
@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_or.Scalar_Tensor, capability_validator=bitwise_type_validator
)
def aten_ops_bitwise_or(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.bitwise_or(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_xor.Tensor, capability_validator=bitwise_type_validator
)
@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_xor.Scalar, capability_validator=bitwise_type_validator
)
@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_xor.Scalar_Tensor,
    capability_validator=bitwise_type_validator,
)
def aten_ops_bitwise_xor(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.bitwise_xor(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


def bitwise_not_type_validator(node: Node) -> bool:
    val = node.args[0]
    val_meta = val.meta.get("tensor_meta")

    if val_meta is None:
        return False

    supported_type = [torch.bool, bool]
    return val_meta.dtype in supported_type


@dynamo_tensorrt_converter(
    torch.ops.aten.bitwise_not.default, capability_validator=bitwise_not_type_validator
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_bitwise_not(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.bitwise_not(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.eq.Tensor)
@dynamo_tensorrt_converter(torch.ops.aten.eq.Scalar)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_eq(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.eq(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.ne.Tensor)
@dynamo_tensorrt_converter(torch.ops.aten.ne.Scalar)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_ne(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.ne(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.gt.Tensor)
@dynamo_tensorrt_converter(torch.ops.aten.gt.Scalar)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_gt(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.gt(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.ge.Tensor)
@dynamo_tensorrt_converter(torch.ops.aten.ge.Scalar)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_ge(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.ge(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.lt.Tensor)
@dynamo_tensorrt_converter(torch.ops.aten.lt.Scalar)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_lt(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.lt(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.le.Tensor)
@dynamo_tensorrt_converter(torch.ops.aten.le.Scalar)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_le(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.le(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


def conv_param_validator(conv_node: Node) -> bool:
    return conv_node.args[7] in ([0], [0, 0], [0, 0, 0])


@dynamo_tensorrt_converter(
    torch.ops.aten.convolution.default, capability_validator=conv_param_validator
)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
        1: (np.ndarray, torch.Tensor, TRTTensor),
        2: (np.ndarray, torch.Tensor, TRTTensor),
    }
)
def aten_ops_convolution(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    is_transposed = args[6]
    if not is_transposed:
        return impl.conv.convNd(
            ctx,
            target,
            source_ir=SourceIR.ATEN,
            name=name,
            is_conv1d=len(args[3]) == 1,
            input=args[0],
            weight=args[1],
            bias=args_bounds_check(args, 2, None),
            stride=args[3],
            padding=args[4],
            dilation=args[5],
            groups=args[8],
        )
    else:
        return impl.deconv.deconvNd(
            ctx,
            target,
            source_ir=SourceIR.ATEN,
            name=name,
            is_deconv1d=len(args[3]) == 1,
            input=args[0],
            weight=args[1],
            bias=args_bounds_check(args, 2, None),
            stride=args[3],
            padding=args[4],
            dilation=args[5],
            groups=args[8],
        )


@dynamo_tensorrt_converter(torch.ops.aten.linear.default)
@dynamo_tensorrt_converter(torch.ops.aten.linear)
def aten_ops_linear(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.linear.linear(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        weight=args[1],
        bias=args_bounds_check(args, 2, None),
    )


def avg_pool_param_validator(pool_node: Node) -> bool:
    ceil_mode = args_bounds_check(pool_node.args, 4, False)
    divisor_override = args_bounds_check(pool_node.args, 6)

    if ceil_mode is not False:
        _LOGGER.debug(
            f"Currently we don't support specifying ceil_mode, got ceil_mode={ceil_mode}."
        )
        return False

    if divisor_override is not None:
        _LOGGER.debug(
            f"Currently we don't support divisor_override, got divisor_override={divisor_override}."
        )
        return False

    return True


# Note: AvgPool1d uses avg_pool2d as it converts to 2D first.
@dynamo_tensorrt_converter(
    torch.ops.aten.avg_pool1d.default, capability_validator=avg_pool_param_validator
)
@dynamo_tensorrt_converter(
    torch.ops.aten.avg_pool2d.default, capability_validator=avg_pool_param_validator
)
@dynamo_tensorrt_converter(
    torch.ops.aten.avg_pool3d.default, capability_validator=avg_pool_param_validator
)
def aten_ops_avg_pool(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.pool.avg_poolNd(
        ctx,
        target,
        source_ir=SourceIR.ATEN,
        name=name,
        input=args[0],
        kernel_size=args[1],
        stride=args_bounds_check(args, 2, replacement=[]),
        padding=args_bounds_check(args, 3, replacement=0),
        ceil_mode=args_bounds_check(args, 4, replacement=False),
        count_include_pad=args_bounds_check(args, 5, replacement=True),
        divisor_override=args_bounds_check(args, 6, replacement=None),
    )


def max_pool_param_validator(pool_node: Node) -> bool:
    dilation = args_bounds_check(pool_node.args, 4, 1)
    ceil_mode = args_bounds_check(pool_node.args, 5, False)

    if dilation != 1:
        _LOGGER.debug(f"Currently we don't support dilation, got dilation={dilation}.")
        return False

    if ceil_mode is not False:
        _LOGGER.debug(
            f"Currently we don't support specifying ceil_mode, got ceil_mode={ceil_mode}."
        )
        return False

    return True


# Note: MaxPool1d uses max_pool2d as it converts to 2D first.
@dynamo_tensorrt_converter(
    torch.ops.aten.max_pool1d.default, capability_validator=max_pool_param_validator
)
@dynamo_tensorrt_converter(
    torch.ops.aten.max_pool2d.default, capability_validator=max_pool_param_validator
)
@dynamo_tensorrt_converter(
    torch.ops.aten.max_pool3d.default, capability_validator=max_pool_param_validator
)
def aten_ops_max_pool(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.pool.max_poolNd(
        ctx,
        target,
        source_ir=SourceIR.ATEN,
        name=name,
        input=args[0],
        kernel_size=args[1],
        stride=args_bounds_check(args, 2, replacement=[]),
        padding=args_bounds_check(args, 3, replacement=0),
        dilation=args_bounds_check(args, 4, replacement=1),
        ceil_mode=args_bounds_check(args, 5, replacement=False),
    )


@dynamo_tensorrt_converter(
    torch.nn.functional.scaled_dot_product_attention,
)
def tensorrt_scaled_dot_product_attention(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.attention.scaled_dot_product_attention(
        ctx,
        target,
        SourceIR.TORCHTRT_LOWERED,
        name,
        args[0],
        args[1],
        args[2],
        kwargs.get("scale", None),
    )


@dynamo_tensorrt_converter(torch.ops.aten.reshape.default)
@dynamo_tensorrt_converter(torch.ops.aten.view.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_reshape(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.shuffle.reshape(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        shape=args[1],
    )


@enforce_tensor_types({0: (TRTTensor,)})
@dynamo_tensorrt_converter(torch.ops.aten.argmax.default)
def aten_ops_argmax(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.topk.argmax(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        dim=args_bounds_check(args, 1),
        keep_dim=args_bounds_check(args, 2, False),
    )


@enforce_tensor_types({0: (TRTTensor,)})
@dynamo_tensorrt_converter(torch.ops.aten.argmin.default)
def aten_ops_argmin(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.topk.argmin(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        dim=args_bounds_check(args, 1),
        keep_dim=args_bounds_check(args, 2, False),
    )


@dynamo_tensorrt_converter(torch.ops.aten.addmm.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
        1: (np.ndarray, torch.Tensor, TRTTensor),
        2: (np.ndarray, torch.Tensor, TRTTensor),
    }
)
def aten_ops_addmm(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.addmm.addmm(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
        args[2],
        beta=kwargs.get("beta", 1),
        alpha=kwargs.get("alpha", 1),
    )


@dynamo_tensorrt_converter(torch.ops.aten.constant_pad_nd.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_constant_pad(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.pad.constant_padNd(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
        args_bounds_check(args, 2, 0),
    )


@dynamo_tensorrt_converter(torch.ops.aten.reflection_pad1d.default)
@dynamo_tensorrt_converter(torch.ops.aten.reflection_pad2d.default)
@dynamo_tensorrt_converter(torch.ops.aten.reflection_pad3d.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_reflection_pad(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.pad.reflection_padNd(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.replication_pad1d.default)
@dynamo_tensorrt_converter(torch.ops.aten.replication_pad2d.default)
@dynamo_tensorrt_converter(torch.ops.aten.replication_pad3d.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_replication_pad(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.pad.replication_padNd(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten._pad_circular.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_circular_pad(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.pad.circular_padNd(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.pad.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_pad(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.pad.pad(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        pad=args[1],
        mode=args_bounds_check(args, 2, "constant"),
        value=args_bounds_check(args, 3, None),
    )


@dynamo_tensorrt_converter(torch.ops.aten.upsample_nearest2d.vec)
def upsample_nearest2d(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.upsample.upsample(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        out_shape=args_bounds_check(args, 1),
        scale_factors=args_bounds_check(args, 2),
        resize_mode="nearest",
        align_corners=False,
    )


@dynamo_tensorrt_converter(torch.ops.aten.upsample_bilinear2d.vec)
def upsample_bilinear2d(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.upsample.upsample(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        out_shape=args_bounds_check(args, 1),
        scale_factors=args_bounds_check(args, 3),
        resize_mode="bilinear",
        align_corners=args_bounds_check(args, 2),
    )


@dynamo_tensorrt_converter(torch.ops.aten.sort.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_sort(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.topk.sort(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        dim=args_bounds_check(args, 1, -1),
        descending=args_bounds_check(args, 2, False),
    )


@dynamo_tensorrt_converter(torch.ops.aten.trunc.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def aten_ops_trunc(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.trunc(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.copy.default)
@enforce_tensor_types(
    {
        1: (TRTTensor,),
    }
)
def aten_ops_copy(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    src = args[1]
    return impl.cast.to_copy(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        src,
        src.dtype,
        force_layer=True,
    )
