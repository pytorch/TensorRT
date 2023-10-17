import logging
import operator
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.fx.node import Argument, Node, Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_registry import (
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.converter_utils import (
    enforce_tensor_types,
    is_only_operator_on_placeholder,
)
from torch_tensorrt.fx.types import TRTTensor

from .converter_utils import dynamic_unsupported_with_args

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


@dynamo_tensorrt_converter(torch.ops.aten.native_batch_norm.default, capability_validator=one_user_validator)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.batch_norm.default)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.batch_norm)  # type: ignore[misc]
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.native_layer_norm.default, capability_validator=one_user_validator)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.layer_norm.default)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.layer_norm)  # type: ignore[misc]
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.native_group_norm.default, capability_validator=one_user_validator)  # type: ignore[misc]
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.group_norm.default)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.group_norm)  # type: ignore[misc]
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.cat.default)  # type: ignore[misc]
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
)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.fmod.Scalar)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.fmod.Tensor)  # type: ignore[misc]
def aten_ops_fmod(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.fmod(ctx, target, SourceIR.ATEN, name, args[0], args[1])


@dynamo_tensorrt_converter(torch.ops.aten.relu.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.sigmoid.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.index.Tensor)  # type: ignore[misc]
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.tanh.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.leaky_relu.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.elu.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.softplus.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.clip.default)  # type: ignore[misc]
def aten_ops_clip(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.clip(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        alpha=args_bounds_check(args, 1),
        beta=args_bounds_check(args, 2),
    )


@dynamo_tensorrt_converter(torch.ops.aten.hardsigmoid.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.matmul)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.mm.default)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.mv.default)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.bmm.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.rsqrt.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.neg.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.squeeze.dim)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.squeeze.dims)  # type: ignore[misc]
def aten_ops_squeeze(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.squeeze.squeeze(ctx, target, SourceIR.ATEN, name, args[0], args[1])


@dynamo_tensorrt_converter(torch.ops.aten.erf.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.unsqueeze.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten._softmax.default)  # type: ignore[misc]
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
)  # type: ignore[misc]
@dynamo_tensorrt_converter(
    torch.ops.aten.split.sizes, capability_validator=dynamic_unsupported_with_args([1])
)  # type: ignore[misc]
@dynamo_tensorrt_converter(
    torch.ops.aten.split_with_sizes.default,
    capability_validator=dynamic_unsupported_with_args([1]),
)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.where.self)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.clamp.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.select.int)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.slice.Tensor)  # type: ignore[misc]
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
        args[1],
        args[2],
        args[3],
        args_bounds_check(args, 4, replacement=1),
    )


@dynamo_tensorrt_converter(torch.ops.aten.chunk.default)  # type: ignore[misc]
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.permute.default)  # type: ignore[misc]
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)  # type: ignore[misc]
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
)  # type: ignore[misc]
@dynamo_tensorrt_converter(
    torch.ops.aten._to_copy.default,
    capability_validator=to_copy_dtype_validator(placeholder_only=False),
)  # type: ignore[misc]
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
        force_layer=False,
    )


@dynamo_tensorrt_converter(
    torch.ops.aten.clone.default,
    capability_validator=is_only_operator_on_placeholder,
)  # type: ignore[misc]
@dynamo_tensorrt_converter(
    torch.ops.aten._to_copy.default,
    capability_validator=to_copy_dtype_validator(placeholder_only=True),
)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.expand.default)  # type: ignore[misc]
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


def amax_param_validator(amax_node: Node) -> bool:
    if len(amax_node.args) < 2:
        _LOGGER.debug(
            f"At least two args input and dim should be provided, but only got {len(amax_node.args)} args."
        )
        return False

    return True


@dynamo_tensorrt_converter(
    torch.ops.aten.amax.default, capability_validator=amax_param_validator
)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.sum.default)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.sum.dim_IntList)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.prims.sum.default)  # type: ignore[misc]
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
            force_layer=False,
        )
    else:
        return sum_


@dynamo_tensorrt_converter(torch.ops.aten.prod.default)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.prod.dim_int)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.max.default)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.max.dim, capability_validator=one_user_validator)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.min.default)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.min.dim, capability_validator=one_user_validator)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.mean.default)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.mean.dim)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.exp.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.log.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.sqrt.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.reciprocal.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.abs.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.sin.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.cos.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.tan.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.sinh.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.cosh.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.asin.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.acos.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.atan.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.asinh.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.acosh.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.atanh.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.ceil.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.floor.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.logical_not.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.sign.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.round.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.isinf.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.add.Tensor)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.add.Scalar)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.mul.Tensor)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.mul.Scalar)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.maximum.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.minimum.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.sub.Tensor)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.sub.Scalar)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.div.Tensor)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.div.Tensor_mode)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.div.Scalar)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.div.Scalar_mode)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.prims.div.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.pow.Tensor_Tensor)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.pow.Scalar)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.pow.Tensor_Scalar)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.floor_divide.default)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.floor_divide.Scalar)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.logical_and.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.logical_or.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.logical_xor.default)  # type: ignore[misc]
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


@dynamo_tensorrt_converter(torch.ops.aten.eq.Tensor)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.eq.Scalar)  # type: ignore[misc]
def aten_ops_equal(
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


@dynamo_tensorrt_converter(torch.ops.aten.gt.Tensor)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.gt.Scalar)  # type: ignore[misc]
def aten_ops_greater(
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


@dynamo_tensorrt_converter(torch.ops.aten.lt.Tensor)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.lt.Scalar)  # type: ignore[misc]
def aten_ops_less(
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


def conv_param_validator(conv_node: Node) -> bool:
    return conv_node.args[7] in ([0], [0, 0], [0, 0, 0])


@dynamo_tensorrt_converter(
    torch.ops.aten.convolution.default, capability_validator=conv_param_validator
)  # type: ignore[misc]
@enforce_tensor_types(
    {
        0: (TRTTensor,),
        1: (np.ndarray, torch.Tensor, TRTTensor),
        2: (np.ndarray, torch.Tensor, TRTTensor),
    }
)  # type: ignore[misc]
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
            bias=args[2],
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
            bias=args[2],
            stride=args[3],
            padding=args[4],
            dilation=args[5],
            groups=args[8],
        )


@dynamo_tensorrt_converter(torch.ops.aten.linear.default)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.linear)  # type: ignore[misc]
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
@dynamo_tensorrt_converter(torch.ops.aten.avg_pool1d.default, capability_validator=avg_pool_param_validator)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.avg_pool2d.default, capability_validator=avg_pool_param_validator)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.avg_pool3d.default, capability_validator=avg_pool_param_validator)  # type: ignore[misc]
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
@dynamo_tensorrt_converter(torch.ops.aten.max_pool1d.default, capability_validator=max_pool_param_validator)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.max_pool2d.default, capability_validator=max_pool_param_validator)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.max_pool3d.default, capability_validator=max_pool_param_validator)  # type: ignore[misc]
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
)  # type: ignore[misc]
def tensorrt_scaled_dot_product_attention(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.attention.scaled_dot_product_attention(
        ctx, target, SourceIR.TORCHTRT_LOWERED, name, args[0], args[1], args[2]
    )


@dynamo_tensorrt_converter(torch.ops.aten.reshape.default)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.view.default)  # type: ignore[misc]
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)  # type: ignore[misc]
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


@enforce_tensor_types({0: (TRTTensor,)})  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.argmax.default)  # type: ignore[misc]
def aten_ops_argmax(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.argmax.argmax(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        dim=args_bounds_check(args, 1),
        keep_dim=args_bounds_check(args, 2, False),
    )
