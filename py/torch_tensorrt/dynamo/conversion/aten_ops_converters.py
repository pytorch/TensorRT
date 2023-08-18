import logging
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from torch.fx.node import Argument, Node, Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion.converter_utils import (
    is_only_operator_on_placeholder,
)
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor

from .converter_registry import dynamo_tensorrt_converter

_LOGGER: logging.Logger = logging.getLogger(__name__)


def args_bounds_check(
    args: Tuple[Argument, ...], i: int, replacement: Optional[Any] = None
) -> Any:
    return args[i] if len(args) > i else replacement


@dynamo_tensorrt_converter(torch.ops.aten.batch_norm)  # type: ignore[misc]
def aten_ops_batch_norm(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.batch_norm(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
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
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.embedding.embedding(
        network,
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
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.fmod(network, target, SourceIR.ATEN, name, args[0], args[1])


@dynamo_tensorrt_converter(torch.ops.aten.relu.default)  # type: ignore[misc]
def aten_ops_relu(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.relu(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.sigmoid.default)  # type: ignore[misc]
def aten_ops_sigmoid(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.sigmoid(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.tanh.default)  # type: ignore[misc]
def aten_ops_tanh(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.tanh(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.leaky_relu.default)  # type: ignore[misc]
def aten_ops_leaky_relu(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.leaky_relu(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, 0.01),
    )


@dynamo_tensorrt_converter(torch.ops.aten.elu.default)  # type: ignore[misc]
def aten_ops_elu(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.elu(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        alpha=args_bounds_check(args, 1, 1.0),
        beta=args_bounds_check(args, 2, None),
    )


@dynamo_tensorrt_converter(torch.ops.aten.softplus.default)  # type: ignore[misc]
def aten_ops_softplus(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.softplus(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        beta=args_bounds_check(args, 1, 1),
    )


@dynamo_tensorrt_converter(torch.ops.aten.clip.default)  # type: ignore[misc]
def aten_ops_clip(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.clip(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        alpha=args_bounds_check(args, 1),
        beta=args_bounds_check(args, 2),
    )


@dynamo_tensorrt_converter(torch.ops.aten.hardsigmoid.default)  # type: ignore[misc]
def aten_ops_hard_sigmoid(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.hard_sigmoid(
        network,
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
def aten_ops_matmul(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.matmul.matrix_multiply(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.layer_norm.default)  # type: ignore[misc]
def aten_ops_layernorm(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.layer_norm(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
    )


@dynamo_tensorrt_converter(torch.ops.aten.rsqrt.default)  # type: ignore[misc]
def aten_ops_rsqrt(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.rsqrt(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.neg.default)  # type: ignore[misc]
def aten_ops_neg(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.neg(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.squeeze.dim)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.squeeze.dims)  # type: ignore[misc]
def aten_ops_squeeze(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.squeeze.squeeze(network, target, SourceIR.ATEN, name, args[0], args[1])


@dynamo_tensorrt_converter(torch.ops.aten.unsqueeze.default)  # type: ignore[misc]
def aten_ops_unsqueeze(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unsqueeze.unsqueeze(
        network, target, SourceIR.ATEN, name, input_t=args[0], dim=args[1]
    )


@dynamo_tensorrt_converter(torch.ops.aten._softmax.default)  # type: ignore[misc]
def aten_ops_softmax(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.normalization.softmax(
        network, target, SourceIR.ATEN, name, args[0], args[1]
    )
    
def dynamic_unsupported_split(node: torch.fx.Node) -> bool:
    # Validate that none of the inputs to the node have Dynamic shapes
    assert isinstance(
        node, torch.fx.Node
    ), "Inputs to validator functions must be FX Nodes"

    if isinstance(node.args[1], torch.fx.Node):
        if getattr(node.args[1].meta["val"], "_has_symbolic_sizes_strides", True):
            return False
    return True


@dynamo_tensorrt_converter(
    torch.ops.aten.split.Tensor, capability_validator=dynamic_unsupported_split
)
@dynamo_tensorrt_converter(
    torch.ops.aten.split.sizes, capability_validator=dynamic_unsupported_split
)
@dynamo_tensorrt_converter(
    torch.ops.aten.split_with_sizes.default,
    capability_validator=dynamic_unsupported_split,
)
def aten_ops_split(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]: 
    return impl.split.split(
        network,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        split_size_or_sections=args[1],
        dim=args_bounds_check(args, 2, 0),
    )


@dynamo_tensorrt_converter(torch.ops.aten.where.self)  # type: ignore[misc]
def aten_ops_where(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.condition.where(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[1],
        args[2],
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.clamp.default)  # type: ignore[misc]
def aten_ops_clamp(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.clamp(
        network,
        target,
        SourceIR.ATEN,
        name,
        input_val=args[0],
        min_val=args_bounds_check(args, 1),
        max_val=args_bounds_check(args, 2),
    )


@dynamo_tensorrt_converter(torch.ops.aten.select.int)  # type: ignore[misc]
def aten_ops_select(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.select.select(
        network, target, SourceIR.ATEN, name, args[0], args[1], args[2]
    )


@dynamo_tensorrt_converter(torch.ops.aten.slice.Tensor)  # type: ignore[misc]
def aten_ops_slice(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.slice.slice_op(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
        args[2],
        args[3],
        args_bounds_check(args, 4, replacement=1),
    )


@dynamo_tensorrt_converter(torch.ops.aten.permute.default)  # type: ignore[misc]
def aten_ops_permute(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.permutation.permute(
        network,
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
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.cast.to_copy(
        network,
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
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    # For clone or copy nodes where the input is also the output,
    # we need to force cast to ensure a layer is added to the TRT engine
    # since TRT engine inputs cannot also be TRT engine outputs
    return impl.cast.to_copy(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        kwargs.get("dtype", args[0].dtype),
        force_layer=True,
    )


@dynamo_tensorrt_converter(torch.ops.aten.expand.default)  # type: ignore[misc]
def aten_ops_expand(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.slice.expand(
        network,
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
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.reduce.amax(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
        args_bounds_check(args, 2, replacement=False),
    )


@dynamo_tensorrt_converter(torch.ops.aten.sum.default)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.sum.dim_IntList)  # type: ignore[misc]
def aten_ops_sum(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.reduce.sum(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, replacement=None),
        args_bounds_check(args, 2, replacement=False),
    )


@dynamo_tensorrt_converter(torch.ops.aten.exp.default)  # type: ignore[misc]
def aten_ops_exp(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.exp(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.log.default)  # type: ignore[misc]
def aten_ops_log(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.log(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.sqrt.default)  # type: ignore[misc]
def aten_ops_sqrt(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.sqrt(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.reciprocal.default)  # type: ignore[misc]
def aten_ops_recip(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.recip(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.abs.default)  # type: ignore[misc]
def aten_ops_abs(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.abs(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.sin.default)  # type: ignore[misc]
def aten_ops_sin(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.sin(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.cos.default)  # type: ignore[misc]
def aten_ops_cos(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.cos(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.tan.default)  # type: ignore[misc]
def aten_ops_tan(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.tan(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.sinh.default)  # type: ignore[misc]
def aten_ops_sinh(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.sinh(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.cosh.default)  # type: ignore[misc]
def aten_ops_cosh(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.cosh(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.asin.default)  # type: ignore[misc]
def aten_ops_asin(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.asin(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.acos.default)  # type: ignore[misc]
def aten_ops_acos(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.acos(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.atan.default)  # type: ignore[misc]
def aten_ops_atan(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.atan(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.asinh.default)  # type: ignore[misc]
def aten_ops_asinh(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.asinh(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.acosh.default)  # type: ignore[misc]
def aten_ops_acosh(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.acosh(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.atanh.default)  # type: ignore[misc]
def aten_ops_atanh(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.atanh(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.ceil.default)  # type: ignore[misc]
def aten_ops_ceil(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.ceil(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.floor.default)  # type: ignore[misc]
def aten_ops_floor(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.floor(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.logical_not.default)  # type: ignore[misc]
def aten_ops_logical_not(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.logical_not(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.sign.default)  # type: ignore[misc]
def aten_ops_sign(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.sign(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.round.default)  # type: ignore[misc]
def aten_ops_round(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.round(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.isinf.default)  # type: ignore[misc]
def aten_ops_isinf(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.unary.isinf(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.add.Tensor)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.add.Scalar)  # type: ignore[misc]
def aten_ops_add(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    other = args[1]
    alpha = kwargs.get("alpha", 1)

    if alpha != 1:
        other = impl.elementwise.mul(
            network,
            target,
            SourceIR.ATEN,
            name,
            other,
            alpha,
        )

    return impl.elementwise.add(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        other,
    )


@dynamo_tensorrt_converter(torch.ops.aten.mul.Tensor)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.mul.Scalar)  # type: ignore[misc]
def aten_ops_mul(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.mul(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.maximum.default)  # type: ignore[misc]
def aten_ops_max(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.max(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.minimum.default)  # type: ignore[misc]
def aten_ops_min(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.min(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.sub.Tensor)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.sub.Scalar)  # type: ignore[misc]
def aten_ops_sub(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    other = args[1]
    alpha = kwargs.get("alpha", 1)

    if alpha != 1:
        other = impl.elementwise.mul(
            network,
            target,
            SourceIR.ATEN,
            name,
            other,
            alpha,
        )

    return impl.elementwise.sub(
        network,
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
def aten_ops_div(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    rounding_mode = kwargs.get("rounding_mode")

    if rounding_mode is None:
        return impl.elementwise.div(
            network,
            target,
            SourceIR.ATEN,
            name,
            args[0],
            args[1],
        )
    elif rounding_mode == "floor":
        return impl.elementwise.floor_divide(
            network,
            target,
            SourceIR.ATEN,
            name,
            args[0],
            args[1],
        )
    elif rounding_mode == "trunc":
        return impl.elementwise.trunc_div(
            network,
            target,
            SourceIR.ATEN,
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
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.pow(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.floor_divide.default)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.floor_divide.Scalar)  # type: ignore[misc]
def aten_ops_floor_div(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.floor_divide(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.logical_and.default)  # type: ignore[misc]
def aten_ops_logical_and(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.logical_and(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.logical_or.default)  # type: ignore[misc]
def aten_ops_logical_or(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.logical_or(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.logical_xor.default)  # type: ignore[misc]
def aten_ops_logical_xor(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.logical_xor(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.eq.Tensor)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.eq.Scalar)  # type: ignore[misc]
def aten_ops_equal(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.eq(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.gt.Tensor)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.gt.Scalar)  # type: ignore[misc]
def aten_ops_greater(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.gt(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


@dynamo_tensorrt_converter(torch.ops.aten.lt.Tensor)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.lt.Scalar)  # type: ignore[misc]
def aten_ops_less(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.lt(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
    )


def conv_param_validator(conv_node: Node) -> bool:
    return (not conv_node.args[6]) and (conv_node.args[7] in ([0], [0, 0], [0, 0, 0]))


@dynamo_tensorrt_converter(
    torch.ops.aten.convolution.default, capability_validator=conv_param_validator
)  # type: ignore[misc]
def aten_ops_convolution(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.conv.convNd(
        network,
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


@dynamo_tensorrt_converter(torch.ops.aten.linear.default)  # type: ignore[misc]
@dynamo_tensorrt_converter(torch.ops.aten.linear)  # type: ignore[misc]
def aten_ops_linear(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.linear.linear(
        network,
        target,
        SourceIR.ATEN,
        name,
        input=args[0],
        weight=args[1],
        bias=args_bounds_check(args, 2, None),
    )
