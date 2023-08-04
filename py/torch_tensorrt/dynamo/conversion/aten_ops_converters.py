import logging
from typing import Dict, Sequence, Tuple, Union
import torch
import tensorrt as trt
from torch_tensorrt.fx.converters import acc_ops_converters
from .converter_registry import dynamo_tensorrt_converter
from torch.fx.node import Argument, Target, Node

from torch_tensorrt.fx.types import TRTNetwork, TRTTensor
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion.converter_utils import cast_trt_tensor
from torch_tensorrt.dynamo.conversion.converter_utils import cast_int_int_div_trt_tensor

_LOGGER: logging.Logger = logging.getLogger(__name__)


def args_bounds_check(args, i, replacement=None):
    return args[i] if len(args) > i else replacement


@dynamo_tensorrt_converter(torch.ops.aten.batch_norm)
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


@dynamo_tensorrt_converter(torch.ops.aten.div.default)
@dynamo_tensorrt_converter(torch.ops.aten.div.Tensor_mode)
@dynamo_tensorrt_converter(torch.ops.aten.div.Tensor)
def aten_ops_div(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    kwargs_new = {
        "input": args[0],
        "other": args[1],
    }
    # If both are TRTTensor, both are cast to float32
    if isinstance(args[0], TRTTensor) and isinstance(args[1], TRTTensor):
        kwargs_new["input"], kwargs_new["other"] = cast_int_int_div_trt_tensor(
            network,
            kwargs_new["input"],
            kwargs_new["other"],
            name,
        )
    # If one is TRTTensor, it is cast to float32
    elif isinstance(args[0], TRTTensor) and (
        kwargs_new["input"].dtype == trt.int8 or kwargs_new["input"].dtype == trt.int32
    ):
        kwargs_new["input"] = cast_trt_tensor(
            network, kwargs_new["input"], trt.float32, name
        )
    elif isinstance(args[1], TRTTensor) and (
        kwargs_new["other"].dtype == trt.int8 or kwargs_new["other"].dtype == trt.int32
    ):
        kwargs_new["other"] = cast_trt_tensor(
            network, kwargs_new["other"], trt.float32, name
        )
    rounding_mode = kwargs.get("rounding_mode")
    if rounding_mode is None:
        return acc_ops_converters.acc_ops_div(network, target, None, kwargs_new, name)
    elif rounding_mode == "floor":
        return acc_ops_converters.acc_ops_floor_div(
            network, target, None, kwargs_new, name
        )
    elif rounding_mode == "trunc":
        return impl.elementwise.trunc_div(
            network, target, SourceIR.ATEN, name, args[0], args[1]
        )
    else:
        raise RuntimeError(
            f"Target {target} does not support rounding mode {rounding_mode}"
        )


def embedding_param_validator(embedding_node: Node):

    max_norm = args_bounds_check(embedding_node.args, 2)
    norm_type = args_bounds_check(embedding_node.args, 3)
    scale_grad_by_freq = args_bounds_check(embedding_node.args, 4)
    sparse = args_bounds_check(embedding_node.args, 5)

    if max_norm is not None:
        _LOGGER.debug(
            f"Currently we don't support specifying max_norm, got {max_norm}."
        )
        return False

    if norm_type is not None and norm_type != 2.0:
        _LOGGER.debug(
            f"Currently we don't support specifying norm_type, got {norm_type}."
        )
        return False

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
        max_norm=args_bounds_check(args, 2),
        norm_type=args_bounds_check(args, 3),
        scale_grad_by_freq=args_bounds_check(args, 4),
        sparse=args_bounds_check(args, 5),
    )


@dynamo_tensorrt_converter(torch.ops.aten.fmod.Scalar)
@dynamo_tensorrt_converter(torch.ops.aten.fmod.Tensor)
def aten_ops_fmod(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.elementwise.fmod(network, target, SourceIR.ATEN, name, args[0], args[1])


@dynamo_tensorrt_converter(torch.ops.aten.gelu.default)
def aten_ops_gelu(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.activation.gelu(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
    )


@dynamo_tensorrt_converter(torch.ops.aten.matmul)
@dynamo_tensorrt_converter(torch.ops.aten.mm.default)
def aten_ops_matmul(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.matmul.matrix_multiply(
        network, target, SourceIR.ATEN, name, args[0], args[1]
    )


@dynamo_tensorrt_converter(torch.ops.aten.layer_norm.default)
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


@dynamo_tensorrt_converter(torch.ops.aten.relu.default)
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


@dynamo_tensorrt_converter(torch.ops.aten.rsqrt.default)
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


@dynamo_tensorrt_converter(torch.ops.aten.squeeze.dim)
@dynamo_tensorrt_converter(torch.ops.aten.squeeze.dims)
def aten_ops_squeeze(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.squeeze.squeeze(network, target, SourceIR.ATEN, name, args[0], args[1])


@dynamo_tensorrt_converter(torch.ops.aten.unsqueeze.default)
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


@dynamo_tensorrt_converter(torch.ops.aten._softmax.default)
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


@dynamo_tensorrt_converter(torch.ops.aten.where.self)
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


@dynamo_tensorrt_converter(torch.ops.aten.clamp.default)
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


@dynamo_tensorrt_converter(torch.ops.aten.select.int)
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


@dynamo_tensorrt_converter(torch.ops.aten.slice.Tensor)
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


@dynamo_tensorrt_converter(torch.ops.aten.permute.default)
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

@dynamo_tensorrt_converter(torch.ops.aten.amax.default)
@dynamo_tensorrt_converter(torch.ops.aten.amax.out)
def aten_ops_amax(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return impl.amax.amax(
        network,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args[1],
        args[2] if len(args) >= 3 else False,
        kwargs.get("out"),
    )
