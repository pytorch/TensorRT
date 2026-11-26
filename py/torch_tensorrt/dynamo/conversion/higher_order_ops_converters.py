# mypy: disallow-untyped-decorators=False

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Argument, Node, Target
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS,
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.converter_utils import makes_uint8_tensor

_LOGGER = logging.getLogger(__name__)


def _fetch_attr(mod: torch.nn.Module, target: str) -> Any:
    cur: Any = mod
    for atom in target.split("."):
        cur = getattr(cur, atom)
    return cur


def _branch_modules(node: Node) -> Optional[List[torch.fx.GraphModule]]:
    """Return the true/false GraphModules captured on a higher_order.cond node."""
    gm = node.graph.owning_module
    if gm is None or len(node.args) < 3:
        return None
    branches: List[torch.fx.GraphModule] = []
    for arg in node.args[1:3]:
        if not isinstance(arg, Node) or arg.op != "get_attr":
            return None
        try:
            attr = _fetch_attr(gm, str(arg.target))
        except AttributeError:
            return None
        if not isinstance(attr, torch.fx.GraphModule):
            return None
        branches.append(attr)
    return branches


def _subgraph_is_supported(gm: torch.fx.GraphModule) -> bool:
    """True if every computational node in ``gm`` (and nested cond branches) can run in TensorRT.

    A node needs a converter, and must not make a uint8 tensor, which the partitioners
    keep in PyTorch for nodes outside a branch. A branch input must not be uint8 either.
    A branch must not return the result of a quantize, nor dequantize int8 values that
    did not come from a quantize or a constant in the branch.
    """
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            # A branch input comes from an IIfConditional input layer, and TensorRT
            # allows uint8 only at engine inputs and outputs.
            if getattr(node.meta.get("val"), "dtype", None) == torch.uint8:
                return False
            continue
        if node.op == "output":
            continue
        if node.op == "get_attr":
            try:
                attr = _fetch_attr(gm, str(node.target))
            except AttributeError:
                return False
            if isinstance(attr, torch.fx.GraphModule) and not _subgraph_is_supported(
                attr
            ):
                return False
            continue
        if node.op == "call_function":
            if node not in DYNAMO_CONVERTERS:
                _LOGGER.debug(
                    "torch.cond subgraph %s has unsupported op %s",
                    gm._get_name(),
                    node.target,
                )
                return False
            if makes_uint8_tensor(node):
                _LOGGER.debug(
                    "torch.cond subgraph %s makes a uint8 tensor in %s",
                    gm._get_name(),
                    node.target,
                )
                return False
            # The quantize converters keep integer results in float32. Unlike an engine output,
            # a branch output is not cast back to the op's dtype, and TensorRT rejects branches
            # whose outputs differ in type.
            if node.target in impl.quantize.QUANTIZED_DECOMPOSED_QUANTIZE_OPS and any(
                user.op == "output" for user in node.users
            ):
                _LOGGER.debug(
                    "torch.cond subgraph %s returns the result of %s",
                    gm._get_name(),
                    node.target,
                )
                return False
            # A dequantize reads an int8 tensor through TensorRT's dequantize layer, which
            # TensorRT rejects for int8 values that enter the conditional. A quantize in the
            # branch gives float32 instead, and a constant is read directly.
            if node.target in impl.quantize.QUANTIZED_DECOMPOSED_DEQUANTIZE_OPS:
                source = node.args[0]
                if (
                    isinstance(source, Node)
                    and source.op != "get_attr"
                    and source.target
                    not in impl.quantize.QUANTIZED_DECOMPOSED_QUANTIZE_OPS
                    and getattr(source.meta.get("val"), "dtype", None) == torch.int8
                ):
                    _LOGGER.debug(
                        "torch.cond subgraph %s dequantizes int8 values that are not from "
                        "a quantize or a constant in %s",
                        gm._get_name(),
                        node.target,
                    )
                    return False
            continue
        _LOGGER.debug(
            "torch.cond subgraph %s has unsupported node.op %s",
            gm._get_name(),
            node.op,
        )
        return False
    return True


def cond_capability_validator(
    node: Node, settings: Optional[CompilationSettings] = None
) -> bool:
    """Support cond only when both branch graphs are fully TRT-convertible."""
    del settings
    branches = _branch_modules(node)
    if not branches:
        return False
    return all(_subgraph_is_supported(branch) for branch in branches)


@dynamo_tensorrt_converter(
    torch.ops.higher_order.cond,
    capability_validator=cond_capability_validator,
    supports_dynamic_shapes=True,
)
def higher_order_ops_cond(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    del kwargs
    return impl.condition.cond(
        ctx,
        target,
        SourceIR.UNKNOWN,
        name,
        pred=args[0],
        true_fn=args[1],
        false_fn=args[2],
        operands=args[3],
    )
