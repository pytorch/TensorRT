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
    """True if every computational node in ``gm`` (and nested cond branches) has a converter."""
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
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
