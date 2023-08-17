import logging
import operator
from typing import Dict, Sequence, Tuple, Union

from torch.fx.node import Argument, Node, Target
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor

from .converter_registry import ConverterRegistry, dynamo_tensorrt_converter

_LOGGER: logging.Logger = logging.getLogger(__name__)


def getitem_validator(getitem_node: Node) -> bool:
    from torch_tensorrt.dynamo.conversion.converter_registry import DYNAMO_CONVERTERS

    # Getitem nodes can only be converted if their parent node also can
    return getitem_node.args[0] in DYNAMO_CONVERTERS


# TODO: Subsequent evaluators should be registered here with their own validators
@dynamo_tensorrt_converter(operator.getitem, capability_validator=getitem_validator)
def generic_evaluator(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    _LOGGER.debug(
        f"Evaluating {ConverterRegistry.qualified_name_or_str(target)} on object with name: {name}"
    )
    return target(*args)
