import logging
import operator
from typing import Optional, Sequence

from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor

LOGGER: logging.Logger = logging.getLogger(__name__)


def getitem(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: Sequence[TRTTensor],
    index: int,
) -> TRTTensor:
    LOGGER.debug(f"Evaluating getitem on object with name: {name}")

    # Directly index the input sequence and return the value
    return operator.getitem(input, index)


def clone(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"clone received input {input} that is not a TensorRT ITensor"
        )

    LOGGER.debug(f"Evaluating clone on object with name: {name}")

    return input
