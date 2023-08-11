import logging
from typing import Optional

from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion.converter_utils import cast_trt_tensor
from torch_tensorrt.fx.types import TRTDataType, TRTNetwork, TRTTensor

LOGGER: logging.Logger = logging.getLogger(__name__)


def to_copy(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dtype: TRTDataType,
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"to_copy received input {input} that is not a TensorRT ITensor"
        )

    casted_tensor = cast_trt_tensor(network, input, dtype, name, target, source_ir)
    return casted_tensor


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
