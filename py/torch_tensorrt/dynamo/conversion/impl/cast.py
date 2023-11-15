import logging
from typing import Optional

from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterRegistry
from torch_tensorrt.dynamo.conversion.converter_utils import cast_trt_tensor
from torch_tensorrt.fx.converters.converter_utils import (
    Frameworks,
    unified_dtype_converter,
)
from torch_tensorrt.fx.types import TRTDataType, TRTTensor

LOGGER: logging.Logger = logging.getLogger(__name__)


def to_copy(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dtype: TRTDataType,
    force_layer: bool = False,
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"to_copy received input {input} that is not a TensorRT ITensor"
        )

    # If cast is forced, insert identity layer regardless of whether the dtype
    # doesn't change
    if force_layer:
        trt_dtype = unified_dtype_converter(dtype, Frameworks.TRT)
        source_ir = source_ir if source_ir is not None else SourceIR.UNKNOWN
        target_str = ConverterRegistry.qualified_name_or_str(target)
        target_name = f"{source_ir}_ops{('.' + target_str) if target_str else ''}"

        identity_layer = ctx.net.add_identity(input)
        identity_layer.set_output_type(0, trt_dtype)
        identity_layer.name = f"Forced Cast ITensor {input.name} from {input.dtype} to {trt_dtype} - [{target_name}]-[{name}]"
        return identity_layer.get_output(0)
    else:
        casted_tensor = cast_trt_tensor(ctx, input, dtype, name, target, source_ir)
        return casted_tensor
