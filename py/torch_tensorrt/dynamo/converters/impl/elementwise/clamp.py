import numpy as np
from typing import Optional
import tensorrt as trt
from torch.fx.node import Target

from torch_tensorrt.dynamo.converters import SourceIR
from torch_tensorrt.fx.utils import (
    unified_dtype_converter,
    Frameworks,
)

from torch_tensorrt.fx.converters.converter_utils import (
    set_layer_name,
    squeeze_left,
    get_trt_tensor,
)

from torch_tensorrt.fx.types import (
    TRTNetwork,
    TRTTensor,
)


def add_clamp(network, input, val, op, name):
    if not len(input.shape):
        # clamping scalar
        acc_ops_clamp_trt = get_trt_tensor(
            network,
            squeeze_left(
                np.array(
                    [val], dtype=unified_dtype_converter(input.dtype, Frameworks.NUMPY)
                )
            ),
            f"{name}_clamp_{val}",
        )
    else:
        acc_ops_clamp_shape = (1,) * len(input.shape)  # broadcast all dimensions
        acc_ops_clamp_tensor = np.full(
            acc_ops_clamp_shape,
            val,
            dtype=unified_dtype_converter(input.dtype, Frameworks.NUMPY),
        )
        acc_ops_clamp_trt = network.add_constant(
            acc_ops_clamp_shape, acc_ops_clamp_tensor
        ).get_output(0)
    layer = network.add_elementwise(input, acc_ops_clamp_trt, op)
    return layer


def clamp(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val,
    min_val=None,
    max_val=None,
) -> TRTTensor:
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"Clamp received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    if min_val is not None:
        clamp_min_layer = add_clamp(
            network, input_val, min_val, trt.ElementWiseOperation.MAX, name
        )
        set_layer_name(clamp_min_layer, target, f"{name}_clamp_min")
        input_val = clamp_min_layer.get_output(0)
    if max_val is not None:
        clamp_max_layer = add_clamp(
            network, input_val, max_val, trt.ElementWiseOperation.MIN, name
        )
        set_layer_name(clamp_max_layer, target, f"{name}_clamp_max")
        input_val = clamp_max_layer.get_output(0)

    return input_val
