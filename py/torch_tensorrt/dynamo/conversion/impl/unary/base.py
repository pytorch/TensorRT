from typing import Optional
import operator
from typing import Any, Callable, Dict, Optional, Sequence, Union
from functools import partial

import tensorrt as trt
from torch.fx.node import Target

from torch_tensorrt.fx.types import (
    TRTNetwork,
    TRTTensor,
)
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor, TRTUnaryOp
from typing import Union, Callable, Any, Optional

SUPPORTED_ATEN_TORCH_UNARY_OPS: Dict[
    TRTUnaryOp, Union[Callable[[Any], Any], Callable[[Any, Any], Any]]
] = {}


def get_python_op_from_trt_unary_op(
    trt_op: TRTUnaryOp,
) -> Union[Callable[[Any], Any], Callable[[Any, Any], Any]]:
    if trt_op == trt.UnaryOperation.SQRT:
        SUPPORTED_ATEN_TORCH_UNARY_OPS[trt_op] = partial(operator.pow(a, b=0.5))
        return True
    else:
        raise RuntimeError(f"{trt_op} is not supported yet!")
        return False


def convert_unary(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    operation_type: trt.UnaryOperation,
    input_val: TRTTensor,
) -> TRTTensor:
    """
    Add a TensorRT Unary layer to `network`.

    Args:
        network (TRTNetwork): TensorRT network object.
        input_val (TRTTensor): Input to the unary op. Must be a TensorRT tensor.
        op_type (trt.ElementWiseOperation): Type of the TensorRT unary operation.
        target (Target): Target of fx node.
        name (str): The name we want to assign to the created TensorRT layer.

    Returns:
        The output of TensorRT Unary layer.
    """
    if not isinstance(input_val, TRTTensor):
        if get_python_op_from_trt_unary_op(operation_type):
            SUPPORTED_ATEN_TORCH_UNARY_OPS[operation_type](input_val)
        else:
            raise RuntimeError(
                f"{operation_type} received input {input_val} that is not part "
                "of the TensorRT region!"
            )
    layer = network.add_unary(input_val, operation_type)
    set_layer_name(layer, target, name, source_ir)
    output = layer.get_output(0)
    output.name = output.name + "_" + target.__name__
    return layer.get_output(0)
