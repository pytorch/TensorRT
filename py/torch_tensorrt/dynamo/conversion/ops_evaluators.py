import logging
import operator
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
from torch.fx.node import Argument, Node, Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    ConverterRegistry,
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor
from torch_tensorrt.dynamo.conversion.impl.elementwise import sub
from torch_tensorrt.fx.types import TRTTensor

_LOGGER: logging.Logger = logging.getLogger(__name__)


def getitem_validator(getitem_node: Node) -> bool:
    from torch_tensorrt.dynamo.conversion._ConverterRegistry import DYNAMO_CONVERTERS

    # Getitem nodes can only be converted if their parent node also can
    return getitem_node.args[0] in DYNAMO_CONVERTERS


# TODO: Subsequent evaluators should be registered here with their own validators
@dynamo_tensorrt_converter(
    operator.getitem,
    capability_validator=getitem_validator,
    supports_dynamic_shapes=True,
)
@dynamo_tensorrt_converter(torch.ops.aten.detach.default, supports_dynamic_shapes=True)
def generic_evaluator(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    _LOGGER.debug(
        f"Evaluating {ConverterRegistry.qualified_name_or_str(target)} on object with name: {name}"
    )
    return target(*args)


@dynamo_tensorrt_converter(
    torch.ops.aten.arange.start_step, supports_dynamic_shapes=True
)
def aten_ops_arange_start_step(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    # Case where inputs to arange are dynamic
    if np.any([isinstance(tensor, TRTTensor) for tensor in args]):
        start_rank_0 = get_trt_tensor(ctx, args[0], name + "_start_rank_0", rank=0)
        start_rank_1 = get_trt_tensor(ctx, args[0], name + "_start_rank_1", rank=1)
        end = get_trt_tensor(ctx, args[1], name + "_end", rank=0)
        if len(args) > 2:
            step = args[2]
        else:
            step = 1
        step = get_trt_tensor(ctx, step, name + "_step", rank=1)
        # Calculate shape = (end-start) / 1 (in this case)
        shape = sub(
            ctx,
            target,
            SourceIR.ATEN,
            name + "_shape",
            end,
            start_rank_1,
        )

        fill_layer = ctx.net.add_fill(shape.shape, trt.FillOperation.LINSPACE)
        fill_layer.set_input(0, shape)
        # Set start index
        fill_layer.set_input(1, start_rank_0)
        # Set delta/step
        fill_layer.set_input(2, step)
        # Set output type to INT32
        fill_layer.set_output_type(0, trt.DataType.INT32)
        return fill_layer.get_output(0)
    return np.arange(*args)
