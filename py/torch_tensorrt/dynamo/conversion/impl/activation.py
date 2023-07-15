import numpy as np
from typing import Any, Optional
import math

import tensorrt as trt
import torch
from torch.fx.node import Target

from torch_tensorrt.fx.converters.impl.activation import *
from torch_tensorrt.fx.converters.converter_utils import (
    mark_as_int8_layer,
    set_layer_name,
    get_trt_plugin,
)
from torch_tensorrt.dynamo.conversion import SourceIR

from torch_tensorrt.fx.types import (
    TRTNetwork,
    TRTTensor,
    TRTPluginFieldCollection,
)


def gelu(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    alpha: Optional[Any] = None,
):
    approximate = alpha
    if approximate is not None:
        raise RuntimeError("GeLU converter currently doesn't support fast gelu compute")
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"GELU received input {input_val} that is not part "
            "of the TensorRT region!"
        )
    if network.has_implicit_batch_dimension:
        raise RuntimeError(
            "GeLU converter currently doesn't support implicit batch dimension"
        )
    plugin_name = "CustomGeluPluginDynamic"
    # type_id 0 for float32, 1 for  float16
    type_id = trt.PluginField(
        "type_id", np.array(0, dtype=np.int32), trt.PluginFieldType.INT32
    )
    field_collection = TRTPluginFieldCollection([type_id])
    plugin_version = "1"

    plugin = get_trt_plugin(plugin_name, field_collection, plugin_version)

    layer = network.add_plugin_v2([input_val], plugin)

    def gelu_dyn_range_fn(dyn_range):
        return (
            dyn_range[0] * 0.5 * (1.0 + torch.erf(dyn_range[0] / math.sqrt(2.0)))
        ), (dyn_range[1] * 0.5 * (1.0 + torch.erf(dyn_range[0] / math.sqrt(2.0))))

    if input_val.dynamic_range is not None:
        dyn_range = gelu_dyn_range_fn(input_val.dynamic_range)
        mark_as_int8_layer(layer, dyn_range)
    set_layer_name(layer, target, name)
    return layer.get_output(0)
