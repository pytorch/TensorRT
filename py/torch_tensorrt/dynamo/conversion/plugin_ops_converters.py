import logging
from typing import Dict, Sequence, Tuple, Union

import torch
from torch.fx.node import Argument, Target
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    dynamo_tensorrt_converter,
)
from torch_tensorrt.fx.types import TRTTensor
from torch_tensorrt.dynamo.conversion.plugin import PluginCreator
import tensorrt as trt
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

logger = logging.getLogger(__name__)

TRT_PLUGIN_REGISTRY = trt.get_plugin_registry()

@dynamo_tensorrt_converter(torch.ops.torchtrt_ex.elementwise_add.default)
def torchtrt_ex_elementwise_add(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
): 
    # logger.debug(f"plugin stuff here2")
    # return torch.add(args)
    
    # How to retrieve a plugin if it is defined elsewhere (e.g. linked library)
    plugin_creator = PluginCreator("elementwise_add_plugin", plugin_namespace="", attrs={})
    TRT_PLUGIN_REGISTRY.register_creator(plugin_creator, "")    
    
    plugin_creator = TRT_PLUGIN_REGISTRY.get_plugin_creator(
        type="elementwise_add_plugin", version="1", plugin_namespace=""
    )
    assert plugin_creator, f"Unable to find elementwise_add_plugin creator"

    # # Pass configurations to the plugin implementation
    # field_configs = <TO BE GENERATED>
    # plugin = plugin_creator.create_plugin(name=name, field_collection=field_configs)
    # assert plugin, "Unable to create <PLUGIN_NAME>"

    # <GENERATE LINK BETWEEN PLUGIN AND INPUTS>
    #    <GET INPUTS INTO LIST>
    #    <PASS TO PLUGIN>     
    
    # return layer.get_output(0)
    field_configs = trt.PluginFieldCollection([])
    
    plugin = plugin_creator.create_plugin(name=name, field_collection=field_configs)
    assert plugin, "Unable to create CircularPaddingPlugin"
    
    # input_tensor = args[
    #     0
    # ]  # Arg 0 `torch.ops.torchtrt_ex.triton_circular_pad` is the input tensor
    # if not isinstance(input_tensor, trt.ITensor):
    #     # Freeze input tensor if not TensorRT Tensor already
    #     input_tensor = get_trt_tensor(ctx, input_tensor, f"{name}_input")
    
    lhs_dtype = None
    rhs_dtype = None
    lhs_val = args[0]
    rhs_val = args[1]
    
    lhs_val = get_trt_tensor(ctx, lhs_val, f"{name}_lhs", lhs_dtype)
    rhs_val = get_trt_tensor(ctx, rhs_val, f"{name}_rhs", rhs_dtype)

    layer = ctx.net.add_plugin_v3(
        [lhs_val, rhs_val], [], plugin
    )  # Add the plugin to the network being constructed
    # layer.name = f"automatic-{name}"
    return layer.get_output(0)


# 1. generate plugin for any pytorch op