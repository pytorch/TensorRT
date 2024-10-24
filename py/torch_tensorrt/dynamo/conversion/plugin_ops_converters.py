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

logger = logging.getLogger(__name__)

@dynamo_tensorrt_converter(torch.ops.torchtrt_ex.elementwise_add.default)
def torchtrt_ex_elementwise_add(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
): 
    logger.debug(f"plugin stuff here2")
    return torch.add(args)
    
    # How to retrieve a plugin if it is defined elsewhere (e.g. linked library)
    # plugin_registry = trt.get_plugin_registry()
    # plugin_creator = plugin_registry.get_plugin_creator(
    #     type="<PLUGIN_NAME>", version="1", plugin_namespace=""
    # )
    # assert plugin_creator, f"Unable to find <PLUGIN_NAME> creator"

    # # Pass configurations to the plugin implementation
    # field_configs = <TO BE GENERATED>
    # plugin = plugin_creator.create_plugin(name=name, field_collection=field_configs)
    # assert plugin, "Unable to create <PLUGIN_NAME>"

    # <GENERATE LINK BETWEEN PLUGIN AND INPUTS>
    #    <GET INPUTS INTO LIST>
    #    <PASS TO PLUGIN>     
    
    # return layer.get_output(0)


# 1. generate plugin for any pytorch op