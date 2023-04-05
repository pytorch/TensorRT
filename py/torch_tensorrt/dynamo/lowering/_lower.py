from py.torch_tensorrt.dynamo.lowering.passes import _partition
import torch

from torch_tensorrt.dynamo.lowering.passes import DefaultPassCollection, partition


def lower_module(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    partitioned_module = partition(module)
    return partitioned_module
