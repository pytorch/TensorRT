from typing import Dict, Tuple
import torch
from torch._custom_op import custom_op
from torch.fx.node import Argument, Target

from torch_tensorrt.fx.converter_registry import tensorrt_converter
from torch_tensorrt.fx.converters import acc_ops_converters
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor

from torch_tensorrt.dynamo.backend.lowering import register_substitution


@custom_op(
    "(Tensor x, int[1] kernel_size, int[1] stride=[], int[1] padding=[], int[1] dilation=[], bool ceil_mode=False) -> Tensor",
    ns="tensorrt",
)
def maxpool1d(x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    # Defines operator schema, name, namespace, and function header
    ...


@maxpool1d.impl("cpu")
@maxpool1d.impl("cuda")
def maxpool1d_generic(
    *args,
    **kwargs,
):
    # Defines a converter implementation for AOT Autograd to use for shape analysis/propagation
    return torch.nn.functional.max_pool1d(
        *args,
        **kwargs,
    )


@tensorrt_converter(torch.ops.tensorrt.maxpool1d.default)
def aten_ops_maxpool1d(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    # Defines converter replacing the default operator for this function
    kwargs_new = {
        "input": args[0],
        "kernel_size": args[1],
        "stride": args[2],
        "padding": args[3],
        "dilation": args[4],
        "ceil_mode": False if len(args) < 6 else args[5],
    }

    return acc_ops_converters.acc_ops_max_pool1d(
        network, target, None, kwargs_new, name
    )


@register_substitution(torch.nn.MaxPool1d, torch.ops.tensorrt.maxpool1d)
def maxpool1d_insertion_fn(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    submodule: torch.nn.Module,
) -> torch.fx.Node:
    # Defines insertion function for new node
    new_node = gm.graph.call_function(
        torch.ops.tensorrt.maxpool1d,
        args=node.args,
        kwargs={
            "kernel_size": submodule.kernel_size,
            "stride": submodule.stride,
            "padding": submodule.padding,
            "dilation": submodule.dilation,
            "ceil_mode": submodule.ceil_mode,
        },
    )

    return new_node
