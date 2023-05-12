from dataclasses import dataclass
import traceback
from typing import Callable, Dict, Tuple
import torch
from torch._custom_op import custom_op
from torch.fx.node import Argument, Target
import logging

from torch_tensorrt.fx.converter_registry import tensorrt_converter
from torch_tensorrt.fx.converters import acc_ops_converters
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor

logger = logging.getLogger(__name__)


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
    # Defines a converter implementation for Autograd to use for shape analysis/propagation
    return torch.nn.functional.max_pool1d(
        *args,
        **kwargs,
    )


def maxpool1d_insertion_fn(
    gm: torch.fx.GraphModule, submodule: torch.nn.Module, node: torch.fx.Node
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


@dataclass(frozen=True)
class ModuleReplacement:
    """Class to store key functionality for module replacement"""

    # torch.ops.___ name for replacement function for module
    new_operator: torch._ops.OpOverload

    # Function taking a containing graph, a submodule, and a 'call_module' node and returning
    # a replacement node, with type 'call_function', or raising an Error if incompatibility is detected
    # Note: subgraph_insertion_fn should NOT delete nodes or recompile the graph
    subgraph_insertion_fn: Callable[
        [torch.fx.GraphModule, torch.nn.Module, torch.fx.Node], torch.fx.Node
    ]


# Dictionary mapping module to ModuleReplacement instance
MODULE_SUBSTITUTION_REGISTRY: Dict[torch.nn.Module, ModuleReplacement] = {
    torch.nn.MaxPool1d: ModuleReplacement(
        new_operator=torch.ops.tensorrt.maxpool1d,
        subgraph_insertion_fn=maxpool1d_insertion_fn,
    ),
}


def pre_aot_module_replacement(gm: torch.fx.GraphModule):
    """Perform module-level graph replacement prior to AOT tracing

    Args:
        gm: FX GraphModule to perform module replacement on
    Returns:
        torch.fx.GraphModule

    """
    # Ensure all parameters are in inference mode
    for param in gm.parameters():
        param.requires_grad = False

    # Iterate over graph nodes, extracting module calls, to check for interceptions
    for n in gm.graph.nodes:
        if n.op == "call_module":
            # Extract submodule from graph
            submodule = gm.get_submodule(n.target)

            # If submodule is a member of the substitution registry, replace it
            if type(submodule) in MODULE_SUBSTITUTION_REGISTRY:

                try:
                    replacement = MODULE_SUBSTITUTION_REGISTRY[type(submodule)]
                    op, insertion_fn = (
                        replacement.new_operator,
                        replacement.subgraph_insertion_fn,
                    )
                    logger.debug(
                        f"Replacing module of type {type(submodule)} with {op}"
                    )

                    # Insert new node prior to older node
                    with gm.graph.inserting_before(n):
                        new_node = insertion_fn(gm, submodule, n)

                    # If submodule is not a native torch.nn module, it must be manually excluded
                    # from Dynamo tracing
                    if not type(submodule).__module__.startswith("torch.nn"):
                        torch._dynamo.allowed_functions._allowed_function_ids.add(
                            id(type(submodule))
                        )

                    # Replace all original node uses and delete node
                    n.replace_all_uses_with(new_node)
                    gm.graph.eliminate_dead_code()
                    gm.recompile()

                # A module replacement can fail in the event that the specific instance of the submodule cannot
                # be replaced
                except Exception:
                    logger.debug(
                        f"Encountered the following error while replacing {type(submodule)}"
                    )
                    logger.debug(traceback.format_exc())
                    continue

    # Perform cleanup and recompilation before returning module
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm
