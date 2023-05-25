from dataclasses import dataclass
from typing import Any, Callable, Dict
import torch
import logging


logger = logging.getLogger(__name__)


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
MODULE_SUBSTITUTION_REGISTRY: Dict[torch.nn.Module, ModuleReplacement] = dict()


def module_substitution(
    module_to_replace: torch.nn.Module,
    new_operator: torch._ops.OpOverload,
    enabled: bool = True,
) -> Callable[[Any], Any]:
    """Decorator to register subgraph insertion functions

    Args:
        module_to_replace: nn.Module to replace
        new_operator: Custom torch operator to replace with
        enabled: Whether the substitution is enabled or disabled
    Returns:
        torch.fx.GraphModule
    """

    def register_substitution(subgraph_insertion_fn):
        """Function for use if substitution is enabled"""
        module_replacement = ModuleReplacement(
            new_operator=new_operator, subgraph_insertion_fn=subgraph_insertion_fn
        )
        MODULE_SUBSTITUTION_REGISTRY[module_to_replace] = module_replacement
        return subgraph_insertion_fn

    def disable_substitution(subgraph_insertion_fn):
        """Function for use if substitution is disabled"""
        return subgraph_insertion_fn

    return register_substitution if enabled else disable_substitution


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

                    # Replace all original node uses and clean up graph
                    n.replace_all_uses_with(new_node)
                    gm.graph.eliminate_dead_code()
                    gm.recompile()

                # A module replacement can fail in the event that the specific instance of the submodule cannot
                # be replaced
                except Exception:
                    logger.debug(
                        f"Encountered error while replacing {type(submodule)}",
                        exc_info=True,
                    )
                    continue

    # Perform cleanup and recompilation before returning module
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm
