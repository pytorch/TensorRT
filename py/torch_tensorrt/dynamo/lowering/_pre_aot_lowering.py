from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

import torch
from torch._ops import OpOverload
from torch.fx import GraphModule, Node
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

SubgraphInsertionFnType: TypeAlias = Callable[
    [GraphModule, Node, Optional[torch.nn.Module]], Node
]


@dataclass(frozen=True)
class Substitution:
    """Class to store key functionality for module replacement"""

    # torch.ops.___ name for replacement function for module
    new_operator: torch._ops.OpOverload

    # Function taking a containing graph, a node, and optionally a submodule (if replacing a module)
    # and returning a replacement node, with type 'call_function', or raising an Error if
    # incompatibility is detected
    # Note: subgraph_insertion_fn should NOT delete nodes or recompile the graph
    subgraph_insertion_fn: SubgraphInsertionFnType


# Dictionary mapping module to Substitution instance
SUBSTITUTION_REGISTRY: Dict[
    (Type[torch.nn.Module] | Callable[..., Any]), Substitution
] = dict()


def register_substitution(
    module_or_function_to_replace: (Type[torch.nn.Module] | Callable[..., Any]),
    new_operator: OpOverload,
    enabled: bool = True,
) -> Callable[[SubgraphInsertionFnType], SubgraphInsertionFnType]:
    """Decorator to register subgraph insertion functions

    Args:
        module_or_function_to_replace: nn.Module or node target Callable to replace
        new_operator: Custom torch operator to replace with
        enabled: Whether the substitution is enabled or disabled
    Returns:
        torch.fx.GraphModule
    """

    def enable_substitution(
        subgraph_insertion_fn: SubgraphInsertionFnType,
    ) -> SubgraphInsertionFnType:
        """Function for use if substitution is enabled"""
        replacement = Substitution(
            new_operator=new_operator, subgraph_insertion_fn=subgraph_insertion_fn
        )
        SUBSTITUTION_REGISTRY[module_or_function_to_replace] = replacement
        return subgraph_insertion_fn

    def disable_substitution(
        subgraph_insertion_fn: SubgraphInsertionFnType,
    ) -> SubgraphInsertionFnType:
        """Function for use if substitution is disabled"""
        return subgraph_insertion_fn

    return enable_substitution if enabled else disable_substitution


def pre_aot_substitutions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Perform graph substitutions prior to AOT tracing

    Args:
        gm: FX GraphModule to perform substitution on
    Returns:
        torch.fx.GraphModule

    """
    logger.debug("Pre-module replacement graph:\n" + str(gm.graph))

    # Iterate over graph nodes, extracting module calls, to check for interceptions
    for n in gm.graph.nodes:
        exists_in_registry = False
        to_replace = None

        if n.op == "call_module":
            # Extract submodule from graph, validate in registry
            submodule = gm.get_submodule(n.target)
            to_replace = type(submodule)
            exists_in_registry = to_replace in SUBSTITUTION_REGISTRY
        elif n.op == "call_function":
            # Extract function from graph, validate in registry
            to_replace = n.target
            exists_in_registry = n.target in SUBSTITUTION_REGISTRY

        # If submodule/function is a member of the substitution registry, replace it
        if exists_in_registry:
            try:
                assert to_replace is not None
                replacement = SUBSTITUTION_REGISTRY[to_replace]
                op, insertion_fn = (
                    replacement.new_operator,
                    replacement.subgraph_insertion_fn,
                )
                logger.debug(f"Replacing node of type {to_replace} with {op}")

                # Insert new node prior to older node
                with gm.graph.inserting_before(n):
                    new_node = insertion_fn(
                        gm, n, submodule if n.op == "call_module" else None
                    )

                # If submodule is not a native torch.nn module, it must be manually excluded
                # from Dynamo tracing
                if n.op == "call_module" and not type(submodule).__module__.startswith(
                    "torch.nn"
                ):
                    torch._dynamo.allowed_functions._allowed_function_ids.add(
                        id(to_replace)
                    )

                # Replace all original node uses and clean up graph
                n.replace_all_uses_with(new_node)
                gm.graph.eliminate_dead_code()
                gm.graph.lint()
                gm.recompile()

            # A replacement can fail in the event that the specific instance of the submodule/function
            # cannot be replaced
            except Exception:
                logger.debug(
                    f"Encountered error while replacing {to_replace}",
                    exc_info=True,
                )
                continue

    # Perform cleanup and recompilation before returning module
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()

    logger.debug("Post-module replacement graph:\n" + str(gm.graph))

    return gm
