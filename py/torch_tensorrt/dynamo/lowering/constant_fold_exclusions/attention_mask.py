from typing import Any, Callable, Iterable

import torch
from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor
from torch.fx.experimental.proxy_tensor import get_proxy_mode, get_proxy_slot

from ._core import (
    _mark_constant_fold_exclusion,
    register_constant_fold_exclusion_rule,
)

ATTENTION_MASK_ARANGE_RULE_ID = "attention_mask_arange"


def _is_arange_node(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and getattr(node.target, "overloadpacket", None) is torch.ops.aten.arange
    )


def _find_ancestor_nodes(
    node: torch.fx.Node,
    predicate: Callable[[torch.fx.Node], bool],
) -> list[torch.fx.Node]:
    """Find ancestors of ``node`` that satisfy ``predicate``."""
    nodes_to_visit: list[torch.fx.Node] = [node]
    visited: set[torch.fx.Node] = set()
    matching_nodes: list[torch.fx.Node] = []

    while nodes_to_visit:
        current = nodes_to_visit.pop()
        if current in visited:
            continue

        visited.add(current)
        if predicate(current):
            matching_nodes.append(current)

        nodes_to_visit.extend(current.all_input_nodes)

    return matching_nodes


def _find_attention_mask_aranges(
    mask_node: torch.fx.Node,
) -> list[torch.fx.Node]:
    """Find aranges that contribute to an attention mask."""
    return _find_ancestor_nodes(mask_node, predicate=_is_arange_node)


def exclude_attn_mask_aranges_from_constant_fold(attn_mask: torch.Tensor) -> None:
    """Mark aranges behind an attention mask while tracing a decomposition.

    ``run_decompositions`` invokes decompositions once for functionalization and
    again while proxy tracing. Only the proxy-tracing invocation has an FX graph
    on which metadata can be set.
    """
    proxy_mode = get_proxy_mode()
    if proxy_mode is None:
        return

    unwrapped_mask = mb_unwrap_functional_tensor(attn_mask)
    tracked_mask = get_proxy_slot(
        unwrapped_mask,
        proxy_mode.tracer,
        default=None,
    )
    proxy = getattr(tracked_mask, "proxy", tracked_mask)
    mask_node = getattr(proxy, "node", None)

    if isinstance(mask_node, torch.fx.Node):
        _mark_constant_fold_exclusion(
            _find_attention_mask_aranges(mask_node),
            ATTENTION_MASK_ARANGE_RULE_ID,
        )


@register_constant_fold_exclusion_rule(ATTENTION_MASK_ARANGE_RULE_ID)
def _attention_mask_arange_rule(node: torch.fx.Node) -> Iterable[torch.fx.Node]:
    """Select aranges feeding an attention mask."""
    # Every SDPA overload that takes a mask keeps it at positional index 3.
    # _scaled_dot_product_flash_attention is absent because it has no mask.
    attention_mask_args: dict[Any, tuple[int, str]] = {
        torch.ops.aten.scaled_dot_product_attention: (3, "attn_mask"),
        torch.ops.aten._scaled_dot_product_efficient_attention: (3, "attn_bias"),
        torch.ops.aten._scaled_dot_product_cudnn_attention: (3, "attn_bias"),
    }

    if node.op != "call_function":
        return ()

    overload_packet = getattr(node.target, "overloadpacket", None)
    mask_arg = attention_mask_args.get(overload_packet)
    if mask_arg is None:
        return ()

    mask_index, mask_kwarg = mask_arg
    mask = node.kwargs.get(
        mask_kwarg,
        node.args[mask_index] if len(node.args) > mask_index else None,
    )
    if not isinstance(mask, torch.fx.Node):
        return ()

    return _find_attention_mask_aranges(mask)
