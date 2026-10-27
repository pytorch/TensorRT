import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)

# Set by constant folding on values computed in the graph body. Observable
# alias-producing operations stay in the graph, so this tag identifies storage
# that can be copied without severing an alias to caller-owned state.
FOLDED_CONSTRUCTOR_META = "folded_constructor"

# Marks the copy inserted below, so running this pass again (once per TensorRT
# submodule after partitioning) does not stack redundant copies.
_FRESH_COPY_META = "folded_constructor_reset"


def _mutates_its_input(user: torch.fx.Node) -> bool:
    target = user.target
    if not isinstance(target, torch._ops.OpOverload):
        return False
    return bool(target._schema.is_mutable)


def _aliases_input(node: torch.fx.Node) -> bool:
    target = node.target
    return isinstance(target, torch._ops.OpOverload) and any(
        result.alias_info is not None for result in target._schema.returns
    )


def _storage_escapes(node: torch.fx.Node) -> bool:
    """Check outputs and mutations, including uses reached through views."""
    pending = list(node.users)
    visited = set()
    while pending:
        user = pending.pop()
        if user in visited:
            continue
        visited.add(user)
        if user.op == "output" or _mutates_its_input(user):
            return True
        if _aliases_input(user):
            pending.extend(user.users)
    return False


def reset_folded_constructors(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Rebuild folded constructors that must not persist between calls.

    Constant folding hoists ops out of the graph body into module attributes,
    which turns per-call values into state shared by every invocation. That is
    only observable when the value escapes as an output or is mutated in place;
    a folded value that is merely read stays a genuine constant and is left as
    is, so weights keep converting to TensorRT constants.

    The copy is inserted at the point of construction rather than at the return,
    so an in-place mutation inside the graph also sees fresh storage. Every user
    reads the same copy, which preserves aliasing when a value is returned more
    than once.

    Values the caller owns, such as placeholders and pre-existing module
    attributes, are untagged and deliberately untouched: Python passes those by
    reference and mutations are expected to persist.
    """
    modified = False

    for node in list(gm.graph.nodes):
        if node.op != "get_attr" or not node.meta.get(FOLDED_CONSTRUCTOR_META):
            continue

        users = list(node.users)
        if not users or any(user.meta.get(_FRESH_COPY_META) for user in users):
            continue
        if not _storage_escapes(node):
            continue

        with gm.graph.inserting_after(node):
            fresh = gm.graph.call_function(torch.ops.aten.clone.default, args=(node,))
        fresh.meta.update(node.meta)
        fresh.meta.pop(FOLDED_CONSTRUCTOR_META, None)
        fresh.meta[_FRESH_COPY_META] = True

        for user in users:
            user.replace_input_with(node, fresh)

        modified = True

    if modified:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug("Graph after resetting folded constructors:\n%s", gm.graph)

    return gm
