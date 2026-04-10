import logging
from typing import Any, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

# call_method / call_function ops that are transparent when resolving to the
# underlying source of a 0-dim tensor for .item() folding.
_IDENTITY_METHODS = frozenset({"clone", "contiguous", "detach", "to"})
_IDENTITY_FUNCTIONS = frozenset(
    {
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.contiguous.default,
    }
)


def _resolve_source(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """Walk back through identity-like ops to the underlying placeholder / get_attr.

    Returns the source node (placeholder or get_attr) if found, else None.
    """
    visited = set()
    cur = node
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        if cur.op in ("placeholder", "get_attr"):
            return cur
        if cur.op == "call_method" and cur.target in _IDENTITY_METHODS:
            if len(cur.args) >= 1 and isinstance(cur.args[0], torch.fx.Node):
                cur = cur.args[0]
                continue
        if cur.op == "call_function" and cur.target in _IDENTITY_FUNCTIONS:
            if len(cur.args) >= 1 and isinstance(cur.args[0], torch.fx.Node):
                cur = cur.args[0]
                continue
        break
    return None


def _get_actual_tensor(
    origin: torch.fx.Node,
    gm: torch.fx.GraphModule,
    sample_inputs: Optional[Sequence[Any]],
    placeholder_index: dict,
) -> Optional[torch.Tensor]:
    """Return the actual (non-fake) tensor for a placeholder or get_attr node.

    For ``get_attr`` nodes: look up the attribute on the GraphModule.
    For ``placeholder`` nodes: try (in order):
      1. ``node.meta["grapharg"].example`` — holds a weakref to the real tensor
         from dynamo's tracing context.
      2. ``sample_inputs[index]`` — works when sample_inputs are real tensors
         (``torch_tensorrt.dynamo.compile`` path).
    """
    from torch._subclasses.fake_tensor import FakeTensor

    if origin.op == "get_attr":
        try:
            obj = gm
            for part in origin.target.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, FakeTensor):
                return None
            return obj if isinstance(obj, torch.Tensor) else None
        except AttributeError:
            return None

    if origin.op == "placeholder":
        # Try grapharg.example first (torch.compile path with FakeTensor inputs).
        ga = origin.meta.get("grapharg", None)
        if ga is not None:
            try:
                example = ga.example  # TensorWeakRef → actual tensor
                if isinstance(example, torch.Tensor) and not isinstance(
                    example, FakeTensor
                ):
                    return example
            except Exception:
                pass

        # Fall back to positional sample_inputs (dynamo.compile or real-tensor path).
        if sample_inputs is not None:
            idx = placeholder_index.get(id(origin))
            if idx is not None and idx < len(sample_inputs):
                val = sample_inputs[idx]
                if isinstance(val, torch.Tensor) and not isinstance(val, FakeTensor):
                    return val

    return None


def fold_get_attr_item_calls(
    gm: torch.fx.GraphModule,
    sample_inputs: Optional[Sequence[Any]] = None,
) -> torch.fx.GraphModule:
    """Fold ``<param>.item()`` patterns into Python scalars before AOT tracing.

    ``aot_export_joint_simple`` re-traces the graph with FakeTensors and raises
    ``DataDependentOutputException`` on any ``.item()`` call — even for scalar
    model parameters such as ``variance_epsilon``, ``scaling``, or ``norm_type``
    that are genuinely constant at inference time.

    In ``torch.compile`` graphs all parameters are lifted as **placeholder**
    inputs, so we look up their actual values via ``grapharg.example`` (a weakref
    to the real tensor kept by dynamo).  In ``torch_tensorrt.dynamo.compile``
    graphs they appear as ``get_attr`` nodes and are resolved directly from the
    module.

    Both paths also handle intermediate identity ops (clone, detach, contiguous,
    to) that dynamo may insert between the parameter and the ``.item()`` call.
    """
    # Build placeholder → positional index map (for sample_inputs fallback).
    placeholder_index: dict = {}
    ph_idx = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            placeholder_index[id(node)] = ph_idx
            ph_idx += 1

    modified = False
    for node in list(gm.graph.nodes):
        if node.op != "call_method" or node.target != "item":
            continue
        if len(node.args) != 1:
            continue
        src = node.args[0]

        # Walk through identity ops to reach the underlying source.
        origin = _resolve_source(src)
        if origin is None:
            continue

        val = _get_actual_tensor(origin, gm, sample_inputs, placeholder_index)
        if val is None or val.numel() != 1:
            continue

        scalar = val.item()
        logger.debug(
            f"fold_get_attr_item_calls: folding {node.name} "
            f"({src.name}.item() via {origin.name}) → {scalar}"
        )

        # Replace every use of this node with the Python scalar.
        # FX allows Python scalars as node arguments.
        for user in list(node.users):
            user.args = _replace_in_args(user.args, node, scalar)
            user.kwargs = {k: scalar if v is node else v for k, v in user.kwargs.items()}

        gm.graph.erase_node(node)
        modified = True

    if modified:
        gm.graph.lint()
        gm.recompile()

    return gm


def _replace_in_args(args, target_node, replacement):
    """Recursively replace *target_node* with *replacement* inside args."""
    if isinstance(args, tuple):
        return tuple(_replace_in_args(a, target_node, replacement) for a in args)
    if isinstance(args, list):
        return [_replace_in_args(a, target_node, replacement) for a in args]
    if args is target_node:
        return replacement
    return args
