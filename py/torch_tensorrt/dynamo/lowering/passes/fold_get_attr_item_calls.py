import logging
from typing import Any, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

# Identity ops that produce an alias / copy of their single input.
# Removing them before folding means .item() nodes will see placeholder/get_attr directly.
_IDENTITY_METHODS = frozenset({"clone", "contiguous", "detach", "to"})
_IDENTITY_FUNCTIONS = frozenset(
    {
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.contiguous.default,
    }
)


def _remove_identity_ops(gm: torch.fx.GraphModule) -> bool:
    """Replace identity op nodes with their single input, in-place.

    Returns True if any nodes were removed.
    """
    modified = False
    for node in list(gm.graph.nodes):
        is_identity = (
            node.op == "call_method" and node.target in _IDENTITY_METHODS
        ) or (node.op == "call_function" and node.target in _IDENTITY_FUNCTIONS)
        if not is_identity:
            continue
        if len(node.args) < 1 or not isinstance(node.args[0], torch.fx.Node):
            continue
        node.replace_all_uses_with(node.args[0])
        gm.graph.erase_node(node)
        modified = True
    return modified


def _get_actual_tensor(
    origin: torch.fx.Node,
    gm: torch.fx.GraphModule,
    sample_inputs: Optional[Sequence[Any]],
    placeholder_index: dict[int, int],
) -> Optional[torch.Tensor]:
    """Return the actual (non-fake) tensor for a placeholder or get_attr node."""
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

    Identity ops (clone, detach, contiguous, alias) that dynamo may insert between
    the parameter and the ``.item()`` call are removed first so the fold step sees
    placeholder/get_attr directly.
    """
    # Strip identity ops so .item() inputs point directly at placeholder/get_attr.
    _remove_identity_ops(gm)

    # Build placeholder → positional index map (for sample_inputs fallback).
    placeholder_index: dict[int, int] = {}
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
        if src.op not in ("placeholder", "get_attr"):
            continue

        val = _get_actual_tensor(src, gm, sample_inputs, placeholder_index)
        if val is None or val.numel() != 1:
            continue

        scalar = val.item()
        logger.debug(
            f"fold_get_attr_item_calls: folding {node.name} "
            f"({src.name}.item()) → {scalar}"
        )

        for user in list(node.users):
            user.args = _replace_in_args(user.args, node, scalar)
            user.kwargs = {
                k: scalar if v is node else v for k, v in user.kwargs.items()
            }

        gm.graph.erase_node(node)
        modified = True

    if modified:
        gm.graph.lint()
        gm.recompile()

    return gm


def _replace_in_args(args: Any, target_node: Any, replacement: Any) -> Any:
    """Recursively replace *target_node* with *replacement* inside args."""
    if isinstance(args, tuple):
        return tuple(_replace_in_args(a, target_node, replacement) for a in args)
    if isinstance(args, list):
        return [_replace_in_args(a, target_node, replacement) for a in args]
    if args is target_node:
        return replacement
    return args
