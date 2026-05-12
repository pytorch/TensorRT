import logging
from typing import Any, Dict, List, Set

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.constant_folding import constant_fold
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def _get_impure_targets() -> Set[torch._ops.OpOverload]:
    """Targets that must not be duplicated or treated as constant.

    Kept in sync with ``_TorchTensorRTConstantFolder.quantization_ops``.
    """
    impure: Set[torch._ops.OpOverload] = set()
    try:
        import modelopt.torch.quantization as mtq  # noqa: F401

        impure.add(torch.ops.tensorrt.quantize_op.default)
        impure.add(torch.ops.tensorrt.dynamic_block_quantize_op.default)
    except Exception:
        pass
    return impure


def _compute_constant_nodes(
    gm: torch.fx.GraphModule, impure_targets: Set[torch._ops.OpOverload]
) -> Set[torch.fx.Node]:
    """Set of nodes whose value is fully determined by ``get_attr`` ancestors.

    A node is constant if it is a ``get_attr`` or a pure ``call_function`` whose
    every input node is itself constant. Graph iteration is topological, so
    inputs are classified before their users.
    """
    constant_nodes: Set[torch.fx.Node] = set()
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            constant_nodes.add(node)
            continue
        if node.op != "call_function":
            continue
        if node.target in impure_targets:
            continue
        if all(inp in constant_nodes for inp in node.all_input_nodes):
            constant_nodes.add(node)
    return constant_nodes


def _register_attr_copy(gm: torch.fx.GraphModule, src_target: str) -> str:
    """Register a fresh copy of an attribute (parameter or buffer) on ``gm``.

    Returns the qualified name of the new attribute. The new attribute holds an
    independent tensor that is a ``clone`` of the source, so each duplicate
    can be specialized (or folded into a different downstream constant) without
    aliasing back to the original.
    """
    src = getattr(gm, src_target)
    idx = 0
    while True:
        new_target = f"{src_target}_dup{idx}"
        if not hasattr(gm, new_target):
            break
        idx += 1
    if isinstance(src, torch.nn.Parameter):
        copy = torch.nn.Parameter(src.detach().clone(), requires_grad=src.requires_grad)
        gm.register_parameter(new_target, copy)
    else:
        gm.register_buffer(new_target, src.detach().clone())
    return new_target


def _clone_constant_subgraph(
    gm: torch.fx.GraphModule,
    root: torch.fx.Node,
    insert_before: torch.fx.Node,
    constant_nodes: Set[torch.fx.Node],
    memo: Dict[torch.fx.Node, torch.fx.Node],
) -> torch.fx.Node:
    """Recursively clone the constant subgraph rooted at ``root``.

    All clones are inserted immediately before ``insert_before``. ``memo`` keeps
    diamond-shaped constant subgraphs coherent within a single duplication
    (e.g. ``mul(p, p)`` where the same constant feeds both args).
    """
    if root in memo:
        return memo[root]

    def _map(arg: Any) -> Any:
        if isinstance(arg, torch.fx.Node) and arg in constant_nodes:
            return _clone_constant_subgraph(
                gm, arg, insert_before, constant_nodes, memo
            )
        if isinstance(arg, (list, tuple)):
            return type(arg)(_map(a) for a in arg)
        if isinstance(arg, dict):
            return {k: _map(v) for k, v in arg.items()}
        return arg

    with gm.graph.inserting_before(insert_before):
        if root.op == "get_attr":
            new_target = _register_attr_copy(gm, root.target)
            new_node = gm.graph.get_attr(new_target)
        elif root.op == "call_function":
            new_args = tuple(_map(a) for a in root.args)
            new_kwargs = {k: _map(v) for k, v in root.kwargs.items()}
            new_node = gm.graph.call_function(root.target, new_args, new_kwargs)
        else:
            return root

    # Carry over every meta entry — ``val`` (FakeTensor with shape / dtype /
    # SymInts bound to the existing ShapeEnv), ``tensor_meta``,
    # ``unbacked_bindings``, ``stack_trace``, ``nn_module_stack`` etc. The
    # clone has identical semantics to ``root`` so the same metadata applies;
    # ``FakeTensorUpdater`` re-fakes call_function clones at end-of-lowering
    # which keeps shape-env bindings consistent for nodes whose recomputed
    # value would differ. get_attr clones are not re-faked, but the copy
    # describes the cloned parameter exactly (we ``detach().clone()`` it).
    new_node.meta.update(root.meta)
    memo[root] = new_node
    # The clone is constant by construction; register it so later candidate
    # iterations (whose consumers may have been rewired onto this clone)
    # recognise it as constant and clone it again instead of resharing.
    constant_nodes.add(new_node)
    return new_node


def constant_duplication(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Duplicate constant subgraphs with multiple users so that subsequent
    constant folding can fold each copy into its dedicated consumer.

    Given a constant node ``A`` with users ``B`` and ``C``, this pass produces
    ``A_b`` (used only by ``B``) and ``A_c`` (used only by ``C``) and re-runs
    ``constant_fold`` so each clone can be folded into its consumer's
    subgraph.

    No-op unless ``settings.constant_duplication`` is True.
    """
    if not getattr(settings, "constant_duplication", False):
        return gm

    impure_targets = _get_impure_targets()
    constant_nodes = _compute_constant_nodes(gm, impure_targets)

    candidates: List[torch.fx.Node] = [
        n for n in list(gm.graph.nodes) if n in constant_nodes and len(n.users) > 1
    ]

    duplications = 0
    for node in candidates:
        users = list(node.users.keys())
        # Leave the first user attached to the original chain; clone the
        # subgraph once per additional user.
        for user in users[1:]:
            memo: Dict[torch.fx.Node, torch.fx.Node] = {}
            new_root = _clone_constant_subgraph(gm, node, user, constant_nodes, memo)
            user.replace_input_with(node, new_root)
            duplications += 1

    if duplications == 0:
        return gm

    logger.debug(f"constant_duplication cloned {duplications} constant subgraph use(s)")

    gm = clean_up_graph_after_modifications(gm)
    gm = constant_fold(gm, settings)
    logger.debug(f"Graph after constant_duplication:\n{gm.graph}")
    return gm
