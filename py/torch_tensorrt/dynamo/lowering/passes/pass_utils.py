# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import operator
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch_tensorrt.dynamo.utils import COMPLEX_DTYPES

# When True, clean_up_graph_after_modifications only marks the graph dirty and
# skips DCE/lint/recompile. post_lowering enables this around the pass list so
# multiple cheap FX repairs share one cleanup cycle (Naren: "one iteration").
_defer_graph_cleanup: bool = False
_graph_cleanup_pending: bool = False


def _has_side_effect(node: torch.fx.Node) -> bool:
    """Whether dead-code elimination must keep this node even though nothing uses it.

    Until run_decompositions, an exported graph still calls functorch's vmap
    bookkeeping (_vmap_increment_nesting, _vmap_decrement_nesting,
    lazy_load_decompositions) only for its side effects. FX does not count those
    calls as impure, so plain dead-code elimination deletes them, and the vmap they
    bracket then fails to decompose ("tensor may have escaped from inside a
    function being vmapped").

    A block such as torch.autocast or torch.no_grad is a call of a higher order op on
    a submodule. When the only effect of the block is an in-place write, such as
    h.add_(1.0) with a view of h that is read after the block, nothing uses the call
    either.
    """
    if node.is_impure() or (
        node.op == "call_function"
        and getattr(node.target, "__module__", None) == "torch._functorch.predispatch"
    ):
        return True
    if node.op != "call_function" or not isinstance(
        node.target, torch._ops.HigherOrderOperator
    ):
        return False
    for arg in node.args:
        if not isinstance(arg, torch.fx.Node) or arg.op != "get_attr":
            continue
        sub: Any = node.graph.owning_module
        for atom in str(arg.target).split("."):
            sub = getattr(sub, atom, None)
        if isinstance(sub, torch.fx.GraphModule) and any(
            n.op == "call_function" and _has_side_effect(n) for n in sub.graph.nodes
        ):
            return True
    return False


def set_defer_graph_cleanup(enabled: bool) -> None:
    """Enable/disable deferred FX cleanup for the current post_lowering run."""
    global _defer_graph_cleanup, _graph_cleanup_pending
    _defer_graph_cleanup = enabled
    if not enabled:
        _graph_cleanup_pending = False


def flush_deferred_graph_cleanup(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Run a real cleanup if any deferred clean_up call happened."""
    global _graph_cleanup_pending
    if _graph_cleanup_pending:
        _graph_cleanup_pending = False
        return _run_graph_cleanup(gm)
    return gm


def _run_graph_cleanup(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Runs dead-code elimination, linting, and recompilation for graph, in-place"""
    gm.graph.eliminate_dead_code(is_impure_node=_has_side_effect)
    gm.graph.lint()
    gm.recompile()
    return gm


def clean_up_graph_after_modifications(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Runs dead-code elimination, linting, and recompilation for graph, in-place.

    If deferred cleanup is enabled (see ``set_defer_graph_cleanup``), records that
    a cleanup is needed and returns immediately so callers can batch mutations.
    """
    global _graph_cleanup_pending
    if _defer_graph_cleanup:
        _graph_cleanup_pending = True
        return gm
    return _run_graph_cleanup(gm)


def iter_cond_subgraphs(gm: torch.fx.GraphModule) -> List[torch.fx.GraphModule]:
    """Return GraphModules used as torch.cond true/false branches.

    Nested cond subgraphs are not flattened here; callers that recurse (e.g.
    post_lowering) will visit them when processing each returned module.
    """
    cond_op = getattr(getattr(torch.ops, "higher_order", None), "cond", None)
    if cond_op is None:
        return []
    subgraphs: List[torch.fx.GraphModule] = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target is not cond_op:
            continue
        if len(node.args) < 3:
            continue
        for arg in node.args[1:3]:
            if not isinstance(arg, torch.fx.Node) or arg.op != "get_attr":
                continue
            attr: Any = gm
            try:
                for atom in str(arg.target).split("."):
                    attr = getattr(attr, atom)
            except AttributeError:
                continue
            if isinstance(attr, torch.fx.GraphModule):
                subgraphs.append(attr)
    return subgraphs


def get_tensor_placeholders(
    gm: torch.fx.GraphModule,
) -> List[torch.fx.Node]:
    """Returns placeholder nodes of GraphModule which are torch.Tensor types"""
    # Tensor placeholders must be subclasses of torch.Tensor
    placeholders = [
        node
        for node in gm.graph.nodes
        if (
            node.op == "placeholder"
            and isinstance(node.type, type)
            and issubclass(node.type, torch.Tensor)
        )
    ]

    return placeholders


def tensors_written_by(node: torch.fx.Node) -> List[torch.fx.Node]:
    """The tensors an op writes into, such as self of add_, out of add.out and each
    tensor of the list self of _foreach_add_."""
    if not isinstance(node.target, torch._ops.OpOverload):
        return []
    written: List[Any] = []
    for i, a in enumerate(node.target._schema.arguments):
        if a.alias_info is not None and a.alias_info.is_write:
            given = node.args[i] if i < len(node.args) else node.kwargs.get(a.name)
            written += given if isinstance(given, (list, tuple)) else [given]
    return [w for w in written if isinstance(w, torch.fx.Node)]


def tensors_maybe_written_by(node: torch.fx.Node) -> List[torch.fx.Node]:
    """The tensors node can write into: those an op writes into, and every operand of
    a higher order op, such as a torch.autocast block, whose body writes in place."""
    if not isinstance(node.target, torch._ops.HigherOrderOperator):
        return tensors_written_by(node)
    bodies = [
        operator.attrgetter(a.target)(node.graph.owning_module)
        for a in node.args
        if isinstance(a, torch.fx.Node) and a.op == "get_attr"
    ]
    if any(
        tensors_maybe_written_by(n)
        for body in bodies
        if isinstance(body, torch.fx.GraphModule)
        for n in body.graph.nodes
    ):
        return [a for a in node.args if isinstance(a, torch.fx.Node)]
    return []


def find_complex_nodes(gm: torch.fx.GraphModule) -> List[torch.fx.Node]:
    complex_nodes: List[torch.fx.Node] = []
    complexNodes: Dict[str, bool] = {}
    for node in gm.graph.nodes:
        if is_node_complex(node, complexNodes):
            complex_nodes.append(node)
    return complex_nodes


def is_node_complex(node: Any, complexNodes: Dict[str, bool]) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.name in complexNodes:
        return True
    if node.op == "call_function" and node.args is not None:
        for arg in node.args:
            if isinstance(arg, int):
                continue
            elif isinstance(arg, (list, tuple)):
                for eachNode in arg:
                    if is_node_complex(eachNode, complexNodes):
                        complexNodes[node.name] = True
                        return True

            elif hasattr(arg, "meta") and "val" in arg.meta:
                if isinstance(arg.meta["val"], (list, tuple)):
                    for eachFakeTensorMeta in arg.meta["val"]:
                        if eachFakeTensorMeta.dtype in COMPLEX_DTYPES:
                            complexNodes[node.name] = True
                            return True
                elif arg.meta["val"].dtype in COMPLEX_DTYPES:
                    complexNodes[node.name] = True
                    return True
    return False


def _wait_tensor_op() -> Optional[torch._ops.OpOverload]:
    """The op that waits for a collective. A PyTorch build without distributed support,
    as some builds for Jetson are, has no collectives and no such op."""
    if not torch.distributed.is_available():
        return None
    return torch.ops._c10d_functional.wait_tensor.default


def _largest_magnitude(value: torch.Tensor) -> Optional[torch.Tensor]:
    # A bool mask cannot leave the low precision range, and abs() has no CPU kernel
    # for bool.
    if value.dtype == torch.bool or value.numel() == 0:
        return None
    # By default nan_to_num turns an overflow to inf into the largest finite value.
    try:
        magnitude = value.abs().nan_to_num_(nan=0.0, posinf=float("inf")).amax()
    except NotImplementedError:
        # float8 and the wide unsigned types lack some of these kernels, and float32
        # is close enough for a range check.
        magnitude = value.float().abs().nan_to_num_(nan=0.0, posinf=float("inf")).amax()
    # The result stays on the device. Reading it here would make the host wait for the
    # GPU at every node.
    return magnitude.double()


def trace_intermediate_node_outputs(
    gm: torch.fx.GraphModule,
    calibration_dataloader: torch.utils.data.DataLoader,
    excluded_ops: Sequence[torch.fx.node.Target] = [],
) -> Tuple[Dict[str, float], Dict[str, Optional[torch.Size]]]:
    """Trace the intermediate node outputs of a graph module.

    Args:
        gm (torch.fx.GraphModule): The graph module to trace the intermediate node outputs of.
        calibration_dataloader (torch.utils.data.DataLoader): The dataloader to use for tracing.
        excluded_ops (Set[torch.fx.node.Target]): The set of ATen ops that should be excluded from the trace. For example, `{torch.ops.higher_order.wrap_with_autocast}`. Default is an empty set.

    Returns:
        Tuple[Dict[str, float], Dict[str, Optional[torch.Size]]]: 2 dictionaries keyed by node name. The first holds the largest absolute value of the node's output in any calibration batch, where a value that overflowed to inf counts as inf. The second holds the largest size each dim of the output had in any batch, or None when batches give outputs of different ranks. For a list output, both records are of its floating-point values joined into one 1-D tensor.
    """

    largest_magnitudes: Dict[str, float] = {}
    largest_shapes: Dict[str, Optional[torch.Size]] = {}
    batch_magnitudes: Dict[str, Optional[torch.Tensor]] = {}
    wait_tensor = _wait_tensor_op()

    class IntermediateNodeTracer(torch.fx.Interpreter):  # type: ignore[misc]
        def run_node(self, n: torch.fx.Node) -> Any:
            out = super().run_node(n)
            # A collective such as all_reduce returns before it has written its
            # output, and its wait_tensor returns that output once it has. So the
            # output is measured at the wait, for the collective too. A collective
            # that returns a list, such as all_reduce_coalesced, has getitem users
            # and is still measured as it returns.
            if (
                n.op == "call_function"
                and n.target not in excluded_ops
                and not any(user.target == wait_tensor for user in n.users)
            ):
                names = [n.name]
                if n.target == wait_tensor and isinstance(n.args[0], torch.fx.Node):
                    names.append(n.args[0].name)
                value = out
                if isinstance(out, (list, tuple)) and all(
                    isinstance(t, torch.Tensor) for t in out
                ):
                    # Ops like chunk, split and unbind return a list. Its values are
                    # recorded too, so the range rule sees them at the op feeding it.
                    # Indices, like those of topk, say nothing about that op's range.
                    floats = [t.flatten() for t in out if t.is_floating_point()]
                    # A list, such as that of _foreach_add, can mix devices, and one
                    # such as float8 data with its float32 scale can mix dtypes that
                    # cat does not promote.
                    dtype = (
                        torch.float32 if len({t.dtype for t in floats}) > 1 else None
                    )
                    value = (
                        torch.cat(
                            [t.to(device=floats[0].device, dtype=dtype) for t in floats]
                        )
                        if floats
                        else None
                    )
                if not isinstance(value, torch.Tensor):
                    return out
                # The range rule needs only the largest absolute value of any batch,
                # and the depth rule the largest reduction, which another batch may
                # have. The value is measured now, because a module can update a
                # buffer in place later in the forward, as a static cache does, and an
                # output can share that buffer.
                magnitude = _largest_magnitude(value)
                for name in names:
                    batch_magnitudes[name] = magnitude
                    shape = largest_shapes.get(name, value.shape)
                    largest_shapes[name] = (
                        torch.Size(map(max, shape, value.shape))
                        if shape is not None and len(shape) == len(value.shape)
                        else None
                    )
            return out

    def read_batch_magnitudes() -> None:
        names_by_device: Dict[torch.device, List[str]] = {}
        for name, magnitude in batch_magnitudes.items():
            if magnitude is not None:
                names_by_device.setdefault(magnitude.device, []).append(name)
        values: Dict[str, float] = {}
        for names in names_by_device.values():
            stacked = torch.stack([batch_magnitudes[name] for name in names])
            values.update(zip(names, stacked.tolist()))
        for name in batch_magnitudes:
            value = values.get(name, float("-inf"))
            largest_magnitudes[name] = max(value, largest_magnitudes.get(name, value))
        batch_magnitudes.clear()

    if calibration_dataloader is not None:
        tracer = IntermediateNodeTracer(gm)
        with torch.no_grad():
            for batch in calibration_dataloader:
                tracer.run(tuple(batch))
                # The loader may write the next batch into the same memory without
                # waiting for the GPU, so this batch is read before it asks for one.
                read_batch_magnitudes()
    return largest_magnitudes, largest_shapes
