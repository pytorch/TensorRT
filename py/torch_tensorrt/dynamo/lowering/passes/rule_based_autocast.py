# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import math
import operator
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch_tensorrt._enums import dtype
from torch_tensorrt.dynamo._settings import CompilationSettings

from .nodeclassifier import NodeClassifier, NodeNameSetRule
from .pass_utils import (
    _wait_tensor_op,
    clean_up_graph_after_modifications,
    tensors_maybe_written_by,
    tensors_written_by,
)

logger = logging.getLogger(__name__)

# Ops such as index, embedding, gather and embedding_bag need these arguments as integers.
_INDEX_ARGUMENTS = ("index", "indices", "offsets")

# Arguments where a scalar is added to a tensor or bounds it, as an epsilon is.
# The eps of layer_norm and batch_norm is a float attribute, not a constant.
_EPSILON_ARGUMENTS: dict[Any, tuple[str, ...]] = {
    torch.ops.aten.add.Tensor: ("other",),
    torch.ops.aten.add.Scalar: ("other",),
    torch.ops.aten.sub.Tensor: ("other",),
    torch.ops.aten.sub.Scalar: ("other",),
    torch.ops.aten.rsub.Tensor: ("other",),
    torch.ops.aten.rsub.Scalar: ("other",),
    torch.ops.aten.clamp.default: ("min", "max"),
    torch.ops.aten.clamp.Tensor: ("min", "max"),
    torch.ops.aten.clamp_min.default: ("min",),
    torch.ops.aten.clamp_min.Tensor: ("min",),
    torch.ops.aten.clamp_max.default: ("max",),
    torch.ops.aten.clamp_max.Tensor: ("max",),
}


def is_tensor_node(n: torch.fx.Node) -> bool:
    val = n.meta.get("val", None)
    if hasattr(val, "dtype"):
        return True
    return False


def _is_given_a_dtype(n: torch.fx.Node) -> bool:
    if not isinstance(n.target, torch._ops.OpOverload):
        return False
    for i, a in enumerate(n.target._schema.arguments):
        if a.name == "dtype":
            given = n.args[i] if i < len(n.args) else n.kwargs.get("dtype")
            return given is not None
    return False


def _has_only_integer_outputs(n: torch.fx.Node) -> bool:
    tensors = [
        v for v in pytree.tree_leaves(n.meta.get("val")) if isinstance(v, torch.Tensor)
    ]
    return bool(tensors) and not any(
        t.dtype.is_floating_point or t.dtype == torch.bool for t in tensors
    )


def _has_no_floating_point_tensors(n: torch.fx.Node) -> bool:
    tensors = [
        v
        for m in (n, *n.all_input_nodes)
        for v in pytree.tree_leaves(m.meta.get("val"))
        if isinstance(v, torch.Tensor)
    ]
    return bool(tensors) and not any(t.dtype.is_floating_point for t in tensors)


def _returns_its_input(n: torch.fx.Node) -> bool:
    """Whether n returns its first input or a view of it, as select, view(dtype) and
    fill_ do."""
    if not isinstance(n.target, torch._ops.OpOverload):
        return False
    schema = n.target._schema
    return bool(schema.arguments and schema.returns) and (
        schema.arguments[0].alias_info is not None
        and schema.returns[0].alias_info is not None
    )


def _sharing_memory(n: torch.fx.Node) -> list[torch.fx.Node]:
    """The tensor whose memory n uses, then its views and the in-place ops on it, such
    as add_ or _foreach_add_. A block, such as torch.no_grad, that may write into it
    counts as an in-place op, and its outputs as views, since it can return what it
    wrote."""
    owner = n
    while _returns_its_input(owner) and isinstance(owner.args[0], torch.fx.Node):
        owner = owner.args[0]
    nodes, i = [owner], 0
    while i < len(nodes):
        for user in nodes[i].users:
            if user not in nodes and (
                (user.args and user.args[0] is nodes[i] and _returns_its_input(user))
                or nodes[i] in tensors_maybe_written_by(user)
                or isinstance(nodes[i].target, torch._ops.HigherOrderOperator)
            ):
                nodes.append(user)
        i += 1
    return nodes


def _retype_tensors(val: Any, target_dtype: torch.dtype) -> Any:
    # Ops like chunk, split and unbind produce a list of tensors. A comparison
    # still makes bool, and topk still makes integer indices.
    return pytree.tree_map_only(
        torch.Tensor,
        lambda t: t.to(target_dtype) if t.is_floating_point() else t,
        val,
    )


def _keep_tiny_constants_nonzero(
    node: torch.fx.Node, target_dtype: torch.dtype
) -> None:
    # An epsilon such as the 1e-12 that F.normalize passes to clamp_min becomes
    # a constant of the input's type. In float16 it rounds to 0, and then
    # x / max(norm, eps) gives 0 / 0 for a zero vector, so use the smallest
    # nonzero value of the type instead, as ModelOpt AutoCast does. Only where a
    # scalar is added or is a bound: x * 1e-12 must still round to 0.
    names = _EPSILON_ARGUMENTS.get(node.target)
    if not names or not isinstance(node.target, torch._ops.OpOverload):
        return
    schema = [a.name for a in node.target._schema.arguments]
    finfo = torch.finfo(target_dtype)
    smallest = finfo.tiny * finfo.eps

    def keep(v: Any) -> Any:
        if isinstance(v, float) and 0 < abs(v) < smallest:
            return math.copysign(smallest, v)
        return v

    node.args = tuple(
        keep(v) if i < len(schema) and schema[i] in names else v
        for i, v in enumerate(node.args)
    )
    node.kwargs = {k: keep(v) if k in names else v for k, v in node.kwargs.items()}


def rule_based_autocast(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Rule-based autocast"""
    if not settings.enable_autocast:
        logger.debug("Autocast is not enabled, skipping rule-based autocast.")
        return gm

    # get config from settings
    autocast_low_precision_type = settings.autocast_low_precision_type
    if autocast_low_precision_type is None:
        return gm
    if isinstance(autocast_low_precision_type, dtype):
        autocast_low_precision_type = autocast_low_precision_type.to(torch.dtype)
    autocast_high_precision_type = torch.float32
    autocast_excluded_nodes = settings.autocast_excluded_nodes
    autocast_excluded_ops = settings.autocast_excluded_ops
    autocast_max_output_threshold = settings.autocast_max_output_threshold
    autocast_max_depth_of_reduction = settings.autocast_max_depth_of_reduction
    # pre_export_lowering sets this on the settings; a caller that skips it has none.
    reference_data: dict[str, float]
    reference_shapes: dict[str, torch.Size]
    reference_data, reference_shapes = vars(settings).pop(
        "autocast_calibration_trace", ({}, {})
    )
    recorded_nodes = settings._autocast_calibrated_high_precision_nodes
    replay_rule = None
    if not reference_data and recorded_nodes is not None:
        missing = recorded_nodes - {n.name for n in gm.graph.nodes}
        if missing:
            raise RuntimeError(
                "Autocast cannot reuse the precisions it chose with calibration data, "
                f"because these nodes are not in the graph: {sorted(missing)}. "
                "Compile the model again with autocast_calibration_dataloader."
            )
        replay_rule = NodeNameSetRule(recorded_nodes)
    elif not reference_data:
        skipped, needs = "autocast_max_output_threshold is", "it needs"
        if autocast_max_depth_of_reduction is not None:
            skipped = (
                "autocast_max_output_threshold and autocast_max_depth_of_reduction are"
            )
            needs = "they need"
        logger.warning(
            f"{skipped} not applied, because {needs} reference values from "
            "autocast_calibration_dataloader and none was given. Nodes with large "
            f"values may then run in {autocast_low_precision_type}."
        )

    node_classifier = NodeClassifier(
        gm.graph.nodes,
        excluded_nodes=autocast_excluded_nodes,
        excluded_node_names=settings.autocast_excluded_node_names,
        excluded_ops=autocast_excluded_ops,
        max_output_threshold=autocast_max_output_threshold,
        max_depth_of_reduction=autocast_max_depth_of_reduction,
        custom_rule=replay_rule,
    )
    low_precision_nodes, high_precision_nodes = node_classifier.run(
        reference_data, reference_shapes
    )
    if reference_data:
        # A saved program drops the calibration data, so refit reuses this.
        settings._autocast_calibrated_high_precision_nodes = frozenset(
            high_precision_nodes
        )

    def _cast_all_tensor_args_to_dtype(
        node: torch.fx.Node, arg: Any, dtype: torch.dtype
    ) -> Any:
        """Cast all tensor args to the given dtype

        Args:
            node: The node to insert the cast before
            arg: The argument to cast
            dtype: The dtype to cast to

        Returns:
            The casted argument
        """
        if isinstance(arg, torch.fx.Node) and is_tensor_node(arg):
            val = arg.meta.get("val", None)
            if isinstance(val, torch.Tensor):
                # A mask such as the condition of torch.where must stay bool.
                if val.dtype == dtype or val.dtype == torch.bool:
                    return arg
                else:
                    with gm.graph.inserting_before(node):
                        cast = gm.graph.call_function(
                            torch.ops.aten.to.dtype, args=(arg, dtype)
                        )
                    # copy the meta of the original tensor to the casted tensor
                    cast.meta.update(arg.meta)
                    # update the dtype of the casted tensor
                    cast.meta["val"] = cast.meta["val"].to(dtype)
                    return cast
        elif isinstance(arg, (tuple, list)):
            return type(arg)(
                _cast_all_tensor_args_to_dtype(node, a, dtype) for a in arg
            )
        elif isinstance(arg, dict):
            return {
                k: _cast_all_tensor_args_to_dtype(node, v, dtype)
                for k, v in arg.items()
            }
        else:
            return arg

    def _cast_tensor_args_to_dtype(node: torch.fx.Node, dtype: torch.dtype) -> None:
        names = (
            [a.name for a in node.target._schema.arguments]
            if isinstance(node.target, torch._ops.OpOverload)
            else []
        )
        node.args = tuple(
            (
                arg
                if i < len(names) and names[i] in _INDEX_ARGUMENTS
                else _cast_all_tensor_args_to_dtype(node, arg, dtype)
            )
            for i, arg in enumerate(node.args)
        )
        node.kwargs = {
            k: (
                v
                if k in _INDEX_ARGUMENTS
                else _cast_all_tensor_args_to_dtype(node, v, dtype)
            )
            for k, v in node.kwargs.items()
        }

    traced_dtypes = {
        n: n.meta["val"].dtype
        for n in gm.graph.nodes
        if isinstance(n.meta.get("val"), torch.Tensor)
    }

    def _cast_args_to_traced_dtypes(node: torch.fx.Node) -> None:
        node.args = tuple(
            (
                _cast_all_tensor_args_to_dtype(node, a, traced_dtypes[a])
                if isinstance(a, torch.fx.Node) and a in traced_dtypes
                else a
            )
            for a in node.args
        )

    # view(dtype) shares the memory of its input. When an in-place op, or a block such
    # as torch.autocast or torch.no_grad whose body writes in place, writes into that
    # memory, a cast copy of the input would miss the write, so the tensor that owns the
    # memory is made in its traced dtype, and its views and in-place ops keep theirs. An
    # in-place op such as index_copy_ gets its source in the dtype it writes, and such a
    # block gets its inputs in the dtypes they were traced with.
    kept: set[torch.fx.Node] = set()
    for n in gm.graph.nodes:
        if n.target == torch.ops.aten.view.dtype:
            nodes = _sharing_memory(n)
            if any(
                m in tensors_maybe_written_by(user) for m in nodes for user in m.users
            ):
                kept.update(nodes)
    owners = {n for n in kept if not _returns_its_input(n) and n in traced_dtypes}
    wait_tensor = _wait_tensor_op()
    for node in list(gm.graph.nodes):
        if node.op == "call_function":
            if node in kept and node not in owners:
                if isinstance(node.target, torch._ops.HigherOrderOperator):
                    _cast_args_to_traced_dtypes(node)
                elif node.args[0] in tensors_written_by(node):
                    _cast_tensor_args_to_dtype(node, traced_dtypes[node])
                elif isinstance(node.args[0], (list, tuple)):
                    # A list writer, such as _foreach_lerp_, gets each source in the
                    # dtype of the tensor it updates.
                    dtypes = [traced_dtypes.get(t) for t in node.args[0]]
                    node.args = node.args[:1] + tuple(
                        (
                            type(arg)(
                                (
                                    a
                                    if d is None
                                    else _cast_all_tensor_args_to_dtype(node, a, d)
                                )
                                for a, d in zip(arg, dtypes)
                            )
                            if isinstance(arg, (list, tuple))
                            and len(arg) == len(dtypes)
                            else arg
                        )
                        for arg in node.args[1:]
                    )
                continue
            if node.target in (
                torch.ops.higher_order.wrap_with_autocast,
                torch.ops.aten.view.dtype,
            ):
                # An autocast region keeps its own precision and checks the dtypes
                # of its inputs, and view(dtype) reinterprets the bytes of its
                # input, so give both the inputs they were traced with.
                _cast_args_to_traced_dtypes(node)
                continue
            if node.target == operator.getitem:
                # Follow the retyped list, so users of this element cast it correctly.
                src, idx = node.args
                if isinstance(src.meta.get("val"), (list, tuple)):
                    node.meta["val"] = src.meta["val"][idx]
                continue
            if node.target == wait_tensor:
                # A cast before the wait would read the output of a collective before
                # the collective has written it, so the wait keeps its input's type.
                src = node.args[0]
                if isinstance(src.meta.get("val"), torch.Tensor):
                    node.meta["val"] = src.meta["val"]
                continue
            if not any(is_tensor_node(n) for n in node.all_input_nodes):
                # A factory op such as torch.zeros has no inputs to cast, so its
                # output keeps the dtype it was created with.
                continue
            if _is_given_a_dtype(node):
                # An op such as x.float() or cumsum(dtype=torch.float32) makes the
                # dtype it is given, whatever its inputs are.
                continue
            if _has_only_integer_outputs(node):
                # An integer op such as ids % k must still make integers, for
                # example when its result is used as an index.
                continue
            if _has_no_floating_point_tensors(node):
                # A comparison such as ids != pad_id must compare the integers
                # themselves; float16 rounds ids above 2048.
                continue

            if node in owners:
                target_dtype = traced_dtypes[node]
            elif node.name in low_precision_nodes:
                target_dtype = autocast_low_precision_type
            elif node.name in high_precision_nodes:
                target_dtype = autocast_high_precision_type
            else:
                continue
            _cast_tensor_args_to_dtype(node, target_dtype)
            if target_dtype == autocast_low_precision_type:
                _keep_tiny_constants_nonzero(node, target_dtype)
            if any(
                getattr(n.meta.get("val"), "dtype", None) == target_dtype
                for n in node.all_input_nodes
            ):
                # With only bool inputs, mask * 0.5 still makes float32.
                node.meta["val"] = _retype_tensors(node.meta["val"], target_dtype)

    gm = clean_up_graph_after_modifications(gm)
    logger.debug("Graph after Autocast based on the rules:\n%s", gm.graph)

    return gm
