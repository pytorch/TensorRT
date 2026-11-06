# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import operator
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch_tensorrt._enums import dtype
from torch_tensorrt.dynamo._settings import CompilationSettings

from .nodeclassifier import NodeClassifier, NodeNameSetRule
from .pass_utils import clean_up_graph_after_modifications

logger = logging.getLogger(__name__)

# Ops such as index, embedding, gather and embedding_bag need these arguments as integers.
_INDEX_ARGUMENTS = ("index", "indices", "offsets")


def is_tensor_node(n: torch.fx.Node) -> bool:
    val = n.meta.get("val", None)
    if hasattr(val, "dtype"):
        return True
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


def _retype_tensors(val: Any, target_dtype: torch.dtype) -> Any:
    # Ops like chunk, split and unbind produce a list of tensors. A comparison
    # still makes bool, and topk still makes integer indices.
    return pytree.tree_map_only(
        torch.Tensor,
        lambda t: t.to(target_dtype) if t.is_floating_point() else t,
        val,
    )


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
    reference_data: dict[str, torch.Tensor] = vars(settings).pop(
        "autocast_intermediate_node_outputs", {}
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
        excluded_ops=autocast_excluded_ops,
        max_output_threshold=autocast_max_output_threshold,
        max_depth_of_reduction=autocast_max_depth_of_reduction,
        custom_rule=replay_rule,
    )
    low_precision_nodes, high_precision_nodes = node_classifier.run(reference_data)
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

    for node in list(gm.graph.nodes):
        if node.op == "call_function":
            if node.target == torch.ops.higher_order.wrap_with_autocast:
                continue
            if node.target == operator.getitem:
                # Follow the retyped list, so users of this element cast it correctly.
                src, idx = node.args
                if isinstance(src.meta.get("val"), (list, tuple)):
                    node.meta["val"] = src.meta["val"][idx]
                continue
            if not any(is_tensor_node(n) for n in node.all_input_nodes):
                # A factory op such as torch.zeros has no inputs to cast, so its
                # output keeps the dtype it was created with.
                continue
            if _has_only_integer_outputs(node):
                # An integer op such as ids % k must still make integers, for
                # example when its result is used as an index.
                continue
            if _has_no_floating_point_tensors(node):
                # A comparison such as ids != pad_id must compare the integers
                # themselves; float16 rounds ids above 2048.
                continue

            if node.name in low_precision_nodes:
                target_dtype = autocast_low_precision_type
            elif node.name in high_precision_nodes:
                target_dtype = autocast_high_precision_type
            else:
                continue
            _cast_tensor_args_to_dtype(node, target_dtype)
            if any(
                getattr(n.meta.get("val"), "dtype", None) == target_dtype
                for n in node.all_input_nodes
            ):
                # With only bool inputs, mask * 0.5 still makes float32.
                node.meta["val"] = _retype_tensors(node.meta["val"], target_dtype)

    gm = clean_up_graph_after_modifications(gm)
    logger.debug("Graph after Autocast based on the rules:\n%s", gm.graph)

    return gm
