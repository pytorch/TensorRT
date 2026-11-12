# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Borrowed from ModelOpt AutoCast's nodeclassifier.py, modified to fit Torch-TensorRT's needs.
import abc
import logging
import operator
import re
from typing import Collection, Optional

import torch
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    ConverterRegistry,
)

from .inline_autocast_regions import NAME_IN_REGION

logger = logging.getLogger(__name__)


class NodeRuleBase:
    """Base class for node classification rules.

    This class defines the interface for rules that determine whether a node
    should be kept in high precision or converted to low precision.
    """

    @abc.abstractmethod
    def _check_inner(self, node):
        """Implement this method to check if node conversion should be skipped based on rule criteria."""

    def _log_skipped(self, node, **kwargs):
        """Log information about skipped nodes."""
        logger.info(f"Skipping node {node.name}: {self.__class__.__name__}")

    def check(self, node):
        """Check if a node should be skipped based on the rule.

        Args:
            node: The torch.fx.Node to check.

        Returns:
            bool: True if the node should be kept in high precision, False otherwise.
        """
        result = self._check_inner(node)
        if result:
            self._log_skipped(node)
            return True
        return False


class DisabledNodeNameRegexRule(NodeRuleBase):
    """Rule for keeping nodes in high precision whose module path matches a user regex."""

    def __init__(self, disabled_node_name_regex):
        """Initialize the rule.

        Args:
            disabled_node_name_regex: List of regex patterns for the paths of the modules whose nodes to keep in high precision.
        """
        self.disabled_node_name_regex = disabled_node_name_regex

    def _check_inner(self, node):
        stack = node.meta.get("nn_module_stack")
        try:
            # get the user specified name of the node
            node_name = stack.get(next(reversed(stack)), [""])[0]
        except Exception as e:
            raise ValueError(
                f"Failed to get the user specified name of the node {node.name} because {e}. Please file a bug with Torch-TensorRT."
            )
        return any(
            re.match(regex, node_name) for regex in self.disabled_node_name_regex
        )


class NodeNameRegexRule(NodeRuleBase):
    """Rule for keeping nodes in high precision whose name in the graph, or in its autocast region, matches a user regex."""

    def __init__(self, node_name_regex: Collection[str]) -> None:
        """Initialize the rule.

        Args:
            node_name_regex: Collection of regex patterns for the names of the nodes to keep in high precision.
        """
        self.node_name_regex = node_name_regex

    def _check_inner(self, node: torch.fx.Node) -> bool:
        name_in_region = node.meta.get(NAME_IN_REGION, node.name)
        return any(
            re.match(regex, node.name) or re.match(regex, name_in_region)
            for regex in self.node_name_regex
        )


class NodeNameSetRule(NodeRuleBase):
    """Rule for keeping the nodes with the given exact names in high precision."""

    def __init__(self, node_names):
        """Initialize the rule.

        Args:
            node_names: Collection of node names to keep in high precision.
        """
        self.node_names = node_names

    def _check_inner(self, node):
        return node.name in self.node_names


class DisabledOpTypes(NodeRuleBase):
    """Rule for keeping nodes with specific ATen ops in high precision."""

    def __init__(self, excluded_ops):
        """Initialize the rule.

        Args:
            excluded_ops: List of ATen ops that should remain in FP32.
        """
        self.excluded_ops = excluded_ops

    def _check_inner(self, node):
        node_name = ConverterRegistry.qualified_name_or_str(node.target)
        return node_name in self.excluded_ops


class IORangeRule(NodeRuleBase):
    """Rule for keeping nodes with out-of-range inputs/outputs in high precision."""

    def __init__(self, max_output_threshold, reference_magnitudes):
        """Initialize the rule.

        Args:
            max_output_threshold: Maximum absolute value allowed for node I/O.
            reference_magnitudes: Largest absolute value of each tensor in the reference data.
        """
        self.max_output_threshold = max_output_threshold
        self.reference_magnitudes = reference_magnitudes
        self.output_magnitude = None

    def _check_inner(self, node):
        def is_io_out_of_range(node):
            tensor_name = node.name
            if tensor_name not in self.reference_magnitudes:
                logger.debug(
                    f"Node {node.name}: Tensor {tensor_name} not found in reference data. Skipping I/O range check."
                )
                return False
            magnitude = self.reference_magnitudes[tensor_name]
            logger.debug(f"Node {node.name}: largest reference magnitude {magnitude}")
            if magnitude > self.max_output_threshold:
                self.output_magnitude = magnitude
                return True
            return False

        if self.reference_magnitudes:
            for in_node in node.all_input_nodes:
                if is_io_out_of_range(in_node):
                    return True
            for out_node in list(node.users):
                if is_io_out_of_range(out_node):
                    return True
        return False

    def _log_skipped(self, node, **kwargs):
        """Log information about skipped nodes with I/O range violations."""
        if self.output_magnitude is not None:
            logger.info(
                f"Skipping node {node.name}: reference IO out of range: largest magnitude "
                f"{self.output_magnitude}, range=[{-self.max_output_threshold}, {self.max_output_threshold}]"
            )
        else:
            super()._log_skipped(node, **kwargs)


class DepthOfReductionRule(NodeRuleBase):
    """
    Rule for keeping nodes with high depth of reduction in high precision. This helps prevent excessive accuracy loss in operations particularly sensitive to reduced precision, as higher-depth reductions may amplify computation errors in low precision formats.
    Reduction ops are those that aggregate data across one or more axes, decreasing the dimensionality of the input tensor, such as convolution, gemm, etc.
    """

    def __init__(self, max_depth_of_reduction, reference_shapes):
        """Initialize the rule.

        Args:
            max_depth_of_reduction: Maximum depth of reduction allowed in low precision.
            reference_shapes: Largest size of each dim of each tensor in the reference data.
        """
        self.max_depth_of_reduction = max_depth_of_reduction
        self.reference_shapes = reference_shapes
        self.reduction_depth = 0

    def _get_tensor_shape(self, tensor_name):
        """Get the largest size of each dim of the tensor in the reference data."""
        if tensor_name in self.reference_shapes and (
            self.reference_shapes[tensor_name] is None
        ):
            raise ValueError(
                f"Calibration batches give {tensor_name} outputs of different ranks, "
                "so the depth rule cannot tell which dims it reduces."
            )
        return self.reference_shapes.get(tensor_name)

    def _log_skipped(self, node, **kwargs):
        """Log information about skipped nodes with depth of reduction violations."""
        if self.reduction_depth > 0:
            logger.info(
                f"Skipping node {node.name}: depth of reduction {self.reduction_depth} exceeds "
                f"{self.max_depth_of_reduction}."
            )
        else:
            super()._log_skipped(node, **kwargs)

    def _check_inner(self, node):
        # All reduction ops rely on shape of input[0]
        input_0_dims = (
            self._get_tensor_shape(node.all_input_nodes[0].name)
            if len(node.all_input_nodes) > 0
            else None
        )
        if input_0_dims is None:
            return False
        self.reduction_depth = 0
        if node.target in [
            torch.ops.aten.scaled_dot_product_attention.default,
        ]:
            # Attention: input (batch_size, sequence_length, hidden_size)
            # or (batch_size, kv_num_heads, total_sequence_length, head_size)
            assert len(input_0_dims) == 3 or len(input_0_dims) == 4
            hidden_size = (
                input_0_dims[2]
                if len(input_0_dims) == 3
                else input_0_dims[1] * input_0_dims[3]
            )
            self.reduction_depth = hidden_size
        elif node.target in [
            torch.ops.aten.convolution.default,
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.conv3d.default,
        ]:
            # Conv: input (N x C x D1 x D2 ... x Dn)
            # weight (out_channels, in_channels, kD1, kD2, ... kDn)
            # Reduction depth = in_channels * kernel_volume
            weight_shape = (
                self._get_tensor_shape(node.all_input_nodes[1].name)
                if len(node.all_input_nodes) > 1
                else None
            )
            if weight_shape is None:
                return False
            in_channels = weight_shape[1]
            kernel_volume = weight_shape[2:].numel()
            self.reduction_depth = in_channels * kernel_volume
        elif node.target in [
            torch.ops.aten.matmul,
            torch.ops.aten.matmul.default,
            torch.ops.aten.dot.default,
            torch.ops.aten.mm.default,
            torch.ops.aten.mv.default,
            torch.ops.aten.bmm.default,
        ]:
            # GEMM: A (M, K) @ B (K, N) = C (M, N)
            self.reduction_depth = input_0_dims[-1]
        # TODO: Add more reduction ops here
        return self.reduction_depth > self.max_depth_of_reduction


class NodeClassifier:
    """Main class for classifying nodes into high and low precision groups."""

    def __init__(
        self,
        nodes,
        excluded_nodes: Collection[str] | None = None,
        excluded_node_names: Collection[str] | None = None,
        excluded_ops: Collection[torch.fx.node.Target] | None = None,
        custom_rule: NodeRuleBase | None = None,
        max_output_threshold: float | None = 512,
        max_depth_of_reduction: int | None = None,
    ):
        """Initialize the node classifier.

        Args:
            nodes: The nodes to classify.
            excluded_nodes: Collection of regex patterns for the module paths of the nodes to keep in high precision.
            excluded_node_names: Collection of regex patterns for the names of the nodes to keep in high precision.
            excluded_ops: Collection of targets to keep in high precision.
            custom_rule: Optional custom classification rule.
            max_output_threshold: Maximum absolute value allowed for node I/O.
            max_depth_of_reduction: Maximum depth of reduction allowed in low precision.
        """
        self.nodes = nodes
        self.excluded_nodes = excluded_nodes
        self.excluded_node_names = excluded_node_names
        self.excluded_ops = excluded_ops
        self.custom_rule = custom_rule
        self.max_output_threshold = max_output_threshold
        self.max_depth_of_reduction = max_depth_of_reduction

    def _gen_block_node_rules(self, reference_magnitudes, reference_shapes):
        """Generate list of rules for blocking nodes from precision conversion.

        Args:
            reference_magnitudes: Largest absolute value of each tensor in the reference data.
            reference_shapes: Largest size of each dim of each tensor in the reference data.

        Returns:
            list[NodeRuleBase]: List of rules to apply.
        """
        block_node_rules: list[NodeRuleBase] = []
        if self.excluded_nodes:
            block_node_rules.append(DisabledNodeNameRegexRule(self.excluded_nodes))
        if self.excluded_node_names:
            block_node_rules.append(NodeNameRegexRule(self.excluded_node_names))
        if self.excluded_ops:
            block_node_rules.append(DisabledOpTypes(self.excluded_ops))
        if reference_magnitudes:
            block_node_rules.append(
                IORangeRule(self.max_output_threshold, reference_magnitudes)
            )
        if self.max_depth_of_reduction is not None:
            block_node_rules.append(
                DepthOfReductionRule(
                    self.max_depth_of_reduction,
                    reference_shapes,
                )
            )
        if self.custom_rule:
            block_node_rules.append(self.custom_rule)
        return block_node_rules

    def run(
        self,
        ref_magnitudes_dict: Optional[dict[str, float]] = None,
        ref_shapes_dict: Optional[dict[str, torch.Size]] = None,
    ) -> tuple[list[str], list[str]]:
        """Run node classification.

        Args:
            ref_magnitudes_dict: Optional largest absolute value of each reference tensor.
            ref_shapes_dict: Optional largest size of each dim of each reference tensor.

        Returns:
            tuple: Lists of node names (low_precision_nodes, high_precision_nodes).
        """
        block_node_rules = self._gen_block_node_rules(
            ref_magnitudes_dict, ref_shapes_dict or {}
        )
        low_precision_nodes = []
        high_precision_nodes = []
        for node in self.nodes:
            if node.op == "call_function":
                if (
                    node.target == torch.ops.higher_order.wrap_with_autocast
                    or node.target == operator.getitem
                ):
                    continue
                # If any condition is met - node will be executed in high precision
                if any(rule.check(node) for rule in block_node_rules):
                    high_precision_nodes.append(node.name)
                else:
                    low_precision_nodes.append(node.name)
        logger.debug(f"Low Precision Nodes: {low_precision_nodes}")
        logger.debug(f"High Precision Nodes: {high_precision_nodes}")
        return low_precision_nodes, high_precision_nodes
