# Borrowed from ModelOpt AutoCast's nodeclassifier.py, modified to fit Torch-TensorRT's needs.
import abc
import logging
import operator
import re
from typing import Collection, Optional

import torch

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
            node: The ONNX node to check.

        Returns:
            bool: True if the node should be kept in high precision, False otherwise.
        """
        result = self._check_inner(node)
        if result:
            self._log_skipped(node)
            return True
        return False


class DisabledNodeNameRegexRule(NodeRuleBase):
    """Rule for keeping nodes with matching names in high precision."""

    def __init__(self, disabled_node_name_regex):
        """Initialize the rule.

        Args:
            disabled_node_name_regex: List of regex patterns for node names to keep in high precision.
        """
        self.disabled_node_name_regex = disabled_node_name_regex

    def _check_inner(self, node):
        return any(
            re.match(regex, node.name) for regex in self.disabled_node_name_regex
        )


class DisabledTargets(NodeRuleBase):
    """Rule for keeping nodes with specific operation types in high precision."""

    def __init__(self, targets_to_exclude):
        """Initialize the rule.

        Args:
            targets_to_exclude: List of operation types to keep in high precision.
        """
        self.targets_to_exclude = targets_to_exclude

    def _check_inner(self, node):
        return node.target in self.targets_to_exclude


class IORangeRule(NodeRuleBase):
    """Rule for keeping nodes with out-of-range inputs/outputs in high precision."""

    def __init__(self, data_max, reference_data):
        """Initialize the rule.

        Args:
            data_max: Maximum absolute value allowed for node I/O.
            reference_data: Reference data for checking I/O ranges.
        """
        self.data_max = data_max
        self.reference_data = reference_data
        self.output_data = None

    def _check_inner(self, node):
        def is_io_out_of_range(node):
            tensor_name = node.name
            if tensor_name not in self.reference_data:
                logger.debug(
                    f"Node {node.name}: Tensor {tensor_name} not found in reference data. Skipping I/O range check."
                )
                return False
            ref_data = self.reference_data[tensor_name]
            if ref_data.numel() == 0:
                logger.debug(
                    f"Node {node.name}: Tensor {tensor_name} has 0 elements. Skipping I/O range check."
                )
                return False
            logger.debug(
                f"Node {node.name}: reference data: min={ref_data.min()}, max={ref_data.max()}"
            )
            if torch.any(torch.abs(ref_data) > self.data_max):
                self.output_data = ref_data
                return True

        if self.reference_data:
            for in_node in node.all_input_nodes:
                if is_io_out_of_range(in_node):
                    return True
            for out_node in list(node.users):
                if is_io_out_of_range(out_node):
                    return True
        return False

    def _log_skipped(self, node, **kwargs):
        """Log information about skipped nodes with I/O range violations."""
        if self.output_data is not None:
            logger.info(
                f"Skipping node {node.name}: reference IO out of range: min={torch.min(self.output_data)}, "
                f"max={torch.max(self.output_data)}, range=[{-self.data_max}, {self.data_max}]"
            )
        else:
            super()._log_skipped(node, **kwargs)


class DepthOfReductionRule(NodeRuleBase):
    """Rule for keeping nodes with high depth of reduction in high precision."""

    def __init__(self, max_depth_of_reduction, reference_data):
        """Initialize the rule.

        Args:
            max_depth_of_reduction: Maximum depth of reduction allowed in low precision.
            reference_data: Reference data for checking I/O ranges.
        """
        self.max_depth_of_reduction = max_depth_of_reduction
        self.reference_data = reference_data
        self.reduction_depth = 0

    def _get_tensor_shape(self, tensor_name):
        """Get tensor shape from reference data."""
        if tensor_name in self.reference_data:
            return self.reference_data[tensor_name].shape
        return None

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
            kernel_volume = torch.prod(weight_shape[2:])
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
        nodes_to_exclude: Collection[str] | None = None,
        targets_to_exclude: Collection[torch.fx.node.Target] | None = None,
        custom_rule: NodeRuleBase | None = None,
        data_max: float | None = 1000.0,
        max_depth_of_reduction: int | None = None,
    ):
        """Initialize the node classifier.

        Args:
            nodes: The nodes to classify.
            nodes_to_exclude: Collection of regex patterns for node names to keep in high precision.
            targets_to_exclude: Collection of targets to keep in high precision.
            custom_rule: Optional custom classification rule.
            data_max: Maximum absolute value allowed for node I/O.
            max_depth_of_reduction: Maximum depth of reduction allowed in low precision.
        """
        self.nodes = nodes
        self.nodes_to_exclude = nodes_to_exclude
        self.targets_to_exclude = targets_to_exclude
        self.custom_rule = custom_rule
        self.data_max = data_max
        self.max_depth_of_reduction = max_depth_of_reduction

    def _gen_block_node_rules(self, reference_data):
        """Generate list of rules for blocking nodes from precision conversion.

        Args:
            reference_data: Reference data for checking I/O ranges.

        Returns:
            list[NodeRuleBase]: List of rules to apply.
        """
        block_node_rules: list[NodeRuleBase] = []
        if self.nodes_to_exclude:
            block_node_rules.append(DisabledNodeNameRegexRule(self.nodes_to_exclude))
        if self.targets_to_exclude:
            block_node_rules.append(DisabledTargets(self.targets_to_exclude))
        if reference_data:
            block_node_rules.append(IORangeRule(self.data_max, reference_data))
        if self.max_depth_of_reduction is not None:
            block_node_rules.append(
                DepthOfReductionRule(
                    self.max_depth_of_reduction,
                    reference_data,
                )
            )
        if self.custom_rule:
            block_node_rules.append(self.custom_rule)
        return block_node_rules

    def run(
        self, ref_outputs_dict: Optional[dict[str, torch.Tensor]] = None
    ) -> tuple[list[str], list[str]]:
        """Run node classification.

        Args:
            ref_outputs_dict: Optional tensors' reference data.

        Returns:
            tuple: Lists of node names (low_precision_nodes, high_precision_nodes).
        """
        block_node_rules = self._gen_block_node_rules(ref_outputs_dict)
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
