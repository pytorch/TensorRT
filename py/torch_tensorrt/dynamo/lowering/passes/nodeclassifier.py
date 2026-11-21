# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Borrowed from ModelOpt AutoCast's nodeclassifier.py, modified to fit Torch-TensorRT's needs.
import abc
import logging
import math
import operator
import re
from typing import Any, Collection, Iterator, Optional

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    ConverterRegistry,
)

from .inline_autocast_regions import NAME_IN_REGION, _aliases_an_input
from .pass_utils import (
    _has_no_floating_point_tensors,
    _has_only_integer_outputs,
    _is_given_a_dtype,
    tensors_maybe_written_by,
)

logger = logging.getLogger(__name__)

# The largest magnitude of the finite values of a weight, on its placeholder, or of a
# value that the graph computes from weights, on its node.
WEIGHT_MAGNITUDE = "_torch_tensorrt_weight_magnitude"

_COPIES = (torch.ops.aten.clone.default, torch.ops.aten.lift_fresh_copy.default)

# Tensor arguments whose values an op only places in its output, such as an attention
# mask value of torch.finfo(torch.float32).min. The next low precision op reads it as
# -inf anyway, as scaled_dot_product_attention does inside, so keeping the op in FP32
# would only add casts.
_PLACED_ARGUMENTS = {
    torch.ops.aten.where.self: ("self", "other"),
    torch.ops.aten.masked_fill.Tensor: ("value",),
    torch.ops.aten.index_put.default: ("values",),
    torch.ops.aten.scaled_dot_product_attention.default: ("attn_mask",),
}

_FILLS = (
    torch.ops.aten.full.default,
    torch.ops.aten.full_like.default,
    torch.ops.aten.new_full.default,
)

# The floating point types whose values can be above the float16 range.
_WIDE_TYPES = (torch.bfloat16, torch.float32, torch.float64)


def _dtype(node: torch.fx.Node) -> Optional[torch.dtype]:
    val = node.meta.get("val")
    if isinstance(val, (list, tuple)):
        val = val[0] if val else None
    return getattr(val, "dtype", None)


def _reads_tensors(node: torch.fx.Node) -> bool:
    return any(hasattr(n.meta.get("val"), "dtype") for n in node.all_input_nodes)


def _is_written(node: torch.fx.Node, by_a_block: bool = False) -> bool:
    """Whether an op, or with by_a_block a block such as torch.no_grad, writes into
    node, also through a view of it."""
    nodes = [node]
    for n in nodes:
        for user in n.users:
            if n in tensors_maybe_written_by(user) and (
                not by_a_block
                or isinstance(user.target, torch._ops.HigherOrderOperator)
            ):
                return True
            if (
                _aliases_an_input(user) and user.args[0] is n
            ) or user.target is operator.getitem:
                nodes.append(user)
    return False


def _overflow(dtype: torch.dtype) -> float:
    """The smallest magnitude that rounds to inf in dtype: half a unit in the last place
    above its largest value, 65520 for float16."""
    finfo = torch.finfo(dtype)
    return finfo.max + 2 ** (math.frexp(finfo.max)[1] - 2) * finfo.eps


def _numbers(node: torch.fx.Node) -> Iterator[tuple[str, float]]:
    """The numbers node is given for a Tensor or a Scalar, such as the 1e6 of x / 1e6,
    and not ints such as a dim or the end of a slice."""
    if not isinstance(node.target, torch._ops.OpOverload):
        return
    for i, arg in enumerate(node.target._schema.arguments):
        value = node.args[i] if i < len(node.args) else node.kwargs.get(arg.name)
        arg_type = arg.type
        if isinstance(arg_type, torch.OptionalType):
            arg_type = arg_type.getElementType()
        if (
            isinstance(arg_type, (torch.TensorType, torch.NumberType))
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            yield arg.name, value


def _largest_finite_magnitude(tensor: torch.Tensor) -> Optional[float]:
    # aminmax reads the tensor once without a copy, but gives NaN for a tensor with a
    # NaN and inf for one with an inf, which do not overflow.
    low, high = torch.aminmax(tensor.detach())
    magnitude = torch.maximum(-low, high).item()
    if not math.isfinite(magnitude):
        finite = tensor.detach()[tensor.detach().isfinite()]
        if not finite.numel():
            return None
        low, high = torch.aminmax(finite)
        magnitude = torch.maximum(-low, high).item()
    return magnitude


def _is_computable(node: torch.fx.Node) -> bool:
    """Whether node is an ATen op that makes one nonempty tensor of a type in _WIDE_TYPES
    from its inputs alone, without writing into them or drawing random numbers."""
    return (
        isinstance(node.target, torch._ops.OpOverload)
        and node.target.namespace == "aten"
        and not node.target._schema.is_mutable
        and torch.Tag.maybe_aliasing_or_mutating not in node.target.tags
        and torch.Tag.nondeterministic_seeded not in node.target.tags
        and isinstance(node.meta.get("val"), torch.Tensor)
        and node.meta["val"].dtype in _WIDE_TYPES
        and statically_known_true(node.meta["val"].numel() > 0)
    )


def record_weight_magnitudes(ep: torch.export.ExportedProgram) -> None:
    """Record the largest magnitude of the finite values of each weight on its
    placeholder, for ConstantRangeRule. Only bfloat16, float32 and float64 weights
    count. A buffer that the program writes into is not a constant. Values that the
    graph computes from weights and numbers alone are constants too, and the magnitude
    of those with an input or a number above the float16 range is recorded on their
    node."""
    signature = ep.graph_signature
    names = {
        **signature.inputs_to_parameters,
        **signature.inputs_to_buffers,
        **signature.inputs_to_lifted_tensor_constants,
    }
    written = set(signature.buffers_to_mutate.values())
    values: dict[torch.fx.Node, torch.Tensor] = {}
    for node in ep.graph.find_nodes(op="placeholder"):
        name = names.get(node.name)
        tensor = ep.state_dict.get(name, ep.constants.get(name))
        if (
            name in written
            or type(tensor) not in (torch.Tensor, torch.nn.Parameter)
            or tensor.dtype not in _WIDE_TYPES
            or tensor.is_meta
            or not tensor.numel()
            or _is_written(node)
        ):
            continue
        values[node] = tensor.detach()
        magnitude = _largest_finite_magnitude(tensor)
        if magnitude is not None:
            node.meta[WEIGHT_MAGNITUDE] = magnitude

    # A value such as the (running_var + eps).rsqrt() of torchvision's
    # FrozenBatchNorm2d, or torch.full((), 1e6), is still above the range when the op
    # that reads the large value runs in FP32, so its users need its magnitude too.
    # constant_fold folds these values later, so computing them here costs no runtime.
    def value(node: torch.fx.Node) -> Any:
        if node not in values:
            values[node] = node.target(
                *torch.fx.node.map_arg(node.args, value),
                **torch.fx.node.map_arg(node.kwargs, value),
            )
        return values[node]

    overflow = _overflow(torch.float16)
    constants = set(values)
    for node in ep.graph.nodes:
        if node.target in _FILLS and _is_computable(node):
            # The magnitude of a fill is its value, so it is not computed itself. That
            # also covers torch.full_like(x, 1e6) and a torch.full of a dynamic size.
            for _, fill_value in _numbers(node):
                if math.isfinite(fill_value):
                    node.meta[WEIGHT_MAGNITUDE] = abs(fill_value)
        if (
            node.op != "call_function"
            or not _is_computable(node)
            or not all(n in constants for n in node.all_input_nodes)
            or _is_written(node)
        ):
            continue
        constants.add(node)
        if WEIGHT_MAGNITUDE not in node.meta and (
            any(
                n.meta.get(WEIGHT_MAGNITUDE, 0) >= overflow
                for n in node.all_input_nodes
            )
            or any(math.isfinite(v) and abs(v) >= overflow for _, v in _numbers(node))
        ):
            try:
                with torch.no_grad():
                    magnitude = _largest_finite_magnitude(value(node))
            except (RuntimeError, NotImplementedError) as e:
                # The check must not fail a compile, for example when a weight was
                # moved to another device than a factory op in the graph.
                logger.warning(f"Cannot compute {node.name} to check its users: {e}")
                continue
            if magnitude is not None:
                node.meta[WEIGHT_MAGNITUDE] = magnitude


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

    def __init__(self, node_names: Collection[str]) -> None:
        """Initialize the rule.

        Args:
            node_names: Collection of node names to keep in high precision.
        """
        self.node_names = node_names

    def _check_inner(self, node: torch.fx.Node) -> bool:
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


class ConstantRangeRule(NodeRuleBase):
    """Rule for keeping nodes in high precision when a constant they read, a weight or a
    number such as the 1e6 of x / 1e6, becomes inf in the low precision type, as a
    BatchNorm running variance of 1e6 does in float16. Needs no calibration data. NaN
    and inf stay as they are in the low precision type, so they do not count. The
    names of the nodes it keeps are in kept."""

    def __init__(self, low_precision_type: torch.dtype) -> None:
        """Initialize the rule.

        Args:
            low_precision_type: The type that autocast would cast the constants to.
        """
        self.overflow = _overflow(low_precision_type)
        self.low_precision_type = low_precision_type
        self.constant: Optional[tuple[str, float]] = None
        self.kept: set[str] = set()

    def _constants(self, node: torch.fx.Node) -> Iterator[tuple[str, float]]:
        names = _PLACED_ARGUMENTS.get(node.target, ())
        arguments = {
            arg.name: node.args[i] if i < len(node.args) else node.kwargs.get(arg.name)
            for i, arg in enumerate(node.target._schema.arguments if names else ())
        }
        # index_put with accumulate=True adds its values instead of placing them.
        placed = set() if arguments.get("accumulate") else {arguments[n] for n in names}
        for n in node.all_input_nodes:
            if n in placed:
                continue
            # A view such as weight.t(), a chunk of a weight, and a copy such as the
            # lift_fresh_copy of torch.tensor(1e6), have the values of the weight,
            # unless the magnitude of their own values was computed, as for a slice.
            while (
                WEIGHT_MAGNITUDE not in n.meta
                and (_aliases_an_input(n) or n.target in (operator.getitem, *_COPIES))
                and isinstance(n.args[0], torch.fx.Node)
            ):
                n = n.args[0]
            if WEIGHT_MAGNITUDE in n.meta:
                yield n.name, n.meta[WEIGHT_MAGNITUDE]
        yield from _numbers(node)

    def _check_inner(self, node: torch.fx.Node) -> bool:
        if _is_written(node, by_a_block=True):
            # A block gets one type for all its inputs and may write into any of them,
            # so a tensor that it takes keeps the type it has without this rule.
            return False
        # A view and an in-place op run in the type of the tensor whose memory they
        # share, since a cast would make an in-place op such as fill_ write into a copy.
        # A block such as torch.no_grad that writes in place counts as an in-place op.
        written = tensors_maybe_written_by(node)
        if written or _aliases_an_input(node):
            shared = written or [
                a for a in node.args[:1] if isinstance(a, torch.fx.Node)
            ]
        elif isinstance(node.target, torch._ops.HigherOrderOperator):
            # A block such as torch.no_grad can return a view of a tensor it takes.
            shared = [n for n in node.all_input_nodes if _dtype(n) is not None]
        else:
            shared = []
        # NodeClassifier skips getitem, so a chunk of a tensor follows the op that made it.
        shared = [n.args[0] if n.target is operator.getitem else n for n in shared]
        # A cast such as h.half() shares memory only when it changes nothing.
        if shared and all(
            n.name in self.kept and _dtype(n) == _dtype(node) for n in shared
        ):
            self.constant = None
            self.kept.add(node.name)
            return True
        if written or (
            shared
            and all(
                n.op == "call_function"
                and n.name not in self.kept
                and _reads_tensors(n)
                for n in shared
            )
        ):
            # An in-place op or a view of a tensor that runs in the low precision type
            # runs in it too. A factory op such as torch.full is never cast.
            return False
        if not _reads_tensors(node):
            # Autocast does not cast a factory op such as torch.full((), 1e6), and its
            # users read the value it makes.
            return False
        if (
            _has_no_floating_point_tensors(node)
            or _has_only_integer_outputs(node)
            or _is_given_a_dtype(node)
        ):
            # Autocast does not cast an integer op such as ids == 128000 or bucketize,
            # or an op given a dtype such as x.half().
            return False
        self.constant = next(
            (
                (name, value)
                for name, value in self._constants(node)
                if math.isfinite(value) and abs(value) >= self.overflow
            ),
            None,
        )
        if self.constant is not None:
            self.kept.add(node.name)
        return self.constant is not None

    def _log_skipped(self, node: torch.fx.Node, **kwargs: Any) -> None:
        if self.constant is None:
            logger.info(
                f"Skipping node {node.name}: it shares the memory of a kept node"
            )
            return
        logger.info(
            f"Skipping node {node.name}: constant {self.constant[0]} has magnitude "
            f"{abs(self.constant[1])}, which is inf in {self.low_precision_type}"
        )


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
        constant_rule: NodeRuleBase | None = None,
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
            constant_rule: Optional rule for constants that do not fit in the low precision type.
        """
        self.nodes = nodes
        self.excluded_nodes = excluded_nodes
        self.excluded_node_names = excluded_node_names
        self.excluded_ops = excluded_ops
        self.custom_rule = custom_rule
        self.max_output_threshold = max_output_threshold
        self.max_depth_of_reduction = max_depth_of_reduction
        self.constant_rule = constant_rule

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
        if self.constant_rule:
            block_node_rules.append(self.constant_rule)
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
