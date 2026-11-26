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
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    ConverterRegistry,
)

from .inline_autocast_regions import (
    NAME_IN_REGION,
    OP_IN_REGION,
    _copies,
    _returns_an_input,
)
from .pass_utils import (
    _has_no_floating_point_tensors,
    _has_only_integer_outputs,
    _is_given_a_dtype,
    tensors_maybe_written_by,
    writes_a_list,
)

logger = logging.getLogger(__name__)

# The largest magnitude of the finite values of a weight, on its placeholder.
WEIGHT_MAGNITUDE = "_torch_tensorrt_weight_magnitude"
# When that magnitude is the limit of its type, as for an attention mask of
# torch.finfo(torch.float32).min, the largest magnitude of the other finite values.
UNMASKED_MAGNITUDE = "_torch_tensorrt_unmasked_magnitude"

_COPIES = (torch.ops.aten.clone.default, torch.ops.aten.lift_fresh_copy.default)

# Tensor arguments whose values an op only places in its output, as where does, or
# adds to scores it then reads in the low precision type, as the attn_mask of
# scaled_dot_product_attention. When the largest finite magnitude of such a tensor is
# the limit of its type, as for an attention mask of torch.finfo(torch.float32).min,
# that value stands for infinity and becomes -inf, as it does without this rule. So a
# row where every value is masked holds only -inf, also as without this rule. The
# other values of the tensor still count.
_PLACED_ARGUMENTS = {
    torch.ops.aten.where.self: ("self", "other"),
    torch.ops.aten.masked_fill.Tensor: ("value",),
    torch.ops.aten.masked_fill_.Tensor: ("value",),
    torch.ops.aten.index_put.default: ("values",),
    torch.ops.aten.index_put_.default: ("values",),
    torch.ops.aten.fill_.Tensor: ("value",),
    torch.ops.aten.copy_.default: ("src",),
    torch.ops.aten.scaled_dot_product_attention.default: ("attn_mask",),
}


def _dtype(node: torch.fx.Node) -> Optional[torch.dtype]:
    val = node.meta.get("val")
    if isinstance(val, (list, tuple)):
        val = val[0] if val else None
    return getattr(val, "dtype", None)


def _reads_tensors(node: torch.fx.Node) -> bool:
    return any(hasattr(n.meta.get("val"), "dtype") for n in node.all_input_nodes)


def _argument(node: torch.fx.Node, name: str) -> Any:
    """The value an op is given for its argument name, or None."""
    if isinstance(node.target, torch._ops.OpOverload):
        for i, arg in enumerate(node.target._schema.arguments):
            if arg.name == name:
                return node.args[i] if i < len(node.args) else node.kwargs.get(name)
    return None


def _shares_an_input(node: torch.fx.Node) -> bool:
    """Whether an op can return its input or a view of it, as weight.t(), add_, dropout
    with train=False or h.type_as(x) of the same dtype do. A cast that changes the type,
    such as mask.half(), or that is given copy=True makes a copy."""
    return _returns_an_input(node) and not _copies(node) and not _argument(node, "copy")


def _is_no_op_cast(node: torch.fx.Node) -> bool:
    """Whether node is a cast such as h.half() or h.type_as(x) that returned its input
    when traced."""
    return (
        node.target is torch.ops.aten.type_as.default
        or (_is_given_a_dtype(node) and _shares_an_input(node))
    ) and _dtype(node) == _dtype(node.args[0])


def _sharing(
    node: torch.fx.Node,
    groups: Optional[dict[torch.fx.Node, list[torch.fx.Node]]] = None,
) -> list[torch.fx.Node]:
    """The tensor that owns the memory of node, then its views and the in-place ops on
    it, also through a chunk. groups, if given, keeps the result for each owner, since
    every node that shares its memory asks for the same list."""
    while (node.target is operator.getitem or _shares_an_input(node)) and isinstance(
        node.args[0], torch.fx.Node
    ):
        node = node.args[0]
    if groups is not None and node in groups:
        return groups[node]
    nodes = [node]
    for n in nodes:
        nodes.extend(
            u
            for u in n.users
            if u.target is operator.getitem or (_shares_an_input(u) and u.args[0] is n)
        )
    if groups is not None:
        groups[node] = nodes
    return nodes


def _tensors_written_by(node: torch.fx.Node) -> list[torch.fx.Node]:
    return [n for n in tensors_maybe_written_by(node) if _dtype(n) is not None]


def _writers(nodes: list[torch.fx.Node]) -> list[torch.fx.Node]:
    """The ops that write into the given tensors, such as fill_, _foreach_add_ or a
    block like torch.no_grad."""
    return list(
        dict.fromkeys(u for n in nodes for u in n.users if n in _tensors_written_by(u))
    )


def _is_written(nodes: list[torch.fx.Node], by_several: bool = False) -> bool:
    """Whether an op writes into the memory of nodes, a list from _sharing. With
    by_several, only a block like torch.no_grad that may also write into other tensors
    counts, or any op when a view in nodes is a cast to a type other than float32 that
    returned its input when traced, such as h.half()."""
    writers = _writers(nodes)
    shared = set(nodes)
    if not by_several or any(
        isinstance(w.target, torch._ops.HigherOrderOperator)
        and set(_tensors_written_by(w)) - shared
        for w in writers
    ):
        return bool(writers)
    # Such a cast makes a copy when its input runs in FP32, so a write into the cast
    # or into its input would miss the other.
    return bool(writers) and any(
        _is_no_op_cast(n) and _dtype(n) != torch.float32 for n in nodes[1:]
    )


def _unmasked_magnitude(tensor: torch.Tensor) -> float:
    """The largest magnitude of the finite values of tensor below the limit of its type,
    or 0."""
    magnitudes = tensor.detach().abs()
    magnitudes = magnitudes[magnitudes < torch.finfo(tensor.dtype).max]
    return magnitudes.max().item() if magnitudes.numel() else 0.0


def record_weight_magnitudes(ep: torch.export.ExportedProgram) -> None:
    """Record the largest magnitude of the finite values of each weight on its
    placeholder, for ConstantRangeRule. Only bfloat16, float32 and float64 weights
    count. A buffer that the program writes into is not a constant."""
    signature = ep.graph_signature
    names = {
        **signature.inputs_to_parameters,
        **signature.inputs_to_buffers,
        **signature.inputs_to_lifted_tensor_constants,
    }
    written = set(signature.buffers_to_mutate.values())
    for node in ep.graph.find_nodes(op="placeholder"):
        name = names.get(node.name)
        tensor = ep.state_dict.get(name, ep.constants.get(name))
        # No op reads an unused weight, and the checks of mapped weight files skip it,
        # so it is not read.
        if (
            not node.users
            or name in written
            or type(tensor) not in (torch.Tensor, torch.nn.Parameter)
            or tensor.dtype not in (torch.bfloat16, torch.float32, torch.float64)
            or tensor.layout != torch.strided
            or tensor.is_meta
            or not tensor.numel()
            or _is_written(_sharing(node))
        ):
            continue
        # aminmax reads the weight once without a copy, but gives NaN for a weight
        # with a NaN and inf for one with an inf, which do not overflow.
        low, high = torch.aminmax(tensor.detach())
        magnitude = torch.maximum(-low, high).item()
        if not math.isfinite(magnitude):
            finite = tensor.detach()[tensor.detach().isfinite()]
            if not finite.numel():
                continue
            low, high = torch.aminmax(finite)
            magnitude = torch.maximum(-low, high).item()
        node.meta[WEIGHT_MAGNITUDE] = magnitude
        if magnitude >= torch.finfo(tensor.dtype).max:
            node.meta[UNMASKED_MAGNITUDE] = _unmasked_magnitude(tensor)


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
        return any(
            ConverterRegistry.qualified_name_or_str(op) in self.excluded_ops
            for op in (node.target, node.meta.get(OP_IN_REGION, node.target))
        )


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
    names of the nodes it keeps, and of the views and in-place ops that share their
    memory, are in kept."""

    def __init__(self, low_precision_type: torch.dtype) -> None:
        """Initialize the rule.

        Args:
            low_precision_type: The type that autocast would cast the constants to.
        """
        finfo = torch.finfo(low_precision_type)
        # Half a unit in the last place above the largest value: the smallest magnitude
        # that rounds to inf, 65520 for float16.
        self.overflow = finfo.max + 2 ** (math.frexp(finfo.max)[1] - 2) * finfo.eps
        self.low_precision_type = low_precision_type
        self.constant: Optional[tuple[str, float]] = None
        self.kept: set[str] = set()
        self.groups: dict[torch.fx.Node, list[torch.fx.Node]] = {}
        # For each owner in groups: the ops that write into its memory, and whether
        # _is_written with by_several holds.
        self.writers: dict[torch.fx.Node, tuple[list[torch.fx.Node], bool]] = {}
        self.written_constants: dict[torch.fx.Node, Optional[tuple[str, float]]] = {}

    def _constants(self, node: torch.fx.Node) -> Iterator[tuple[str, float]]:
        names = _PLACED_ARGUMENTS.get(node.target, ())
        # index_put with accumulate=True adds its values instead of placing them.
        placed = (
            set()
            if _argument(node, "accumulate")
            else {_argument(node, n) for n in names}
        )
        for n in node.all_input_nodes:
            is_placed = n in placed
            # A view such as weight.t(), a chunk of a weight, and a copy such as the
            # lift_fresh_copy of torch.tensor(1e6), have the values of the weight. A
            # cast that changes the type, such as mask.half(), makes a copy in that
            # type, so a write into it must not run in FP32.
            while (
                (_shares_an_input(n) or n.target in (operator.getitem, *_COPIES))
                and isinstance(n.args[0], torch.fx.Node)
                and _dtype(n.args[0]) == _dtype(n)
            ):
                n = n.args[0]
            magnitude = n.meta.get(WEIGHT_MAGNITUDE)
            if (
                is_placed
                and magnitude is not None
                and magnitude >= torch.finfo(_dtype(n)).max
            ):
                magnitude = n.meta.get(UNMASKED_MAGNITUDE)
            if magnitude is not None:
                yield n.name, magnitude
        if not isinstance(node.target, torch._ops.OpOverload):
            return
        for i, arg in enumerate(node.target._schema.arguments):
            value = node.args[i] if i < len(node.args) else node.kwargs.get(arg.name)
            arg_type = arg.type
            if isinstance(arg_type, torch.OptionalType):
                arg_type = arg_type.getElementType()
            # A number given for a Tensor or a Scalar, not an int such as a dim or the
            # end of a slice.
            if (
                isinstance(arg_type, (torch.TensorType, torch.NumberType))
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                yield arg.name, value

    def _first_constant(
        self, nodes: list[torch.fx.Node]
    ) -> Optional[tuple[str, float]]:
        return next(
            (
                (name, value)
                for n in nodes
                for name, value in self._constants(n)
                if math.isfinite(value) and abs(value) >= self.overflow
            ),
            None,
        )

    def _check_inner(self, node: torch.fx.Node) -> bool:
        sharing = _sharing(node, self.groups)
        if sharing[0] not in self.writers:
            self.writers[sharing[0]] = (
                _writers(sharing),
                _is_written(sharing, by_several=True),
            )
        writers, written_by_several = self.writers[sharing[0]]
        if written_by_several:
            # A block gets one type for all its inputs and may write into any of them,
            # so a tensor that it takes with other tensors keeps the type it has
            # without this rule. So does a written tensor with a cast such as h.half()
            # that returned it when traced.
            return False
        # A view of, and an in-place op on, a tensor that this rule keeps run in FP32
        # with it, since a cast would make an in-place op such as fill_ write into a
        # copy.
        # A block such as torch.no_grad that writes in place counts as an in-place op.
        written = _tensors_written_by(node)
        if written or _shares_an_input(node):
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
            n.name in self.kept and (written or _dtype(n) == _dtype(node))
            for n in shared
        ):
            self.constant = None
            self.kept.add(node.name)
            return True
        if (
            written
            and (
                writes_a_list(node)
                or (
                    isinstance(node.target, torch._ops.HigherOrderOperator)
                    and any(
                        n.op != "placeholder" or _dtype(n) != torch.float32
                        for n in shared
                    )
                )
            )
        ) or (
            not written
            and shared
            and all(
                n.op == "call_function"
                and n.name not in self.kept
                and _reads_tensors(n)
                and not _is_given_a_dtype(n)
                for n in shared
            )
        ):
            # A list writer such as _foreach_add_ writes each tensor in its own type,
            # so only the tensors it writes count. A view of ops that this rule did not
            # keep runs in the low precision type, and so does a block that writes into
            # one, or into a weight, buffer or graph input that is not float32, since in
            # FP32 it would write into a copy. NodeClassifier stops at the first rule
            # that keeps a node, so this rule does not see an op that another rule
            # keeps in FP32. A factory op such as torch.full, or an op given a dtype
            # such as w.float(), is never cast.
            return False
        if (
            _has_no_floating_point_tensors(node)
            or _has_only_integer_outputs(node)
            or _is_given_a_dtype(node)
        ):
            # Autocast does not cast an integer op such as ids == 128000 or bucketize,
            # or an op given a dtype such as x.half().
            return False
        # An in-place op runs in the type of the tensor it writes into, so the
        # constants it reads count for the op that made that tensor, unless in-place
        # ops are its only readers, as the masked_fill_ of attention scores is: they
        # then compute on an FP32 copy that the ops after them read. When this rule
        # cannot keep that tensor, as for a weight, buffer or graph input, an op given
        # a dtype, a factory op or an op that another rule keeps, they count for every
        # in-place op on it. On one of another floating type, they then compute in
        # FP32 and write the result in that type, rounded as in eager for float16.
        if sharing[0] not in self.written_constants:
            self.written_constants[sharing[0]] = self._first_constant(writers)
        self.constant = self._first_constant([node])
        if self.constant is None and not (
            node is sharing[0]
            and all(
                _tensors_written_by(u) == [node] and u.args[0] is node
                for u in node.users
            )
        ):
            self.constant = self.written_constants[sharing[0]]
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
