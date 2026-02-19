import logging
import math
from typing import Callable, List, Optional, Set, Tuple

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import GraphModule, Node
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily

from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering._SubgraphBuilder import SubgraphBuilder
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)

# Ops that are elementwise-safe on the [..., 2] real layout used to represent
# complex tensors.  These ops apply independently to every scalar in the tensor
# (including both the real and imaginary components stored in the last dim) so
# no explicit rewrite is needed — the pass-through behaviour is correct.
#
# NOTE: add.Scalar / sub.Scalar are NOT in this set.  (a+bi)+s = (a+s)+bi
# adds the scalar only to the real part, but on the [...,2] layout
# add.Scalar would add to both parts.  Those need explicit rewrites.
_ELEMENTWISE_SAFE: frozenset = frozenset(
    {
        # Arithmetic — component-wise operations are correct by construction
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.neg.default,
        torch.ops.aten.mul.Scalar,  # scalar*(re,im) — both parts scaled equally
        torch.ops.aten.div.Scalar,  # (re,im)/scalar — both parts divided equally
        # Structural / copy — operate on the whole tensor without touching content.
        # Note: permute.default is NOT here; it needs an explicit rewrite to append
        # the trailing real/imag dimension index to the dims list.
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.t.default,
        # Construction — producing zero/one tensors of the same shape is layout-neutral
        torch.ops.aten.zeros_like.default,
        torch.ops.aten.ones_like.default,
        # Conditional selection — correct on the real layout when mask broadcasts
        torch.ops.aten.where.self,
        # Rounding — applies to each float independently; complex rounding is
        # undefined in PyTorch so these only appear after the rewrite anyway
        torch.ops.aten.ceil.default,
        torch.ops.aten.floor.default,
        torch.ops.aten.round.default,
        torch.ops.aten.trunc.default,
    }
)


def _complex_unpacker(*ops: object) -> Callable:
    """Decorator that registers a rewrite method for a complex aten op into a real value subgraph.

    Usage::

        @_complex_unpacker(aten.sin.default, aten.cos.default)
        def _rewrite_sin_cos(self, node): ...

    The ops are stored on the function as ``._complex_unpacker_ops`` and picked up by
    ``@_register_unpackers`` when the class is fully defined.
    """

    def decorator(fn: Callable) -> Callable:
        fn._complex_unpacker_ops = ops
        return fn

    return decorator


def _register_unpackers(cls: type) -> type:
    """Class decorator that builds ``cls._DISPATCH`` from all methods tagged
    with ``@_complex_unpacker``.  Applied once at class-definition time."""
    dispatch: dict = {}
    for attr in vars(cls).values():
        for op in getattr(attr, "_complex_unpacker_ops", ()):
            dispatch[op] = attr
    cls._DISPATCH = dispatch
    return cls


class ComplexSubGraphInfo:
    def __init__(
        self,
        anchor_nodes: List[Node],
        subgraph_nodes: List[Node],
        input_nodes: List[Node],
    ):
        self.anchor_nodes = anchor_nodes
        self.subgraph_nodes = subgraph_nodes
        self.input_nodes = input_nodes

    def __repr__(self) -> str:
        return (
            f"ComplexOpSubGraphInfo(anchor_nodes={[n.name for n in self.anchor_nodes]}, "
            f"subgraph={[n.name for n in self.subgraph_nodes]}, "
            f"inputs={[n.name for n in self.input_nodes]})"
        )


class ComplexOpDetector:
    def __init__(self) -> None:
        pass

    def is_complex_dtype(self, node: Node) -> bool:
        # Check if node's metadata or dtype is complex
        dtype = None
        if "val" in node.meta:
            val = node.meta["val"]
            if hasattr(val, "dtype"):
                dtype = val.dtype

        return dtype in {torch.complex64, torch.complex128}

    def has_complex_input(self, node: Node) -> bool:
        """Return True if any input to node has complex dtype."""
        return any(self.is_complex_dtype(inp) for inp in node.all_input_nodes)

    def node_include_in_subgraph(self, node: Node) -> bool:
        # Include call_function ops that either output complex OR consume complex inputs.
        # The second condition catches real-output ops like abs, angle, real, imag whose
        # inputs are complex and must be rewritten alongside the rest of the subgraph.
        if node.op != "call_function":
            return False
        return self.is_complex_dtype(node) or self.has_complex_input(node)

    def subgraph_from_anchor(self, anchor_node: Node) -> ComplexSubGraphInfo:
        subgraph_nodes: Set[Node] = set()
        input_nodes: Set[Node] = set()
        stack = [anchor_node]
        while stack:
            n = stack.pop()
            if n in subgraph_nodes:
                continue
            subgraph_nodes.add(n)
            for inp in n.all_input_nodes:
                if self.node_include_in_subgraph(inp):
                    stack.append(inp)
                else:
                    input_nodes.add(inp)
        # Sort subgraph_nodes in topological (graph) order so the rewriter
        # processes producers before consumers.  The set has no stable order,
        # which caused bugs when e.g. mul(sin, sin) was processed before sin
        # was rewritten (sin still had complex dtype, so the mul pattern ran
        # against the original complex node and produced wrong results).
        node_order = {n: i for i, n in enumerate(anchor_node.graph.nodes)}
        ordered_subgraph = sorted(subgraph_nodes, key=lambda n: node_order.get(n, 0))
        return ComplexSubGraphInfo(
            [anchor_node], ordered_subgraph, list(input_nodes)
        )

    def find_complex_op_subgraphs(
        self, gm: GraphModule, anchor_target: str
    ) -> List[ComplexSubGraphInfo]:
        complex_op_subgraphs: List[ComplexSubGraphInfo] = []
        for node in gm.graph.nodes:
            if node.target == anchor_target:
                new_sub = self.subgraph_from_anchor(node)
                # if any intersecting nodes between seen and sub.subgraph_nodes they should be merged
                merged = False
                for existing_sub in complex_op_subgraphs:
                    if set(existing_sub.subgraph_nodes) & set(new_sub.subgraph_nodes):
                        logger.debug(f"merging subgraphs {existing_sub} {new_sub}")
                        # merge the two subgraphs, preserving topological order
                        merged_nodes = set(existing_sub.subgraph_nodes) | set(new_sub.subgraph_nodes)
                        node_order = {n: i for i, n in enumerate(gm.graph.nodes)}
                        existing_sub.subgraph_nodes = sorted(merged_nodes, key=lambda n: node_order.get(n, 0))
                        existing_sub.input_nodes = list(
                            set(existing_sub.input_nodes) | set(new_sub.input_nodes)
                        )
                        existing_sub.anchor_nodes = list(
                            set(existing_sub.anchor_nodes) | set(new_sub.anchor_nodes)
                        )
                        merged = True
                        break
                if not merged:
                    complex_op_subgraphs.append(new_sub)
        return complex_op_subgraphs

    def find_all_complex_subgraphs(self, gm: GraphModule) -> List[ComplexSubGraphInfo]:
        """Forward scan: collect all complex-dtype call_function nodes as one subgraph.

        Unlike find_complex_op_subgraphs (which walks backwards from a single anchor),
        this scans forward over every node and collects all call_function nodes whose
        output is complex — regardless of whether they are bounded by view_as_real.
        This ensures complex ops that feed directly into graph outputs (no view_as_real)
        are still rewritten to real arithmetic.
        """
        subgraph_nodes: Set[Node] = set()
        input_nodes: Set[Node] = set()
        for node in gm.graph.nodes:
            if not self.node_include_in_subgraph(node):
                continue
            subgraph_nodes.add(node)
            for inp in node.all_input_nodes:
                if not self.node_include_in_subgraph(inp):
                    input_nodes.add(inp)
        if not subgraph_nodes:
            return []
        # Sort in topological (graph) order so the rewriter processes producers
        # before consumers, avoiding the case where e.g. a mul node is rewritten
        # before its sin/cos inputs are rewritten (which causes wrong results).
        node_order = {n: i for i, n in enumerate(gm.graph.nodes)}
        ordered = sorted(subgraph_nodes, key=lambda n: node_order.get(n, 0))
        return [
            ComplexSubGraphInfo(
                anchor_nodes=ordered,
                subgraph_nodes=ordered,
                input_nodes=list(input_nodes),
            )
        ]


@_register_unpackers
class ComplexGraphRewriter:
    def __init__(self, gm: GraphModule, truncate_double: bool = False) -> None:
        self.gm = gm
        self.truncate_double = truncate_double

    def extract_shape_dtype_device(
        self, input_node: Node
    ) -> Tuple[Tuple[int, ...], torch.dtype, torch.device]:
        if input_node.op == "placeholder":
            tensor_val = input_node.meta["val"]

        elif input_node.op == "get_attr":
            tensor_val = self.get_attr_tensor(input_node.target)  # type: ignore

        else:
            raise ValueError(f"Unsupported node type: {input_node.op}")

        node_shape = tensor_val.size()
        dtype = tensor_val.dtype
        new_node_shape = node_shape + (2,)
        device = tensor_val.device

        if dtype == torch.complex64:
            new_node_dtype = torch.float32
        elif dtype == torch.complex128 and self.truncate_double:
            new_node_dtype = torch.float32
        else:
            new_node_dtype = torch.float64

        return new_node_shape, new_node_dtype, device

    def get_attr_tensor(self, target):  # type: ignore
        # Check if target is param or buffer
        if target in dict(self.gm.named_parameters()):
            return self.gm.get_parameter(target)
        elif target in dict(self.gm.named_buffers()):
            return self.gm.get_buffer(target)
        else:
            raise ValueError(
                f"Attribute {target} not found in gm parameters or buffers."
            )

    def replace_input_node(
        self, input_node: Node, fake_mode: Optional[FakeTensorMode] = None
    ) -> None:
        modified = False
        new_shape, new_dtype, device = self.extract_shape_dtype_device(input_node)

        if input_node.op == "placeholder":
            if fake_mode is None:
                fake_mode = FakeTensorMode()
            # Preserve symbolic dimensions from the original placeholder's fake
            # tensor so that dynamic-shape information (SymInt ranges from
            # torch.export) survives the rewrite.  We build the new fake tensor
            # by appending a concrete 2 to the original symbolic shape.
            #
            # We use the *original* placeholder's FakeTensorMode
            # (which owns the ShapeEnv with the export's range constraints) so
            # that the new SymInt dimensions belong to the same ShapeEnv as all
            # other nodes in the graph.  Using shared_fake_mode would create a
            # separate ShapeEnv and cause "symbol from different env" errors
            # during FakeTensorProp.
            orig_fake = input_node.meta.get("val", None)
            if orig_fake is not None and hasattr(orig_fake, "shape"):
                # orig_fake.shape contains the symbolic sizes; append 2 for real/imag.
                sym_shape = list(orig_fake.shape) + [2]
                orig_mode = getattr(orig_fake, "fake_mode", None)
                create_mode = orig_mode if orig_mode is not None else fake_mode
                with create_mode:
                    fake_tensor = torch.empty(sym_shape, dtype=new_dtype, device=device)
            else:
                concrete_shape = tuple(
                    int(s) if not isinstance(s, int) else s for s in new_shape
                )
                real_tensor = torch.empty(
                    concrete_shape, dtype=new_dtype, device=device
                )
                fake_tensor = fake_mode.from_tensor(real_tensor)
            with self.gm.graph.inserting_before(input_node):
                new_node = self.gm.graph.placeholder(
                    input_node.target + "_unpacked_complex"
                )
            new_node.meta["val"] = fake_tensor
            logger.debug(
                "  unpack placeholder  %s%s  ->  %s%s",
                input_node.name, tuple(fake_tensor.shape[:-1]),
                new_node.name, tuple(fake_tensor.shape),
            )

        elif input_node.op == "get_attr":
            # Sanitize dots from nested-module targets (e.g. "block1.freq")
            # so register_buffer does not raise KeyError on dotted names.
            sanitized = input_node.target.replace(".", "__")  # type: ignore
            new_attr_name = sanitized + "_unpacked_complex"
            with unset_fake_temporarily():
                original_tensor = self.get_attr_tensor(input_node.target)  # type: ignore
                stacked_tensor = torch.stack(
                    [original_tensor.real, original_tensor.imag], dim=-1
                )
                self.gm.register_buffer(new_attr_name, stacked_tensor)
            with self.gm.graph.inserting_after(input_node):
                new_node = self.gm.graph.get_attr(new_attr_name)
            logger.debug(
                "  unpack get_attr  %s%s  ->  %s%s",
                input_node.target, tuple(original_tensor.shape),
                new_attr_name, tuple(stacked_tensor.shape),
            )
        else:
            pass  # call_function inputs are rewritten in-place by the op handlers
        input_node.replace_all_uses_with(new_node)
        self.gm.graph.erase_node(input_node)
        clean_up_graph_after_modifications(self.gm)

    # ------------------------------------------------------------------
    # Private graph-building helpers
    #
    # Each helper takes a SubgraphBuilder and emits a sub-sequence of nodes,
    # advancing the builder's cursor.  They return the last node(s) they
    # inserted.
    # ------------------------------------------------------------------

    @staticmethod
    def _inline_select_re_im(b: SubgraphBuilder, inp: Node) -> Tuple[Node, Node]:
        """Select re (index 0) and im (index 1) from a [..., 2] tensor."""
        re = b(torch.ops.aten.select.int, inp, -1, 0)
        im = b(torch.ops.aten.select.int, inp, -1, 1)
        return re, im

    @staticmethod
    def _inline_cat_re_im(b: SubgraphBuilder, out_re: Node, out_im: Node) -> Node:
        """Rebuild a [..., 2] complex-layout tensor from re and im nodes."""
        re_u = b(torch.ops.aten.unsqueeze.default, out_re, -1)
        im_u = b(torch.ops.aten.unsqueeze.default, out_im, -1)
        return b(torch.ops.aten.cat.default, [re_u, im_u], -1)

    @staticmethod
    def _inline_complex_log(
        b: SubgraphBuilder, re: Node, im: Node
    ) -> Tuple[Node, Node]:
        """log(a+bi) = 0.5*log(a²+b²) + i*atan2(b, a)"""
        re2 = b(torch.ops.aten.mul.Tensor, re, re)
        im2 = b(torch.ops.aten.mul.Tensor, im, im)
        r2 = b(torch.ops.aten.add.Tensor, re2, im2)
        log_r2 = b(torch.ops.aten.log.default, r2)
        log_re = b(torch.ops.aten.mul.Tensor, log_r2, 0.5)
        log_im = b(torch.ops.aten.atan2.default, im, re)
        return log_re, log_im

    @staticmethod
    def _inline_complex_exp(
        b: SubgraphBuilder, re: Node, im: Node
    ) -> Tuple[Node, Node]:
        """exp(a+bi) = e^a*cos(b) + i*e^a*sin(b)"""
        ea = b(torch.ops.aten.exp.default, re)
        cos_b = b(torch.ops.aten.cos.default, im)
        sin_b = b(torch.ops.aten.sin.default, im)
        exp_re = b(torch.ops.aten.mul.Tensor, ea, cos_b)
        exp_im = b(torch.ops.aten.mul.Tensor, ea, sin_b)
        return exp_re, exp_im

    @staticmethod
    def _inline_complex_mul(
        b: SubgraphBuilder, re1: Node, im1: Node, re2: Node, im2: Node
    ) -> Tuple[Node, Node]:
        """(a+bi)(c+di) = (ac-bd) + (ad+bc)i"""
        ac = b(torch.ops.aten.mul.Tensor, re1, re2)
        bd = b(torch.ops.aten.mul.Tensor, im1, im2)
        ad = b(torch.ops.aten.mul.Tensor, re1, im2)
        bc = b(torch.ops.aten.mul.Tensor, im1, re2)
        out_re = b(torch.ops.aten.sub.Tensor, ac, bd)
        out_im = b(torch.ops.aten.add.Tensor, ad, bc)
        return out_re, out_im

    @staticmethod
    def _inline_complex_div(
        b: SubgraphBuilder, re1: Node, im1: Node, re2: Node, im2: Node
    ) -> Tuple[Node, Node]:
        """(a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)"""
        c2 = b(torch.ops.aten.mul.Tensor, re2, re2)
        d2 = b(torch.ops.aten.mul.Tensor, im2, im2)
        denom = b(torch.ops.aten.add.Tensor, c2, d2)
        ac = b(torch.ops.aten.mul.Tensor, re1, re2)
        bd = b(torch.ops.aten.mul.Tensor, im1, im2)
        bc = b(torch.ops.aten.mul.Tensor, im1, re2)
        ad = b(torch.ops.aten.mul.Tensor, re1, im2)
        numer_re = b(torch.ops.aten.add.Tensor, ac, bd)
        numer_im = b(torch.ops.aten.sub.Tensor, bc, ad)
        out_re = b(torch.ops.aten.div.Tensor, numer_re, denom)
        out_im = b(torch.ops.aten.div.Tensor, numer_im, denom)
        return out_re, out_im

    @staticmethod
    def _inline_complex_sqrt(
        b: SubgraphBuilder, re: Node, im: Node
    ) -> Tuple[Node, Node]:
        """sqrt(z) = r^0.5 * (cos(θ/2) + i*sin(θ/2))"""
        re2 = b(torch.ops.aten.mul.Tensor, re, re)
        im2 = b(torch.ops.aten.mul.Tensor, im, im)
        r2 = b(torch.ops.aten.add.Tensor, re2, im2)
        r = b(torch.ops.aten.sqrt.default, r2)
        r_sq = b(torch.ops.aten.pow.Tensor_Scalar, r, 0.5)
        theta = b(torch.ops.aten.atan2.default, im, re)
        half_theta = b(torch.ops.aten.mul.Tensor, theta, 0.5)
        cos_ht = b(torch.ops.aten.cos.default, half_theta)
        sin_ht = b(torch.ops.aten.sin.default, half_theta)
        sq_re = b(torch.ops.aten.mul.Tensor, r_sq, cos_ht)
        sq_im = b(torch.ops.aten.mul.Tensor, r_sq, sin_ht)
        return sq_re, sq_im

    # ------------------------------------------------------------------
    # Per-op rewrite handlers
    #
    # Each method receives the node to rewrite and returns True if it
    # modified the graph.  They are registered in _build_dispatch_table()
    # which is called at the end of __init__.
    # ------------------------------------------------------------------

    @_complex_unpacker(torch.ops.aten.view_as_complex.default)
    def _rewrite_view_as_complex(self, node: Node) -> bool:
        node.replace_all_uses_with(node.args[0])
        self.gm.graph.erase_node(node)
        return False  # bypass only, no structural change that needs propagation

    @_complex_unpacker(torch.ops.aten.view_as_real.default)
    def _rewrite_view_as_real(self, node: Node) -> bool:
        node.replace_all_uses_with(node.args[0])
        self.gm.graph.erase_node(node)
        return False

    @_complex_unpacker(torch.ops.aten.permute.default)
    def _rewrite_permute(self, node: Node) -> bool:
        # permute on a complex tensor: after rewrite the tensor has an extra
        # trailing dim of size 2 (real/imag).  Append the index for that
        # trailing dim so the permutation stays valid.
        inp = node.args[0]
        orig_dims = list(node.args[1])
        n_orig = len(orig_dims)
        new_dims = [d % n_orig for d in orig_dims] + [n_orig]
        with SubgraphBuilder(self.gm.graph, node) as b:
            out = b(torch.ops.aten.permute.default, inp, new_dims)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.mul.Tensor, torch.ops.aten.div.Tensor)
    def _rewrite_mul_div_tensor(self, node: Node) -> bool:
        arg0_is_node = isinstance(node.args[0], torch.fx.Node)
        arg1_is_node = isinstance(node.args[1], torch.fx.Node)

        if not arg0_is_node and not arg1_is_node:
            return False  # both scalars

        if node.target == torch.ops.aten.mul.Tensor and (
            not arg0_is_node or not arg1_is_node
        ):
            return False  # scalar * complex — elementwise-safe

        if node.target == torch.ops.aten.div.Tensor and not arg1_is_node:
            return False  # complex / scalar — elementwise-safe

        if node.target == torch.ops.aten.div.Tensor and not arg0_is_node:
            # scalar / complex: s/(a+bi) = (s*a/(a²+b²)) + i*(-s*b/(a²+b²))
            scalar_val = node.args[0]
            z_node = node.args[1]
            with SubgraphBuilder(self.gm.graph, node) as b:
                re = b(torch.ops.aten.select.int, z_node, -1, 0)
                im = b(torch.ops.aten.select.int, z_node, -1, 1)
                re2 = b(torch.ops.aten.mul.Tensor, re, re)
                im2 = b(torch.ops.aten.mul.Tensor, im, im)
                denom = b(torch.ops.aten.add.Tensor, re2, im2)
                re_s = b(torch.ops.aten.mul.Tensor, re, scalar_val)
                out_re = b(torch.ops.aten.div.Tensor, re_s, denom)
                im_s = b(torch.ops.aten.mul.Tensor, im, scalar_val)
                neg_im_s = b(torch.ops.aten.neg.default, im_s)
                out_im = b(torch.ops.aten.div.Tensor, neg_im_s, denom)
                out = self._inline_cat_re_im(b, out_re, out_im)
                node.replace_all_uses_with(out)
                self.gm.graph.erase_node(node)
                return True

        # Both args are Nodes from here on.
        if node.target == torch.ops.aten.div.Tensor:
            detector = ComplexOpDetector()

            def _is_complex_layout(n: Node) -> bool:
                if detector.is_complex_dtype(n):
                    return True
                val = n.meta.get("val", None)
                if val is not None and hasattr(val, "shape"):
                    return len(val.shape) >= 1 and val.shape[-1] == 2
                return False

            arg0_layout = _is_complex_layout(node.args[0])
            arg1_layout = _is_complex_layout(node.args[1])

            if arg0_layout and not arg1_layout:
                # complex_layout / real — unsqueeze denom for correct broadcast
                with SubgraphBuilder(self.gm.graph, node) as b:
                    denom_unsq = b(torch.ops.aten.unsqueeze.default, node.args[1], -1)
                    out = b(torch.ops.aten.div.Tensor, node.args[0], denom_unsq)
                    node.replace_all_uses_with(out)
                    self.gm.graph.erase_node(node)
                    return True
            elif not arg0_layout and not arg1_layout:
                return False  # both real — elementwise-safe
            else:
                # complex / complex — full div rewrite
                x_pf = node.args[0].op != "get_attr"
                y_pf = node.args[1].op != "get_attr"
                original_div, replacement = complex_div_replacement(x_pf, y_pf)

                def match_complex_div(
                    match: torch.fx.subgraph_rewriter.Match,
                    original_graph: object,
                    pattern_graph: object,
                ) -> bool:
                    for original_node in match.nodes_map.values():
                        if not isinstance(original_node, torch.fx.Node):
                            continue
                        if original_node.name == node.name:
                            return True
                    return False

                torch.fx.subgraph_rewriter.replace_pattern_with_filters(
                    self.gm,
                    original_div,
                    replacement,
                    match_filters=[match_complex_div],
                    ignore_literals=True,
                )
                return True

        # mul.Tensor, both nodes — complex × complex
        # Use SubgraphBuilder directly rather than replace_pattern_with_filters so
        # that self-multiplication (mul(x, x)) is handled correctly.
        # replace_pattern_with_filters requires distinct placeholder nodes for x and y,
        # so it silently produces no matches when both args are the same node.
        if node in self._originally_complex:
            x, y = node.args[0], node.args[1]
            x_is_get_attr = x.op == "get_attr"
            y_is_get_attr = y.op == "get_attr"

            if not x_is_get_attr and not y_is_get_attr:
                # Both are ITensors — use select.int (TRT-compatible)
                with SubgraphBuilder(self.gm.graph, node) as b:
                    x_re = b(torch.ops.aten.select.int, x, -1, 0)
                    x_im = b(torch.ops.aten.select.int, x, -1, 1)
                    y_re = b(torch.ops.aten.select.int, y, -1, 0)
                    y_im = b(torch.ops.aten.select.int, y, -1, 1)
                    ac = b(torch.ops.aten.mul.Tensor, x_re, y_re)
                    bd = b(torch.ops.aten.mul.Tensor, x_im, y_im)
                    ad = b(torch.ops.aten.mul.Tensor, x_re, y_im)
                    bc = b(torch.ops.aten.mul.Tensor, x_im, y_re)
                    out_re = b(torch.ops.aten.sub.Tensor, ac, bd)
                    out_im = b(torch.ops.aten.add.Tensor, ad, bc)
                    out = self._inline_cat_re_im(b, out_re, out_im)
                    node.replace_all_uses_with(out)
                    self.gm.graph.erase_node(node)
                return True
            else:
                # At least one arg is a get_attr buffer — fall back to the
                # pattern rewriter which uses tensor indexing for get_attr nodes.
                x_pf = not x_is_get_attr
                y_pf = not y_is_get_attr
                original_mul, replacement = complex_mul_replacement(x_pf, y_pf)

                def match_complex_mul(
                    match: torch.fx.subgraph_rewriter.Match,
                    original_graph: object,
                    pattern_graph: object,
                ) -> bool:
                    for original_node in match.nodes_map.values():
                        if not isinstance(original_node, torch.fx.Node):
                            continue
                        if original_node.name == node.name:
                            return True
                    return False

                torch.fx.subgraph_rewriter.replace_pattern_with_filters(
                    self.gm,
                    original_mul,
                    replacement,
                    match_filters=[match_complex_mul],
                    ignore_literals=True,
                )
                return True
        return False

    @_complex_unpacker(torch.ops.aten.add.Tensor, torch.ops.aten.sub.Tensor)
    def _rewrite_add_sub_tensor_scalar(self, node: Node) -> bool:
        # add.Tensor(z, scalar) / sub.Tensor(z, scalar): scalar applies to real part only.
        if len(node.args) < 2 or isinstance(node.args[1], torch.fx.Node):
            return False
        inp, scalar = node.args[0], node.args[1]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            new_re = b(node.target, re, scalar)
            out = self._inline_cat_re_im(b, new_re, im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten._conj.default)
    def _rewrite_conj(self, node: Node) -> bool:
        # conj(a+bi) = a - bi
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            neg_im = b(torch.ops.aten.neg.default, im)
            out = self._inline_cat_re_im(b, re, neg_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.abs.default)
    def _rewrite_abs(self, node: Node) -> bool:
        # |a+bi| = sqrt(a²+b²)
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            re2 = b(torch.ops.aten.mul.Tensor, re, re)
            im2 = b(torch.ops.aten.mul.Tensor, im, im)
            sum_ = b(torch.ops.aten.add.Tensor, re2, im2)
            out = b(torch.ops.aten.sqrt.default, sum_)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.exp.default)
    def _rewrite_exp(self, node: Node) -> bool:
        # exp(a+bi) = e^a*cos(b) + i*e^a*sin(b)
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            exp_re, exp_im = self._inline_complex_exp(b, re, im)
            out = self._inline_cat_re_im(b, exp_re, exp_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.log.default)
    def _rewrite_log(self, node: Node) -> bool:
        # log(a+bi) = 0.5*log(a²+b²) + i*atan2(b, a)
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            log_re, log_im = self._inline_complex_log(b, re, im)
            out = self._inline_cat_re_im(b, log_re, log_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.pow.Tensor_Scalar, torch.ops.aten.sqrt.default)
    def _rewrite_pow_sqrt(self, node: Node) -> bool:
        # pow(a+bi, n) / sqrt via polar form: r^n*(cos(n*θ) + i*sin(n*θ))
        inp = node.args[0]
        exponent = (
            node.args[1] if node.target == torch.ops.aten.pow.Tensor_Scalar else 0.5
        )
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            re2 = b(torch.ops.aten.mul.Tensor, re, re)
            im2 = b(torch.ops.aten.mul.Tensor, im, im)
            r2 = b(torch.ops.aten.add.Tensor, re2, im2)
            r = b(torch.ops.aten.sqrt.default, r2)
            rn = b(torch.ops.aten.pow.Tensor_Scalar, r, exponent)
            theta = b(torch.ops.aten.atan2.default, im, re)
            n_theta = b(torch.ops.aten.mul.Tensor, theta, exponent)
            cos_n = b(torch.ops.aten.cos.default, n_theta)
            sin_n = b(torch.ops.aten.sin.default, n_theta)
            out_re = b(torch.ops.aten.mul.Tensor, rn, cos_n)
            out_im = b(torch.ops.aten.mul.Tensor, rn, sin_n)
            out = self._inline_cat_re_im(b, out_re, out_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.real.default)
    def _rewrite_real(self, node: Node) -> bool:
        with SubgraphBuilder(self.gm.graph, node) as b:
            out = b(torch.ops.aten.select.int, node.args[0], -1, 0)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.imag.default)
    def _rewrite_imag(self, node: Node) -> bool:
        with SubgraphBuilder(self.gm.graph, node) as b:
            out = b(torch.ops.aten.select.int, node.args[0], -1, 1)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.angle.default)
    def _rewrite_angle(self, node: Node) -> bool:
        # angle(a+bi) = atan2(b, a)
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            out = b(torch.ops.aten.atan2.default, im, re)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.polar.default)
    def _rewrite_polar(self, node: Node) -> bool:
        # polar(r, theta) = r*cos(theta) + i*r*sin(theta)
        r_arg, theta_arg = node.args[0], node.args[1]
        with SubgraphBuilder(self.gm.graph, node) as b:
            cos_t = b(torch.ops.aten.cos.default, theta_arg)
            sin_t = b(torch.ops.aten.sin.default, theta_arg)
            out_re = b(torch.ops.aten.mul.Tensor, r_arg, cos_t)
            out_im = b(torch.ops.aten.mul.Tensor, r_arg, sin_t)
            out = self._inline_cat_re_im(b, out_re, out_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.add.Scalar, torch.ops.aten.sub.Scalar)
    def _rewrite_add_sub_scalar(self, node: Node) -> bool:
        # (a+bi) ± s = (a±s) + bi — scalar applies to real part only
        inp, scalar = node.args[0], node.args[1]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            new_re = b(node.target, re, scalar)
            out = self._inline_cat_re_im(b, new_re, im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.log2.default, torch.ops.aten.log10.default)
    def _rewrite_log2_log10(self, node: Node) -> bool:
        # log_b(z) = log(z) / log(b)
        base_val = (
            math.log(2.0)
            if node.target == torch.ops.aten.log2.default
            else math.log(10.0)
        )
        inp = node.args[0]
        inv_base = 1.0 / base_val
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            log_re, log_im = self._inline_complex_log(b, re, im)
            out_re = b(torch.ops.aten.mul.Tensor, log_re, inv_base)
            out_im = b(torch.ops.aten.mul.Tensor, log_im, inv_base)
            out = self._inline_cat_re_im(b, out_re, out_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.isnan.default, torch.ops.aten.isinf.default)
    def _rewrite_isnan_isinf(self, node: Node) -> bool:
        # isnan/isinf(z) = isnan/isinf(re) | isnan/isinf(im)
        inp = node.args[0]
        op = node.target
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            re_flag = b(op, re)
            im_flag = b(op, im)
            out = b(torch.ops.aten.logical_or.default, re_flag, im_flag)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.log1p.default)
    def _rewrite_log1p(self, node: Node) -> bool:
        # log1p(a+bi) = log((a+1) + bi)
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            re1 = b(torch.ops.aten.add.Tensor, re, 1.0)
            log_re, log_im = self._inline_complex_log(b, re1, im)
            out = self._inline_cat_re_im(b, log_re, log_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.expm1.default)
    def _rewrite_expm1(self, node: Node) -> bool:
        # expm1(a+bi) = (exp(a)*cos(b) - 1) + i*(exp(a)*sin(b))
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            exp_re, exp_im = self._inline_complex_exp(b, re, im)
            out_re = b(torch.ops.aten.sub.Tensor, exp_re, 1.0)
            out = self._inline_cat_re_im(b, out_re, exp_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.sin.default)
    def _rewrite_sin(self, node: Node) -> bool:
        # sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            sin_a = b(torch.ops.aten.sin.default, re)
            cosh_b = b(torch.ops.aten.cosh.default, im)
            cos_a = b(torch.ops.aten.cos.default, re)
            sinh_b = b(torch.ops.aten.sinh.default, im)
            out_re = b(torch.ops.aten.mul.Tensor, sin_a, cosh_b)
            out_im = b(torch.ops.aten.mul.Tensor, cos_a, sinh_b)
            out = self._inline_cat_re_im(b, out_re, out_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.cos.default)
    def _rewrite_cos(self, node: Node) -> bool:
        # cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            cos_a = b(torch.ops.aten.cos.default, re)
            cosh_b = b(torch.ops.aten.cosh.default, im)
            sin_a = b(torch.ops.aten.sin.default, re)
            sinh_b = b(torch.ops.aten.sinh.default, im)
            out_re = b(torch.ops.aten.mul.Tensor, cos_a, cosh_b)
            raw_im = b(torch.ops.aten.mul.Tensor, sin_a, sinh_b)
            out_im = b(torch.ops.aten.neg.default, raw_im)
            out = self._inline_cat_re_im(b, out_re, out_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.sinh.default)
    def _rewrite_sinh(self, node: Node) -> bool:
        # sinh(a+bi) = sinh(a)*cos(b) + i*cosh(a)*sin(b)
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            sinh_a = b(torch.ops.aten.sinh.default, re)
            cos_b = b(torch.ops.aten.cos.default, im)
            cosh_a = b(torch.ops.aten.cosh.default, re)
            sin_b = b(torch.ops.aten.sin.default, im)
            out_re = b(torch.ops.aten.mul.Tensor, sinh_a, cos_b)
            out_im = b(torch.ops.aten.mul.Tensor, cosh_a, sin_b)
            out = self._inline_cat_re_im(b, out_re, out_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.cosh.default)
    def _rewrite_cosh(self, node: Node) -> bool:
        # cosh(a+bi) = cosh(a)*cos(b) + i*sinh(a)*sin(b)
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            cosh_a = b(torch.ops.aten.cosh.default, re)
            cos_b = b(torch.ops.aten.cos.default, im)
            sinh_a = b(torch.ops.aten.sinh.default, re)
            sin_b = b(torch.ops.aten.sin.default, im)
            out_re = b(torch.ops.aten.mul.Tensor, cosh_a, cos_b)
            out_im = b(torch.ops.aten.mul.Tensor, sinh_a, sin_b)
            out = self._inline_cat_re_im(b, out_re, out_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.tan.default)
    def _rewrite_tan(self, node: Node) -> bool:
        # tan(a+bi) = sin(2a)/(cos(2a)+cosh(2b)) + i*sinh(2b)/(cos(2a)+cosh(2b))
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            two_re = b(torch.ops.aten.mul.Tensor, re, 2.0)
            two_im = b(torch.ops.aten.mul.Tensor, im, 2.0)
            sin_2a = b(torch.ops.aten.sin.default, two_re)
            cos_2a = b(torch.ops.aten.cos.default, two_re)
            sinh_2b = b(torch.ops.aten.sinh.default, two_im)
            cosh_2b = b(torch.ops.aten.cosh.default, two_im)
            denom = b(torch.ops.aten.add.Tensor, cos_2a, cosh_2b)
            out_re = b(torch.ops.aten.div.Tensor, sin_2a, denom)
            out_im = b(torch.ops.aten.div.Tensor, sinh_2b, denom)
            out = self._inline_cat_re_im(b, out_re, out_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.tanh.default)
    def _rewrite_tanh(self, node: Node) -> bool:
        # tanh(a+bi) = sinh(2a)/(cosh(2a)+cos(2b)) + i*sin(2b)/(cosh(2a)+cos(2b))
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re = b(torch.ops.aten.select.int, inp, -1, 0)
            im = b(torch.ops.aten.select.int, inp, -1, 1)
            two_re = b(torch.ops.aten.mul.Tensor, re, 2.0)
            two_im = b(torch.ops.aten.mul.Tensor, im, 2.0)
            sinh_2a = b(torch.ops.aten.sinh.default, two_re)
            cosh_2a = b(torch.ops.aten.cosh.default, two_re)
            sin_2b = b(torch.ops.aten.sin.default, two_im)
            cos_2b = b(torch.ops.aten.cos.default, two_im)
            denom = b(torch.ops.aten.add.Tensor, cosh_2a, cos_2b)
            out_re = b(torch.ops.aten.div.Tensor, sinh_2a, denom)
            out_im = b(torch.ops.aten.div.Tensor, sin_2b, denom)
            out = self._inline_cat_re_im(b, out_re, out_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.asinh.default)
    def _rewrite_asinh(self, node: Node) -> bool:
        # asinh(z) = log(z + sqrt(z² + 1))
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re, im = self._inline_select_re_im(b, inp)
            re2 = b(torch.ops.aten.mul.Tensor, re, re)
            im2 = b(torch.ops.aten.mul.Tensor, im, im)
            z2_re = b(torch.ops.aten.sub.Tensor, re2, im2)
            re_im = b(torch.ops.aten.mul.Tensor, re, im)
            z2_im = b(torch.ops.aten.mul.Tensor, re_im, 2.0)
            w_re = b(torch.ops.aten.add.Scalar, z2_re, 1.0)  # w = z²+1
            sq_re, sq_im = self._inline_complex_sqrt(b, w_re, z2_im)
            sum_re = b(torch.ops.aten.add.Tensor, re, sq_re)
            sum_im = b(torch.ops.aten.add.Tensor, im, sq_im)
            log_re, log_im = self._inline_complex_log(b, sum_re, sum_im)
            out = self._inline_cat_re_im(b, log_re, log_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.acosh.default)
    def _rewrite_acosh(self, node: Node) -> bool:
        # acosh(z) = log(z + sqrt(z² - 1))
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re, im = self._inline_select_re_im(b, inp)
            re2 = b(torch.ops.aten.mul.Tensor, re, re)
            im2 = b(torch.ops.aten.mul.Tensor, im, im)
            z2_re = b(torch.ops.aten.sub.Tensor, re2, im2)
            re_im = b(torch.ops.aten.mul.Tensor, re, im)
            z2_im = b(torch.ops.aten.mul.Tensor, re_im, 2.0)
            w_re = b(torch.ops.aten.sub.Scalar, z2_re, 1.0)  # w = z²-1
            sq_re, sq_im = self._inline_complex_sqrt(b, w_re, z2_im)
            sum_re = b(torch.ops.aten.add.Tensor, re, sq_re)
            sum_im = b(torch.ops.aten.add.Tensor, im, sq_im)
            log_re, log_im = self._inline_complex_log(b, sum_re, sum_im)
            out = self._inline_cat_re_im(b, log_re, log_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.atanh.default)
    def _rewrite_atanh(self, node: Node) -> bool:
        # atanh(z) = (1/2) * log((1+z) / (1-z))
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re, im = self._inline_select_re_im(b, inp)
            p_re = b(torch.ops.aten.add.Scalar, re, 1.0)  # 1+re
            q_re = b(torch.ops.aten.sub.Scalar, re, 1.0)  # re-1
            neg_q_re = b(torch.ops.aten.neg.default, q_re)  # 1-re
            neg_im = b(torch.ops.aten.neg.default, im)
            div_re, div_im = self._inline_complex_div(b, p_re, im, neg_q_re, neg_im)
            log_re, log_im = self._inline_complex_log(b, div_re, div_im)
            out_re = b(torch.ops.aten.mul.Tensor, log_re, 0.5)
            out_im = b(torch.ops.aten.mul.Tensor, log_im, 0.5)
            out = self._inline_cat_re_im(b, out_re, out_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.asin.default)
    def _rewrite_asin(self, node: Node) -> bool:
        # asin(z) = -i * log(iz + sqrt(1 - z²))
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re, im = self._inline_select_re_im(b, inp)
            iz_re = b(torch.ops.aten.neg.default, im)  # iz = (-im, re)
            re2 = b(torch.ops.aten.mul.Tensor, re, re)
            im2 = b(torch.ops.aten.mul.Tensor, im, im)
            z2_re = b(torch.ops.aten.sub.Tensor, re2, im2)
            re_im = b(torch.ops.aten.mul.Tensor, re, im)
            z2_im = b(torch.ops.aten.mul.Tensor, re_im, 2.0)
            ones = b(torch.ops.aten.ones_like.default, z2_re)
            w_re = b(torch.ops.aten.sub.Tensor, ones, z2_re)  # 1-z²
            w_im = b(torch.ops.aten.neg.default, z2_im)
            sq_re, sq_im = self._inline_complex_sqrt(b, w_re, w_im)
            sum_re = b(torch.ops.aten.add.Tensor, iz_re, sq_re)
            sum_im = b(torch.ops.aten.add.Tensor, re, sq_im)  # iz_im = re
            log_re, log_im = self._inline_complex_log(b, sum_re, sum_im)
            # -i*(log_re + i*log_im) = log_im + i*(-log_re)
            out_im = b(torch.ops.aten.neg.default, log_re)
            out = self._inline_cat_re_im(b, log_im, out_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.acos.default)
    def _rewrite_acos(self, node: Node) -> bool:
        # acos(z) = -i * log(z + i*sqrt(1 - z²))
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re, im = self._inline_select_re_im(b, inp)
            re2 = b(torch.ops.aten.mul.Tensor, re, re)
            im2 = b(torch.ops.aten.mul.Tensor, im, im)
            z2_re = b(torch.ops.aten.sub.Tensor, re2, im2)
            re_im = b(torch.ops.aten.mul.Tensor, re, im)
            z2_im = b(torch.ops.aten.mul.Tensor, re_im, 2.0)
            ones = b(torch.ops.aten.ones_like.default, z2_re)
            w_re = b(torch.ops.aten.sub.Tensor, ones, z2_re)  # 1-z²
            w_im = b(torch.ops.aten.neg.default, z2_im)
            sq_re, sq_im = self._inline_complex_sqrt(b, w_re, w_im)
            isq_re = b(torch.ops.aten.neg.default, sq_im)  # i*sqrt = (-sq_im, sq_re)
            sum_re = b(torch.ops.aten.add.Tensor, re, isq_re)
            sum_im = b(torch.ops.aten.add.Tensor, im, sq_re)
            log_re, log_im = self._inline_complex_log(b, sum_re, sum_im)
            # -i*(log_re + i*log_im) = log_im + i*(-log_re)
            out_im = b(torch.ops.aten.neg.default, log_re)
            out = self._inline_cat_re_im(b, log_im, out_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.atan.default)
    def _rewrite_atan(self, node: Node) -> bool:
        # atan(z) = (i/2) * log((1-iz) / (1+iz))
        inp = node.args[0]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re, im = self._inline_select_re_im(b, inp)
            iz_re = b(torch.ops.aten.neg.default, im)  # iz = (-im, re)
            ones = b(torch.ops.aten.ones_like.default, re)
            p_re = b(torch.ops.aten.sub.Tensor, ones, iz_re)  # 1-iz
            p_im = b(torch.ops.aten.neg.default, re)
            q_re = b(torch.ops.aten.add.Tensor, ones, iz_re)  # 1+iz
            q_im = re  # iz_im = re
            div_re, div_im = self._inline_complex_div(b, p_re, p_im, q_re, q_im)
            log_re, log_im = self._inline_complex_log(b, div_re, div_im)
            # (i/2)*(log_re+i*log_im) = (-log_im/2) + i*(log_re/2)
            out_re = b(torch.ops.aten.mul.Tensor, log_im, -0.5)
            out_im = b(torch.ops.aten.mul.Tensor, log_re, 0.5)
            out = self._inline_cat_re_im(b, out_re, out_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.pow.Tensor_Tensor)
    def _rewrite_pow_tensor_tensor(self, node: Node) -> bool:
        # z1**z2 = exp(z2 * log(z1))
        z1_inp, z2_inp = node.args[0], node.args[1]
        with SubgraphBuilder(self.gm.graph, node) as b:
            re1, im1 = self._inline_select_re_im(b, z1_inp)
            re2 = b(torch.ops.aten.select.int, z2_inp, -1, 0)
            im2 = b(torch.ops.aten.select.int, z2_inp, -1, 1)
            log_re, log_im = self._inline_complex_log(b, re1, im1)
            mul_re, mul_im = self._inline_complex_mul(b, re2, im2, log_re, log_im)
            exp_re, exp_im = self._inline_complex_exp(b, mul_re, mul_im)
            out = self._inline_cat_re_im(b, exp_re, exp_im)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.scalar_tensor.default)
    def _rewrite_scalar_tensor(self, node: Node) -> bool:
        # scalar_tensor(val, dtype=complex64) → scalar_tensor(0.0, float32)
        if dict(node.kwargs).get("dtype") not in (torch.complex64, torch.complex128):
            return False
        with SubgraphBuilder(self.gm.graph, node) as b:
            out = b(torch.ops.aten.scalar_tensor.default, 0.0)
            out.kwargs = {"dtype": torch.float32}  # type: ignore[assignment]
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    @_complex_unpacker(torch.ops.aten.where.self)
    def _rewrite_where(self, node: Node) -> bool:
        # where.self: unsqueeze mask and optionally expand true-branch for complex layout.
        if len(node.args) != 3:
            return False
        node_val = node.meta.get("val", None)
        if node_val is None or not hasattr(node_val, "dtype"):
            return False
        if node_val.dtype not in (torch.complex64, torch.complex128):
            return False
        mask_node, true_node, other_node = node.args
        target_shape = list(node_val.shape) + [2]
        with SubgraphBuilder(self.gm.graph, node) as b:
            mask_unsq = b(torch.ops.aten.unsqueeze.default, mask_node, -1)
            true_arg = true_node
            if isinstance(true_node, torch.fx.Node):
                true_val = true_node.meta.get("val", None)
                if (
                    true_val is not None
                    and hasattr(true_val, "shape")
                    and list(true_val.shape) == [2]
                ):
                    true_arg = b(torch.ops.aten.expand.default, true_node, target_shape)
            out = b(torch.ops.aten.where.self, mask_unsq, true_arg, other_node)
            node.replace_all_uses_with(out)
            self.gm.graph.erase_node(node)
            return True

    def rewrite_subgraph_nodes(self, subgraphs: List[ComplexSubGraphInfo]) -> None:
        modified = False
        # Detect the existing FakeTensorMode from the graph's placeholders
        # *before* any rewrites.  We pass this to replace_input_node so that
        # new placeholder fake tensors are created under the same mode as the
        # rest of the graph.  Using a fresh FakeTensorMode would cause "mode
        # mismatch" assertions under torch.compile (where a mode is already
        # active) and would lose SymInt information for torch.export graphs.
        detected_fake_mode = torch._export.utils._detect_fake_mode_from_gm(self.gm)

        # Record the set of all nodes that have complex dtype BEFORE any rewriting.
        # This is needed because after replace_input_node (which changes dtype from
        # complex to float32), is_complex_dtype() would return False for those nodes —
        # but we still need to know they were originally complex when we later decide
        # whether a mul.Tensor operand should be treated as complex-layout.
        detector = ComplexOpDetector()
        self._originally_complex: Set[Node] = set()
        for subgraph in subgraphs:
            for node in subgraph.input_nodes:
                if detector.is_complex_dtype(node):
                    self._originally_complex.add(node)
            for node in subgraph.subgraph_nodes:
                if detector.is_complex_dtype(node):
                    self._originally_complex.add(node)

        # _DISPATCH maps op -> unbound method; bind self here once per call.
        dispatch = {op: method.__get__(self) for op, method in self._DISPATCH.items()}

        logger.debug(
            "complex_graph_rewrite  begin  subgraphs=%d  nodes=%s",
            len(subgraphs),
            [n.name for s in subgraphs for n in s.subgraph_nodes],
        )

        for subgraph in subgraphs:
            for input_node in subgraph.input_nodes:
                if input_node.op not in ("call_function"):
                    # Only rewrite inputs that are themselves complex — real inputs
                    # to complex-output ops (e.g. r, theta for polar) must NOT be
                    # renamed to *_unpacked_complex.
                    if not detector.is_complex_dtype(input_node):
                        continue
                    self.replace_input_node(input_node, fake_mode=detected_fake_mode)
            for node in subgraph.subgraph_nodes:
                # Skip nodes that were already erased by a previous pattern replacement
                if node.graph is not self.gm.graph:
                    continue
                handler = dispatch.get(node.target)
                if handler is not None:
                    if handler(node):
                        modified = True
                elif node.target in _ELEMENTWISE_SAFE:
                    logger.debug("  pass-through  %s  (elementwise-safe)", node.name)
                else:
                    logger.warning(
                        "Complex op '%s' has no explicit rewrite rule. "
                        "It will be passed through as-is on the real [..., 2] layout, "
                        "which may produce incorrect results or fail TRT compilation. "
                        "Consider adding a rewrite in complex_graph_rewrite.py.",
                        node.target,
                    )
        if modified:
            # After rewriting complex ops, any view_as_real node that now receives a
            # real tensor must be erased. The subgraph_rewriter replaces the original
            # complex mul with a cat of real/imag parts; view_as_real on that result
            # is invalid. We detect this by checking whether the input to view_as_real
            # is no longer complex-typed (its meta val dtype is real, or has no val yet
            # but its target is the real-arithmetic cat output).
            for node in list(self.gm.graph.nodes):
                if node.target != torch.ops.aten.view_as_real.default:
                    continue
                inp = node.args[0]
                if not isinstance(inp, torch.fx.Node):
                    continue
                inp_val = inp.meta.get("val", None)
                # If meta is available and dtype is real, erase view_as_real
                is_real_input = (
                    inp_val is not None
                    and hasattr(inp_val, "dtype")
                    and inp_val.dtype not in {torch.complex64, torch.complex128}
                )
                # If meta not yet propagated, use the target as a heuristic:
                # the real-arithmetic replacement ends with aten.cat.default
                if inp_val is None:
                    is_real_input = inp.target == torch.ops.aten.cat.default
                if is_real_input:
                    inp_desc = (
                        f"{inp.name}[{tuple(inp_val.shape)},{inp_val.dtype}]"
                        if inp_val is not None and hasattr(inp_val, "shape")
                        else inp.name
                    )
                    logger.debug(
                        "  erase view_as_real  %s  (input %s is already real)",
                        node.name,
                        inp_desc,
                    )
                    node.replace_all_uses_with(inp)
                    self.gm.graph.erase_node(node)
            logger.debug("complex_graph_rewrite  propagating metadata")
            self.propagate_metadata(detected_fake_mode)
            self.gm.graph.lint()
            self.gm.recompile()
            logger.debug("complex_graph_rewrite  done")

    def propagate_metadata(
        self, existing_fake_mode: Optional[FakeTensorMode] = None
    ) -> None:
        """Re-propagate FakeTensor metadata after graph rewrites via FakeTensorProp.

        Uses *existing_fake_mode* (detected from the graph's placeholder fake
        tensors) when available.  This ensures the propagation mode matches the
        mode under which the graph was originally traced — critical for both
        torch.compile (where a FakeTensorMode is already active on the thread)
        and torch.export (where we must preserve the ShapeEnv / SymInt ranges).

        Falls back to a fresh FakeTensorMode only for plain FX graphs that have
        no fake tensor metadata at all.
        """
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp

        fake_inputs = []
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                if "val" in node.meta:
                    fake_val = node.meta["val"]
                    fake_inputs.append(
                        fake_val.to("cuda")
                        if fake_val.device.type == "cuda"
                        else fake_val
                    )
                else:
                    fake_tensor = torch.empty(
                        [s if s != 0 else 1 for s in node.meta["tensor_meta"].shape],
                        dtype=node.meta["tensor_meta"].dtype,
                        device=node.meta["tensor_meta"].device,
                    )
                    fake_inputs.append(fake_tensor)

        prop_mode = (
            existing_fake_mode
            if existing_fake_mode is not None
            else FakeTensorMode(allow_non_fake_inputs=True)
        )
        FakeTensorProp(self.gm, mode=prop_mode).propagate(*fake_inputs)


def extract_real_imag(input, placeholder_or_func: bool = True):  # type: ignore
    """Extract real and imaginary parts from a tensor.
    This function handles different tensor types based on whether they are placeholder/function
    tensors or get_attr tensors. For placeholder/function tensors, it uses select operations,
    while for get_attr tensors, it uses indexing.
    Args:
        input: Input tensor to extract real and imaginary parts from
        placeholder_or_func: Boolean flag indicating if the input is a placeholder/function tensor (True)
                           or a get_attr tensor (False). Defaults to True.
    Returns:
        Tuple of (real_part, imaginary_part) where both parts have the same type as the input
    Note:
        - When placeholder_or_func=True: Uses torch.ops.aten.select.int operations
        - When placeholder_or_func=False: Uses tensor indexing [..., 0] and [..., 1]
    """
    if placeholder_or_func:
        # For ITensor, use select operations
        real_part = torch.ops.aten.select.int(input, -1, 0)
        imag_part = torch.ops.aten.select.int(input, -1, 1)
        return real_part, imag_part
    else:
        # For get_attr, use indexing
        return input[..., 0], input[..., 1]


def complex_mul_replacement(
    x_placeholder_or_func: bool = True, y_placeholder_or_func: bool = True
) -> Tuple[
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
]:
    """Constructs the original and replacement functions for complex multiplication.

    The original functions correspond to native complex multiplication
    via torch.mul or operator.mul on complex tensors.

    The replacement function assumes x and y are real tensors with the last
    dimension size 2 representing real and imaginary parts, and performs
    complex multiplication manually returning the same shaped tensor.
    """

    # Original pattern: torch.mul for complex tensors
    def original_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.mul.Tensor(x, y)

    # Replacement function: manual complex multiplication on real/imag stacked tensors
    def replacement(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_real, x_imag = extract_real_imag(x, x_placeholder_or_func)
        y_real, y_imag = extract_real_imag(y, y_placeholder_or_func)

        real_part1 = torch.ops.aten.mul.Tensor(x_real, y_real)
        real_part2 = torch.ops.aten.mul.Tensor(x_imag, y_imag)
        real = torch.ops.aten.sub.Tensor(real_part1, real_part2)

        imag_part1 = torch.ops.aten.mul.Tensor(x_real, y_imag)
        imag_part2 = torch.ops.aten.mul.Tensor(x_imag, y_real)
        imag = torch.ops.aten.add.Tensor(imag_part1, imag_part2)

        return torch.ops.aten.cat.default(
            [
                torch.ops.aten.unsqueeze.default(real, -1),
                torch.ops.aten.unsqueeze.default(imag, -1),
            ],
            -1,
        )

    return (original_mul, replacement)


def complex_div_replacement(
    x_placeholder_or_func: bool = True, y_placeholder_or_func: bool = True
) -> Tuple[
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
]:
    """Constructs the original and replacement functions for complex division.

    (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)
    """

    def original_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.div.Tensor(x, y)

    def replacement(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_real, x_imag = extract_real_imag(x, x_placeholder_or_func)
        y_real, y_imag = extract_real_imag(y, y_placeholder_or_func)

        denom = torch.ops.aten.add.Tensor(
            torch.ops.aten.mul.Tensor(y_real, y_real),
            torch.ops.aten.mul.Tensor(y_imag, y_imag),
        )
        real = torch.ops.aten.div.Tensor(
            torch.ops.aten.add.Tensor(
                torch.ops.aten.mul.Tensor(x_real, y_real),
                torch.ops.aten.mul.Tensor(x_imag, y_imag),
            ),
            denom,
        )
        imag = torch.ops.aten.div.Tensor(
            torch.ops.aten.sub.Tensor(
                torch.ops.aten.mul.Tensor(x_imag, y_real),
                torch.ops.aten.mul.Tensor(x_real, y_imag),
            ),
            denom,
        )

        return torch.ops.aten.cat.default(
            [
                torch.ops.aten.unsqueeze.default(real, -1),
                torch.ops.aten.unsqueeze.default(imag, -1),
            ],
            -1,
        )

    return (original_div, replacement)


def _get_complex_output_indices(gm: GraphModule) -> List[int]:
    """Return indices of output nodes that have complex dtype, before rewriting."""
    complex_dtypes = {torch.complex64, torch.complex128}
    output_node = next((n for n in reversed(gm.graph.nodes) if n.op == "output"), None)
    if output_node is None:
        return []
    # output args is a tuple of the return values
    outputs = output_node.args[0]
    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)
    indices = []
    for i, out in enumerate(outputs):
        if isinstance(out, torch.fx.Node) and "val" in out.meta:
            val = out.meta["val"]
            if hasattr(val, "dtype") and val.dtype in complex_dtypes:
                indices.append(i)
    return indices


def _get_complex_input_names(gm: GraphModule) -> List[str]:
    """Return the original names of placeholder nodes that have complex dtype, before rewriting.

    complex_graph_detection renames complex placeholders from 'name' to 'name_unpacked_complex'
    and changes their dtype to float. This captures the original names so the post-partition
    pass can insert view_as_real at the graph input boundary.
    """
    complex_dtypes = {torch.complex64, torch.complex128}
    names = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val", None)
        if val is not None and hasattr(val, "dtype") and val.dtype in complex_dtypes:
            names.append(node.name)
    return names


def _get_complex_input_dtypes(gm: GraphModule) -> dict:
    """Return a mapping of placeholder name -> complex dtype for complex-dtype inputs.

    Used by the post-partition boundary pass to know which inputs were complex128
    so it can insert float32 casts when truncate_double=True.
    """
    complex_dtypes = {torch.complex64, torch.complex128}
    dtypes = {}
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val", None)
        if val is not None and hasattr(val, "dtype") and val.dtype in complex_dtypes:
            dtypes[node.name] = val.dtype
    return dtypes


# This lowering pass is used to detect and rewrite complex subgraphs in the graph
def complex_graph_detection(
    gm: GraphModule, settings: CompilationSettings
) -> GraphModule:
    """Detect and rewrite complex subgraphs in the graph.
    This lowering pass is used to detect and rewrite complex subgraphs in the graph.
    This lowering pass works for complex tensor in mul which are parameter or buffers in the graph.
    Args:
        gm: The GraphModule to process
        settings: Compilation settings
    Returns:
        The modified GraphModule with complex subgraphs rewritten
    """
    # Capture I/O signature before rewriting — used post-partition to restore
    # the complex tensor interface at the graph boundaries.
    gm.meta["complex_output_indices"] = _get_complex_output_indices(gm)
    gm.meta["complex_input_names"] = _get_complex_input_names(gm)
    gm.meta["complex_input_dtypes"] = _get_complex_input_dtypes(gm)
    if gm.meta["complex_output_indices"]:
        logger.debug(
            f"Complex output indices captured: {gm.meta['complex_output_indices']}"
        )
    if gm.meta["complex_input_names"]:
        logger.debug(f"Complex input names captured: {gm.meta['complex_input_names']}")

    complex_op_detector = ComplexOpDetector()
    complex_subgraphs = complex_op_detector.find_all_complex_subgraphs(gm)
    for subgraph in complex_subgraphs:
        logger.debug(f"Complex subgraph info: {subgraph}")
    complex_graph_rewriter = ComplexGraphRewriter(gm, settings.truncate_double)
    complex_graph_rewriter.rewrite_subgraph_nodes(complex_subgraphs)
    return gm
