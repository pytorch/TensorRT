# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import logging
import operator
from typing import Any, Dict, Optional, Tuple

import torch
from torch._export.utils import _detect_fake_mode_from_gm
from torch.fx.experimental.const_fold import get_unique_attr_name_in_module
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from .pass_utils import (
    clean_up_graph_after_modifications,
    tensors_maybe_written_by,
    tensors_written_by,
)
from .remove_assert_nodes import apply_remove_assert_nodes

logger = logging.getLogger(__name__)

# Inlining can rename an op, as matmul to matmul_1, so autocast_excluded_node_names
# also matches the name it had in its region.
NAME_IN_REGION = "_torch_tensorrt_name_in_autocast_region"


def _aliases_an_input(node: torch.fx.Node) -> bool:
    return isinstance(node.target, torch._ops.OpOverload) and any(
        r.alias_info is not None for r in node.target._schema.returns
    )


def _copies(node: torch.fx.Node) -> bool:
    """Whether node is a cast, such as x.float(), that changed the dtype in its region
    and so made a copy. Inlining keeps that copy."""
    return (
        isinstance(node.target, torch._ops.OpOverload)
        and node.target.overloadpacket is torch.ops.aten.to
        and node.args[0].meta["val"].dtype != node.meta["val"].dtype
    )


def _region_output(value: torch.fx.Node) -> Optional[torch.fx.Node]:
    """For the getitem of a region's output, the value its body returns."""
    region = value.args[0] if value.target == operator.getitem else None
    if region is None or region.target != torch.ops.higher_order.wrap_with_autocast:
        return None
    body = getattr(value.graph.owning_module, region.args[4].target)
    return body.graph.output_node().args[0][value.args[1]]


def _base(value: torch.fx.Node) -> torch.fx.Node:
    """The value whose memory a value shares, through views, in-place ops and nested
    regions."""
    output = _region_output(value)
    if output is not None:
        base = _base(output)
        placeholders = list(base.graph.find_nodes(op="placeholder"))
        if base not in placeholders:
            return base
        return _base(value.args[0].args[5 + placeholders.index(base)])
    op = value.args[0] if value.target == operator.getitem else value
    if _aliases_an_input(op) and not _copies(op):
        return _base((tensors_written_by(op) or [op.args[0]])[0])
    return value


def _is_written(gm: torch.fx.GraphModule, value: torch.fx.Node) -> bool:
    """Whether an op of gm writes in place into the memory of value, through value or
    any tensor that shares that memory."""
    base = _base(value)
    shared = {n for n in gm.graph.nodes if _base(n) is base}
    return any(w in shared for n in gm.graph.nodes for w in tensors_maybe_written_by(n))


def _may_read_storage(node: torch.fx.Node) -> bool:
    """Whether node can read the memory around the elements of a tensor it is given:
    an op that takes a storage offset, such as as_strided, an op from outside aten,
    such as a custom op, whose code is not in the graph, or a higher order op, such
    as a torch.no_grad block, whose body has such an op."""
    if isinstance(node.target, torch._ops.OpOverload):
        return node.target.namespace != "aten" or any(
            a.name == "storage_offset" for a in node.target._schema.arguments
        )
    if not isinstance(node.target, torch._ops.HigherOrderOperator):
        return False
    bodies = [
        operator.attrgetter(a.target)(node.graph.owning_module)
        for a in node.args
        if isinstance(a, torch.fx.Node) and a.op == "get_attr"
    ]
    return any(
        _may_read_storage(n)
        for body in bodies
        if isinstance(body, torch.fx.GraphModule)
        for n in body.graph.nodes
    )


def _reads_storage(gm: torch.fx.GraphModule, value: torch.fx.Node) -> bool:
    """Whether an op of gm can read the memory around the elements of value, through
    value or any tensor that shares that memory."""
    base = _base(value)
    shared = {n for n in gm.graph.nodes if _base(n) is base}
    return any(
        not shared.isdisjoint(n.all_input_nodes) and _may_read_storage(n)
        for n in gm.graph.nodes
    )


def _writes_only_a_new_value(
    value: Any,
    writer: torch.fx.Node,
    operands: Dict[torch.fx.Node, Tuple[Any, torch.fx.Node]],
) -> bool:
    """Whether writer can write in place into value: a new tensor, or the result of
    in-place ops on one, that no other op keeps a view of. operands maps an input of a
    region body to the value and the region it comes from.

    Other ops can read value only before the write, as in y.div_(y.norm()). Export
    makes the ops after add_ read its result. The ops after _foreach_add_, which
    returns nothing, read value itself and would miss a write into a cast.
    """
    while isinstance(value, torch.fx.Node):
        if not all(
            u is writer
            or (
                u < writer
                and isinstance(u.target, torch._ops.OpOverload)
                and (not _aliases_an_input(u) or _copies(u))
            )
            for u in value.users
        ):
            return False
        if value in operands:
            value, writer = operands[value]
            continue
        output = _region_output(value)
        if output is not None:
            # The output of a region that gets inlined too is a value of its body.
            if not _is_inlined(value.graph.owning_module, value.args[0]):
                return False
            value, writer = output, output.graph.output_node()
            continue
        written = tensors_written_by(value)
        if len(written) != 1:
            op = value.args[0] if value.target == operator.getitem else value
            return isinstance(op.target, torch._ops.OpOverload) and (
                not _aliases_an_input(op) or _copies(op)
            )
        value, writer = written[0], value
    return False


def _can_inline(
    gm: torch.fx.GraphModule,
    region: torch.fx.Node,
    operands: Optional[Dict[torch.fx.Node, Tuple[Any, torch.fx.Node]]] = None,
) -> bool:
    """Whether a region computes the same values without its autocast.

    Autocast later casts the inputs of an op to the precision it chooses. An in-place
    op would then write into the cast, so the region must write only into new values
    that no other op keeps a view of, not into a buffer or an input of the model. The casts at
    the region's boundary would separate outputs that share memory, and view(dtype)
    reads the bytes of the dtype autocast gave. An output that shares memory with an
    input of the region, such as a slice of it, could become a view of a cast of that
    input. The cast of an output that is a view, such as a strided slice, can be a new
    contiguous tensor, which an op such as contiguous() returns as it is, where it
    copied the view. All of these would see a later write into that memory wrongly, so
    a region with such outputs is inlined only when nothing writes into the memory they
    share. The cast of an output that is a view holds only the elements of the view,
    so a region with such an output is inlined only when no op can read the memory
    around them, see _may_read_storage.
    """
    # wrap_with_autocast(device_type, dtype, enabled, cache_enabled, body, *operands)
    body, args = getattr(gm, region.args[4].target), region.args[5:]
    operands = {
        **(operands or {}),
        **{
            p: (a, region)
            for p, a in zip(body.graph.find_nodes(op="placeholder"), args)
        },
    }
    outputs = body.graph.output_node().args[0]
    bases = [_base(o) if isinstance(o, torch.fx.Node) else o for o in outputs]
    placeholders = set(body.graph.find_nodes(op="placeholder"))
    for user in region.users:
        output, base = outputs[user.args[1]], bases[user.args[1]]
        if (
            bases.count(base) > 1 or base in placeholders or base is not output
        ) and _is_written(gm, user):
            return False
        if base is not output and _reads_storage(gm, user):
            return False
    for node in body.graph.nodes:
        if node.target == torch.ops.aten.view.dtype:
            return False
        if node.target == torch.ops.higher_order.wrap_with_autocast and not (
            _can_inline(body, node, operands)
        ):
            return False
        if not all(
            _writes_only_a_new_value(w, node, operands)
            for w in tensors_written_by(node)
        ):
            return False
    return True


def _is_inlined(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Whether node is a region that gets inlined. A region with enabled=False is
    how a model keeps ops in FP32, so it keeps its autocast."""
    return (
        node.target == torch.ops.higher_order.wrap_with_autocast
        and bool(node.args[2])
        and _can_inline(gm, node)
    )


def _cast(
    gm: torch.fx.GraphModule,
    value: torch.fx.Node,
    dtype: torch.dtype,
    user: torch.fx.Node,
) -> torch.fx.Node:
    cast = gm.graph.call_function(torch.ops.aten.to.dtype, (value, dtype))
    # Autocast reads the module of every node, as for autocast_excluded_nodes. A
    # value can be a placeholder, which has none, so take the module of its user.
    cast.meta["nn_module_stack"] = user.meta["nn_module_stack"]
    return cast


def _inline_region(
    gm: torch.fx.GraphModule, region: torch.fx.Node, copied: Dict[torch.fx.Node, bool]
) -> None:
    device_type, body_node = region.args[0], region.args[4]
    autocast_key = getattr(torch._C.DispatchKey, f"Autocast{device_type.upper()}")
    body = getattr(gm, body_node.target)
    env = dict(zip(body.graph.find_nodes(op="placeholder"), region.args[5:]))
    with gm.graph.inserting_before(region):
        for node in body.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "output":
                outputs = node.args[0]
            elif node.op == "get_attr":
                # The body of a nested region, which the next round inlines.
                name = get_unique_attr_name_in_module(gm, node.target)
                setattr(gm, name, operator.attrgetter(node.target)(body))
                env[node] = gm.graph.get_attr(name)
            else:
                env[node] = gm.graph.node_copy(node, env.__getitem__)
                env[node].meta[NAME_IN_REGION] = node.name
                if _copies(node):
                    # to() returns its input when that already has the dtype, as it
                    # now can. Keep the copy, which a view of the result can need.
                    names = [a.name for a in node.target._schema.arguments]
                    if names.index("copy") < len(node.args):
                        env[node].update_arg(names.index("copy"), True)
                    else:
                        env[node].update_kwarg("copy", True)
                # Ops such as matmul and lerp need their floating point inputs in one
                # dtype. Autocast gave them one when it has a kernel for the op, which
                # casts them, or when they all had one dtype in the region.
                dtypes = {
                    a.meta["val"].dtype
                    for a in node.all_input_nodes
                    if isinstance(a.meta.get("val"), torch.Tensor)
                    and a.meta["val"].is_floating_point()
                }
                copied[env[node]] = len(dtypes) == 1 or (
                    isinstance(node.target, torch._ops.OpOverload)
                    and node.target.has_kernel_for_dispatch_key(autocast_key)
                )
        # Each user of a region takes one of its outputs with getitem. Ops get the
        # output in the dtype the region gave it, which the rest of the graph expects,
        # and regions that get inlined too get it as it is.
        for user in list(region.users):
            output = env[outputs[user.args[1]]]
            val = user.meta.get("val")
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                cast = _cast(gm, output, val.dtype, user)
                cast.meta["val"] = val
                user.replace_all_uses_with(
                    cast, delete_user_cb=lambda u: not _is_inlined(gm, u)
                )
                if not cast.users:
                    gm.graph.erase_node(cast)
            user.replace_all_uses_with(output)
            gm.graph.erase_node(user)
    gm.graph.erase_node(region)
    # Export gives each region a body of its own, which nothing uses now.
    gm.graph.erase_node(body_node)
    delattr(gm, body_node.target)


class _MatchInputDtypes(FakeTensorProp):  # type: ignore[misc]
    """FakeTensorProp that also casts floating point inputs that differ in dtype, of
    the given ops that autocast gave them in one dtype, to their promoted dtype.
    Other ops keep inputs that differ, such as a float8 matmul with float32 scales.
    A region that keeps its autocast gets its inputs in the dtypes it was traced with.
    """

    def __init__(
        self, gm: torch.fx.GraphModule, nodes: Dict[torch.fx.Node, bool]
    ) -> None:
        super().__init__(gm, _detect_fake_mode_from_gm(gm))
        self.nodes = nodes

    def _cast_input(
        self, n: torch.fx.Node, a: torch.fx.Node, dtype: torch.dtype
    ) -> None:
        with self.graph.inserting_before(n):
            cast = _cast(self.module, a, dtype, n)
        self.env[cast] = cast.meta["val"] = self.env[a].to(dtype)
        n.replace_input_with(a, cast)

    def run_node(self, n: torch.fx.Node) -> Any:
        if n.target == torch.ops.higher_order.wrap_with_autocast:
            # An inlined op can now give it an input in another dtype. Its body
            # asserts the traced dtypes, and its ops can depend on them, as x.float()
            # copies x only when x has another dtype.
            body = getattr(self.module, n.args[4].target)
            for p, a in zip(body.graph.find_nodes(op="placeholder"), n.args[5:]):
                val = self.env[a]
                if isinstance(val, torch.Tensor) and val.dtype != p.meta["val"].dtype:
                    self._cast_input(n, a, p.meta["val"].dtype)
        elif self.nodes.get(n) and isinstance(n.target, torch._ops.OpOverload):
            inputs = [
                a
                for a in n.all_input_nodes
                if isinstance(self.env[a], torch.Tensor)
                and self.env[a].is_floating_point()
            ]
            dtypes = {self.env[a].dtype for a in inputs}
            if len(dtypes) > 1:
                dtype = functools.reduce(torch.promote_types, dtypes)
                dims = {self.env[a].dtype for a in inputs if self.env[a].dim()}
                # As in type promotion, a 0-dim float64 input, such as a constant, does
                # not make the op float64. Otherwise no input gets a narrower dtype,
                # which would round it before autocast chooses the precision of the op.
                if dtype == torch.float64 and dims and torch.float64 not in dims:
                    dtype = torch.float32
                for a in inputs:
                    if self.env[a].dtype != dtype:
                        self._cast_input(n, a, dtype)
        return super().run_node(n)


def inline_autocast_regions(gm: torch.fx.GraphModule) -> None:
    """Replace each torch.autocast region of the graph with its body, without the region's autocast.

    Torch-TensorRT autocast then chooses the precision of the body's ops like that
    of any other op. Where autocast gave the floating point inputs of an op one
    dtype, by casting them as for matmul or by computing them in it, and they now
    differ, they are cast to their promoted dtype. The rest of the graph gets
    each output of a region in the dtype the region gave it. A region whose values
    inlining could change keeps its autocast, see _can_inline. A region that keeps
    its autocast gets its inputs in the dtypes it was traced with.
    """
    copied: Dict[torch.fx.Node, bool] = {}
    count = 0
    while regions := [node for node in gm.graph.nodes if _is_inlined(gm, node)]:
        for region in regions:
            _inline_region(gm, region, copied)
        count += len(regions)
    if not count:
        return
    # The asserts of the inlined bodies check the dtypes that autocast gave their
    # values, and pre_export_lowering removes all asserts after calibration anyway.
    apply_remove_assert_nodes(gm)
    # Export marks a buffer that an op writes into from a value that requires grad
    # as a leaf that requires grad, and autograd refuses that write. Propagation
    # needs only the dtypes and shapes.
    with torch.no_grad():
        _MatchInputDtypes(gm, copied).propagate_dont_convert_inputs(
            *[node.meta["val"] for node in gm.graph.find_nodes(op="placeholder")]
        )
    clean_up_graph_after_modifications(gm)
    logger.debug(f"Inlined {count} autocast regions")
