# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import logging
import operator
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch._export.utils import _detect_fake_mode_from_gm
from torch._guards import detect_fake_mode
from torch.fx.experimental.const_fold import get_unique_attr_name_in_module
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.multiprocessing.reductions import StorageWeakRef

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
# Inlining makes a type_as that copies a to, so autocast_excluded_ops also matches
# the op it was in its region.
OP_IN_REGION = "_torch_tensorrt_op_in_autocast_region"
# The traced values of the inputs of an op and its result when it ran on them again,
# see _runs_again.
_RAN_AGAIN = "_torch_tensorrt_ran_again"


def _storages(val: Any) -> Set[StorageWeakRef]:
    """The memory of the tensors in a traced value."""
    return {
        StorageWeakRef(v.untyped_storage())
        for v in (val if isinstance(val, (list, tuple)) else [val])
        if isinstance(v, torch.Tensor) and torch._C._has_storage(v)
    }


def _can_return_an_input(op: torch.fx.Node) -> bool:
    """Whether op is an aten op whose kernel is its decomposition, such as type_as,
    dropout or einsum, which can return an input though its schema does not say it.
    Other ops follow their schema."""
    return (
        isinstance(op.target, torch._ops.OpOverload)
        and op.target.namespace == "aten"
        and all(
            a.alias_info is None
            for a in (*op.target._schema.arguments, *op.target._schema.returns)
        )
        and op.target.has_kernel_for_dispatch_key(
            torch._C.DispatchKey.CompositeImplicitAutograd
        )
    )


def _runs_again(op: torch.fx.Node) -> bool:
    """Whether op, which can return an input, runs again on the traced values of its
    inputs to show the memory its result shares, since a program loaded with
    torch.export.load gives each traced value memory of its own. It runs without
    autocast, as it does once inlined: in a float16 region, einsum of a float32 input
    returns a view of a cast, and without autocast a view of the input. A result with
    a data dependent shape would get a new symbol in a second run, and an op whose
    floating point inputs differ in dtype, which autocast allowed, can fail outside
    autocast, so these do not run again."""
    inputs = [a.meta.get("val") for a in op.all_input_nodes]
    dtypes = {v.dtype for v in inputs if isinstance(v, torch.Tensor)}
    return (
        _can_return_an_input(op)
        and not free_unbacked_symbols(op.meta.get("val"))
        and len({d for d in dtypes if d.is_floating_point}) <= 1
    )


def _inputs_sharing_memory(
    op: torch.fx.Node, value: torch.fx.Node
) -> List[torch.fx.Node]:
    """The inputs of op whose memory value, op or a getitem of it, shares in the
    traced values, see _runs_again."""
    inputs = op.all_input_nodes
    val = value.meta.get("val")
    if _runs_again(op):
        # The passes ask about the same op many times, and the result changes only
        # with the traced values of its inputs.
        vals = [a.meta["val"] for a in inputs]
        ran = op.meta.get(_RAN_AGAIN)
        if (
            ran is None
            or len(ran[0]) != len(vals)
            or any(map(operator.is_not, ran[0], vals))
        ):
            args, kwargs = torch.fx.node.map_arg(
                (op.args, op.kwargs), lambda a: a.meta["val"]
            )
            with detect_fake_mode(vals), torch._C._DisableAutocast():
                ran = op.meta[_RAN_AGAIN] = (vals, op.target(*args, **kwargs))
        val = ran[1] if value is op else ran[1][value.args[1]]
    storages = _storages(val)
    return [a for a in inputs if not storages.isdisjoint(_storages(a.meta.get("val")))]


def _aliases_an_input(node: torch.fx.Node) -> bool:
    """Whether the schema of an op says it can return an input or a view of it. It
    runs nothing, see _returns_an_input for ops that do so without saying it."""
    return isinstance(node.target, torch._ops.OpOverload) and any(
        r.alias_info is not None for r in node.target._schema.returns
    )


def _returns_an_input(node: torch.fx.Node) -> bool:
    """Whether an op can return an input or a view of it. Some ops, such as type_as
    with the same dtype, dropout with train=False and broadcast_tensors, do so though
    their schema does not say it, see _runs_again."""
    return _aliases_an_input(node) or (
        isinstance(node.target, torch._ops.OpOverload)
        and bool(_inputs_sharing_memory(node, node))
    )


# The ops known to return an input or a view of it though their schema does not say
# it. Only for these does _may_be_an_input assume so, since ops such as linear and
# batch_norm, whose inputs often differ in dtype in a region, return new memory.
# einsum returns a view only of a single operand, which has one dtype, so it runs
# again.
_MAY_RETURN_AN_INPUT = {
    torch.ops.aten.alpha_dropout,
    torch.ops.aten.atleast_1d,
    torch.ops.aten.atleast_2d,
    torch.ops.aten.atleast_3d,
    torch.ops.aten.broadcast_tensors,
    torch.ops.aten.dropout,
    torch.ops.aten.feature_alpha_dropout,
    torch.ops.aten.feature_dropout,
    torch.ops.aten.meshgrid,
    torch.ops.aten.type_as,
}


def _may_be_an_input(op: torch.fx.Node, writes: List[Tuple[Any, Any]]) -> bool:
    """Whether op can return an input that its traced value does not show: it is an
    op known to return an input, not a cast that copied, it does not run again, and
    the traced values do not show the memory that ops share, as on a program loaded
    with torch.export.load, where an in-place op among writes, pairs of an op and the
    tensor it writes into, has memory of its own. A writer that returns nothing to
    compare, such as _foreach_add_ or a region, counts as such an op."""
    return (
        isinstance(op.target, torch._ops.OpOverload)
        and op.target.overloadpacket in _MAY_RETURN_AN_INPUT
        and not _copies(op)
        and not _runs_again(op)
        and any(
            not _aliases_an_input(n)
            or _storages(n.meta.get("val")).isdisjoint(_storages(w.meta.get("val")))
            for n, w in writes
        )
    )


def _copies(node: torch.fx.Node) -> bool:
    """Whether node is a cast, such as x.float() or x.type_as(y), that changed the
    dtype in its region and so made a copy. Inlining keeps that copy."""
    return (
        isinstance(node.target, torch._ops.OpOverload)
        and (
            node.target.overloadpacket is torch.ops.aten.to
            or node.target == torch.ops.aten.type_as.default
        )
        and node.args[0].meta["val"].dtype != node.meta["val"].dtype
    )


# The index of the body among the arguments of a region and of a torch.no_grad block,
# whose operands follow it: wrap_with_autocast(device_type, dtype, enabled,
# cache_enabled, body, *operands) and wrap_with_set_grad_enabled(enabled, body,
# *operands).
_BODY = {
    torch.ops.higher_order.wrap_with_autocast: 4,
    torch.ops.higher_order.wrap_with_set_grad_enabled: 1,
}


def _body(block: torch.fx.Node) -> torch.fx.GraphModule:
    return getattr(block.graph.owning_module, block.args[_BODY[block.target]].target)


def _operands(block: torch.fx.Node) -> Tuple[Any, ...]:
    return tuple(block.args[_BODY[block.target] + 1 :])


def _region_output(value: torch.fx.Node) -> Optional[torch.fx.Node]:
    """For the getitem of an output of a region or a torch.no_grad block, the value
    its body returns."""
    block = value.args[0] if value.target == operator.getitem else None
    if block is None or block.target not in _BODY:
        return None
    return _body(block).graph.output_node().args[0][value.args[1]]


def _source(value: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The tensor whose memory a value shares directly: the input of a view or an
    in-place op, or for an output of a region or a torch.no_grad block, the value of
    its body or the operand that output shares memory with. None for new memory."""
    output = _region_output(value)
    if output is not None:
        base = _base(output)
        placeholders = list(base.graph.find_nodes(op="placeholder"))
        if base not in placeholders:
            return base
        return _operands(value.args[0])[placeholders.index(base)]
    op = value.args[0] if value.target == operator.getitem else value
    if op.target == torch.ops.aten.set_.source_Tensor:
        # set_ gives its input the memory of the other tensor.
        return op.args[1]
    if _returns_an_input(op) and not _copies(op):
        return (
            tensors_written_by(op) or _inputs_sharing_memory(op, value) or [op.args[0]]
        )[0]
    return None


def _base(value: torch.fx.Node) -> torch.fx.Node:
    """The value whose memory a value shares, through views, in-place ops, nested
    regions and torch.no_grad blocks."""
    # A loop, since a chain of in-place ops can be longer than the recursion limit.
    while (source := _source(value)) is not None:
        value = source
    return value


def _sharing(gm: torch.fx.GraphModule, value: torch.fx.Node) -> Set[torch.fx.Node]:
    """The values of gm that share the memory of value. They follow, through views,
    in-place ops and regions, from the value of gm that value follows from, or from
    the outputs of the region or block whose body gives them that memory."""
    root = value
    while (source := _source(root)) is not None and source.graph is gm.graph:
        root = source
    base = _base(root)
    shared = (
        {root} if base is root else {g for g in root.args[0].users if _base(g) is base}
    )
    todo = list(shared)
    while todo:
        n = todo.pop()
        for u in n.users:
            for c in (u, *(g for g in u.users if g.target == operator.getitem)):
                if c not in shared and _source(c) is n:
                    shared.add(c)
                    todo.append(c)
    return shared


def _written_by(node: torch.fx.Node) -> List[torch.fx.Node]:
    """The tensors node may write into in place. A region or a torch.no_grad block
    writes into the operands its body writes into. Another higher order op may write
    into all of its operands, see tensors_maybe_written_by."""
    if node.target not in _BODY:
        return tensors_maybe_written_by(node)
    body = _body(node)
    placeholders = body.graph.find_nodes(op="placeholder")
    writes = [(n, w) for n in body.graph.nodes for w in _written_by(n)]
    values = {w for _, w in writes}
    # Each op of a chain of in-place ops writes into the one before, which has the
    # same memory, so only the start of a chain is followed back.
    written = {_base(w) for w in values if _source(w) not in values}
    if any(
        _may_be_an_input(op, writes)
        for op in (w.args[0] if w.target == operator.getitem else w for w in written)
    ):
        return list(_operands(node))
    return [a for p, a in zip(placeholders, _operands(node)) if p in written]


def _writers(gm: torch.fx.GraphModule, value: torch.fx.Node) -> Set[torch.fx.Node]:
    """The ops of gm that may write in place into the memory of value, through value
    or any tensor that shares that memory. Such an op takes one of those tensors."""
    shared = _sharing(gm, value)
    return {u for n in shared for u in n.users if not shared.isdisjoint(_written_by(u))}


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
    return any(_may_read_storage(u) for n in _sharing(gm, value) for u in n.users)


def _writes_only_a_new_value(
    value: Any,
    writer: torch.fx.Node,
    operands: Dict[torch.fx.Node, Tuple[Any, torch.fx.Node]],
    include_disabled: bool,
    inlined: Dict[torch.fx.Node, bool],
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
                and (not _returns_an_input(u) or _copies(u))
                and not _may_be_an_input(u, [(writer, value)])
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
            if not _is_inlined(
                value.graph.owning_module, value.args[0], include_disabled, inlined
            ):
                return False
            value, writer = output, output.graph.output_node()
            continue
        written = tensors_written_by(value)
        if len(written) != 1:
            op = value.args[0] if value.target == operator.getitem else value
            return (
                isinstance(op.target, torch._ops.OpOverload)
                and (not _returns_an_input(op) or _copies(op))
                and not _may_be_an_input(op, [(writer, value)])
            )
        value, writer = written[0], value
    return False


def _can_inline(
    gm: torch.fx.GraphModule,
    region: torch.fx.Node,
    include_disabled: bool,
    inlined: Dict[torch.fx.Node, bool],
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
        ) and _writers(gm, user):
            return False
        if base is not output and _reads_storage(gm, user):
            return False
    for node in body.graph.nodes:
        if node.target == torch.ops.aten.view.dtype:
            return False
        if node.target == torch.ops.higher_order.wrap_with_autocast and not (
            _can_inline(body, node, include_disabled, inlined, operands)
        ):
            return False
        if not all(
            _writes_only_a_new_value(w, node, operands, include_disabled, inlined)
            for w in tensors_written_by(node)
        ):
            return False
    return True


def _is_inlined(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    include_disabled: bool,
    inlined: Dict[torch.fx.Node, bool],
) -> bool:
    """Whether node is a region that gets inlined. A region with enabled=False is
    how a model keeps ops in FP32, so it keeps its autocast unless include_disabled.
    inlined keeps the answers for the graph as it is, since each write into the
    output of a region asks about that region again."""
    if node not in inlined:
        inlined[node] = (
            node.target == torch.ops.higher_order.wrap_with_autocast
            and (include_disabled or bool(node.args[2]))
            and _can_inline(gm, node, include_disabled, inlined)
        )
    return inlined[node]


def log_kept_regions(
    gm: torch.fx.GraphModule, low_precision_type: torch.dtype, warn: bool = False
) -> None:
    """Log the regions of the graph that keep a precision other than
    low_precision_type, regions with enabled=False or in another dtype. It logs at
    INFO, since the model's own regions are what "keep" asks for and a model keeps ops
    in FP32 on purpose with enabled=False. With warn, as under "reclassify", which
    asks autocast to choose in regions in another dtype, it counts only the regions
    with enabled=False and those nested in them, which "reclassify_all" can inline,
    and those in another dtype log as a warning. It analyzes nothing, so it can also
    count a region that no value inlines, such as one that writes into a buffer."""
    if not logger.isEnabledFor(logging.WARNING if warn else logging.INFO):
        return
    disabled = other = 0
    # wrap_with_autocast(device_type, dtype, enabled, cache_enabled, body, *operands)
    regions = [
        r
        for r in gm.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.wrap_with_autocast
        )
        if not warn or not r.args[2]
    ]
    while regions:
        region = regions.pop()
        if not region.args[2]:
            disabled += 1
        elif region.args[1] != low_precision_type:
            other += 1
        regions += _body(region).graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.wrap_with_autocast
        )

    def count(n: int) -> str:
        return f"{n} region" + ("" if n == 1 else "s")

    kinds = [f"{count(disabled)} with enabled=False"] if disabled else []
    if other:
        kinds.append(f"{count(other)} in a dtype other than {low_precision_type}")
    if kinds:
        logger.log(
            logging.WARNING if warn and other else logging.INFO,
            "Autocast does not choose the precision of the ops in torch.autocast "
            f"regions of the model that keep their own: {' and '.join(kinds)}. Set "
            f'autocast_model_regions="{"reclassify_all" if disabled else "reclassify"}" '
            "to let autocast choose there too, where it can, and check the accuracy, "
            "since a model can keep ops in FP32 on purpose."
            + (
                " autocast_excluded_nodes can keep the ops of a module in FP32, such "
                "as a rotary embedding."
                if disabled
                else ""
            ),
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
    gm: torch.fx.GraphModule,
    region: torch.fx.Node,
    regions: Set[torch.fx.Node],
    copied: Dict[torch.fx.Node, bool],
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
                    # to() and type_as() return their input when that already has the
                    # dtype, as it now can. Keep the copy, which a write into the
                    # result or a view of it can need. type_as(y) is to(y), which
                    # takes copy.
                    if node.target == torch.ops.aten.type_as.default:
                        env[node].target = torch.ops.aten.to.other
                        env[node].meta[OP_IN_REGION] = node.target
                    names = [a.name for a in env[node].target._schema.arguments]
                    if names.index("copy") < len(node.args):
                        env[node].update_arg(names.index("copy"), True)
                    else:
                        env[node].update_kwarg("copy", True)
                # Ops such as matmul and lerp need their floating point inputs in one
                # dtype. Autocast gave them one when it ran and has a kernel for the op,
                # which casts them, or when they all had one dtype in the region.
                dtypes = {
                    a.meta["val"].dtype
                    for a in node.all_input_nodes
                    if isinstance(a.meta.get("val"), torch.Tensor)
                    and a.meta["val"].is_floating_point()
                }
                copied[env[node]] = len(dtypes) == 1 or (
                    bool(region.args[2])
                    and isinstance(node.target, torch._ops.OpOverload)
                    and node.target.has_kernel_for_dispatch_key(autocast_key)
                )
        # Each user of a region takes one of its outputs with getitem. Ops get the
        # output in the dtype the region gave it, which the rest of the graph expects,
        # and regions inlined in this round too get it as it is. When another op after
        # the region may write into the output before such a region reads it, it
        # writes into the cast, so that region gets the cast too.
        for user in list(region.users):
            output = env[outputs[user.args[1]]]
            val = user.meta.get("val")
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                cast = _cast(gm, output, val.dtype, user)
                cast.meta["val"] = val
                readers = {u for u in user.users if u in regions}
                writes = (
                    [w for w in _writers(gm, user) if w > region] if readers else []
                )
                readers = {u for u in readers if all(u <= w for w in writes)}
                user.replace_all_uses_with(
                    cast, delete_user_cb=lambda u: u not in readers
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
    the given ops that had them in one dtype in their region, to their promoted dtype.
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


def inline_autocast_regions(
    gm: torch.fx.GraphModule,
    include_disabled: bool = False,
    low_precision_type: Optional[torch.dtype] = None,
) -> None:
    """Replace each torch.autocast region of the graph with its body, without the region's autocast.

    Torch-TensorRT autocast then chooses the precision of the body's ops like that
    of any other op. Where the floating point inputs of an op had one dtype in its
    region, because autocast cast them as for matmul or because they were computed
    in one dtype, and they now differ, they are cast to their promoted dtype. The rest of the graph gets
    each output of a region in the dtype the region gave it. A region whose values
    inlining could change keeps its autocast, see _can_inline. A region that keeps
    its autocast gets its inputs in the dtypes it was traced with. Regions with
    enabled=False keep theirs too, unless include_disabled. With low_precision_type,
    it logs the regions with enabled=False, see log_kept_regions.
    """
    copied: Dict[torch.fx.Node, bool] = {}
    inlined: Dict[torch.fx.Node, bool] = {}
    count = 0
    while regions := [
        node
        for node in gm.graph.nodes
        if _is_inlined(gm, node, include_disabled, inlined)
    ]:
        inlining = set(regions)
        for region in regions:
            _inline_region(gm, region, inlining, copied)
        count += len(regions)
        inlined.clear()
    # The results of the ops that ran again belong to this run, see _runs_again.
    for module in gm.modules():
        if isinstance(module, torch.fx.GraphModule):
            for node in module.graph.nodes:
                node.meta.pop(_RAN_AGAIN, None)
    if low_precision_type is not None and not include_disabled:
        # On the graph the loop above decided on. With include_disabled, no region
        # left would be inlined.
        log_kept_regions(gm, low_precision_type, warn=True)
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
