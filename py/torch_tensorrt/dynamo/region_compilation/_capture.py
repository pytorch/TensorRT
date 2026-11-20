# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Consume traced scope markers before decomposition and partitioning."""

import copy
import inspect
import itertools
import logging
import operator
from typing import Any, Iterable, Optional, Sequence

import torch
from torch._prims_common import is_cpu_scalar_tensor
from torch.fx.operator_schemas import normalize_function
from torch.fx.passes.utils.fuser_utils import erase_nodes, fuse_as_graphmodule
from torch_tensorrt.dynamo.lowering.passes.remove_assert_nodes import (
    apply_remove_assert_nodes,
)
from torch_tensorrt.region._constants import (
    BACKEND_KEY,
    NAME_KEY,
    OPERATOR_KEY,
    PLUGIN_KEY,
    ROUTE_KEY,
    SELECTED_KEY,
    SUPPORTED_BACKENDS,
    TOKEN_KEY,
)
from torch_tensorrt.region._errors import RegionCaptureError, RegionError
from torch_tensorrt.region._operators import check_operator, find_operator
from torch_tensorrt.region._plugin import find_kernel

from ._types import (
    HOP_SCHEMA_KEY,
    REGION_ID_META_KEY,
    REGION_META_KEY,
    RegionRecord,
    attach_region_records,
    audit_region_placement,
)
from ._validation import validate_output_aliases

logger = logging.getLogger(__name__)
# The metadata a region scope puts on its members.
_SCOPE_TAGS = (
    TOKEN_KEY,
    NAME_KEY,
    BACKEND_KEY,
    ROUTE_KEY,
    PLUGIN_KEY,
    OPERATOR_KEY,
    SELECTED_KEY,
)
# The guards tracing adds before casts and on sizes; lowering removes them.
_GUARDS = frozenset(
    {
        torch.ops.aten._assert_tensor_metadata.default,
        torch.ops.aten._assert_scalar.default,
    }
)
# Tokens for scopes tagged by position; a traced scope's token is an object id.
_POSITION_TOKENS = itertools.count(1)
# Marks the copies _copy_shared_inputs makes of inputs that share storage.
_ALIAS_COPY_KEY = "torch_tensorrt_region_alias_copy"


def _marker_target_kind(target: Any) -> Optional[str]:
    if isinstance(target, torch._ops.OpOverload):
        if target._schema.name == "torch_tensorrt_region::begin":
            return "begin"
        if target._schema.name == "torch_tensorrt_region::end":
            return "end"
    return None


def _marker_kind(node: torch.fx.Node) -> Optional[str]:
    return _marker_target_kind(node.target) if node.op == "call_function" else None


def has_region_markers(gm: torch.fx.GraphModule) -> bool:
    """Whether a trace recorded region scopes that capture has not consumed."""
    return any(_marker_kind(node) for node in gm.graph.nodes)


def _reject_effect_tokens(gm: torch.fx.GraphModule, signature: Any) -> None:
    """Name the pass that hid the markers before they read as lost markers."""
    # ExportedProgram.run_decompositions() functionalizes ordered effects: each
    # marker becomes with_effects(token, marker, ...), which capture cannot pair.
    wrapped = any(
        node.op == "call_function"
        and isinstance(node.target, torch._ops.HigherOrderOperator)
        and node.target.name() == "with_effects"
        and len(node.args) > 1
        and _marker_target_kind(node.args[1])
        for node in gm.graph.nodes
    )
    tokens = (
        signature is not None
        and has_region_markers(gm)
        and any(
            spec.kind.name == "TOKEN"
            for spec in (*signature.input_specs, *signature.output_specs)
        )
    )
    if wrapped or tokens:
        raise RegionCaptureError(
            "Region markers reached effect-token functionalization: "
            "ExportedProgram.run_decompositions() rewrote the region markers "
            "of this program. Compile the program as torch.export produced "
            "it, before run_decompositions()"
        )


def is_region_hop(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.HigherOrderOperator)
        and node.target.name() == "hints_wrapper"
        and node.kwargs.get("hints", {}).get(HOP_SCHEMA_KEY) == 1
    )


def _has_unresolved_regions(gm: torch.fx.GraphModule) -> bool:
    return any(
        _marker_kind(node) or is_region_hop(node)
        for module in gm.modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
    )


def reject_region_capture(
    ep: torch.export.ExportedProgram, path: str, rules: Any = None
) -> None:
    """Refuse regions, written in the source or selected by rules, on ``path``."""
    if rules or _has_unresolved_regions(ep.graph_module):
        raise RegionError(f"Explicit regions do not support {path}")


def assert_no_unresolved_regions(gm: torch.fx.GraphModule) -> None:
    if _has_unresolved_regions(gm):
        raise RegionError(
            "Region capture markers/HOPs must be resolved before support analysis"
        )


def _reject_higher_order_nesting(gm: torch.fx.GraphModule) -> None:
    """Reject HOP nesting by name before it reads as lost markers.

    A HOP body is a separate graph, so it splits a scope's tags from its markers.
    """
    depth = 0
    for node in gm.graph.nodes:
        kind = _marker_kind(node)
        if kind:
            depth += 1 if kind == "begin" else -1
        elif isinstance(node.target, torch._ops.HigherOrderOperator) and (
            depth
            or any(
                _has_unresolved_regions(gm.get_submodule(str(arg.target)))
                for arg in node.all_input_nodes
                if arg.op == "get_attr"
            )
        ):
            raise RegionCaptureError(
                f"Higher-order operator {node.target.name()!r} "
                f"{'inside' if depth else 'around'} a region is unsupported; "
                "export captures torch.cond, flex_attention, torch.autocast, and "
                "grad-mode changes such as torch.no_grad() as higher-order "
                "operators. Compile under torch.no_grad() so that no_grad blocks "
                "are not captured, use explicit casts instead of torch.autocast, "
                "or keep the operator and the region apart"
            )


def _without_capture_tags(
    meta: dict[str, Any], tags: Sequence[str] = _SCOPE_TAGS
) -> dict[str, Any]:
    result = copy.copy(meta)
    if "custom" in result:
        custom = dict(result["custom"])
        for key in tags:
            custom.pop(key, None)
        if custom:
            result["custom"] = custom
        else:
            result.pop("custom")
    return result


def region_hop_records(gm: torch.fx.GraphModule) -> list[RegionRecord]:
    records = []
    for node in gm.graph.nodes:
        if is_region_hop(node):
            hints = node.kwargs["hints"]
            records.append(
                RegionRecord(
                    id=hints["region_id"],
                    name=hints["region_name"],
                    child_target=str(node.args[0].target),
                    backend=hints.get("backend", "eager"),
                    route=hints.get("route", "torch"),
                    plugin_key=hints.get("plugin_key", ""),
                    operator_key=hints.get("operator_key", ""),
                )
            )
    return records


def _tensor_values(value: Any) -> list[torch.Tensor]:
    values: list[torch.Tensor] = torch.utils._pytree.tree_leaves(value)
    if not values or any(not isinstance(item, torch.Tensor) for item in values):
        raise RegionCaptureError("Region capture requires Tensor region boundaries")
    for tensor in values:
        if any(isinstance(size, torch.SymInt) for size in tensor.shape):
            raise RegionCaptureError("Region capture currently requires static shapes")
    return values


# nondeterministic_seeded tags these ops by name, but they draw no random
# numbers when any listed argument is 0 or False, as in eval-mode dropout.
_SAMPLING_SWITCHES: dict[torch._ops.OpOverloadPacket, tuple[str, ...]] = {
    torch.ops.aten.dropout: ("p", "train"),
    torch.ops.aten.feature_dropout: ("p", "train"),
    torch.ops.aten.alpha_dropout: ("p", "train"),
    torch.ops.aten.feature_alpha_dropout: ("p", "train"),
    torch.ops.aten.native_dropout: ("p", "train"),
    torch.ops.aten.rrelu: ("training",),
    torch.ops.aten.scaled_dot_product_attention: ("dropout_p",),
    torch.ops.aten.lstm: ("dropout", "train"),
    torch.ops.aten.gru: ("dropout", "train"),
    torch.ops.aten.rnn_tanh: ("dropout", "train"),
    torch.ops.aten.rnn_relu: ("dropout", "train"),
}


def _draws_random_numbers(node: torch.fx.Node) -> bool:
    target = node.target
    if not isinstance(target, torch._ops.OpOverload) or (
        torch.Tag.nondeterministic_seeded not in target.tags
    ):
        return False
    switches = _SAMPLING_SWITCHES.get(target.overloadpacket)
    if switches is None:
        # dropout_ and its relatives take the switches of the plain operator.
        plain = target._schema.name.partition("::")[2].rstrip("_")
        switches = _SAMPLING_SWITCHES.get(getattr(torch.ops.aten, plain, None))
    if switches is None:
        return True
    arguments = normalize_function(
        target, node.args, node.kwargs, normalize_to_only_use_kwargs=True
    )
    # Symbolic values and native_dropout's train=None keep sampling enabled.
    return arguments is None or not any(
        arguments.kwargs[name] in (0, False) for name in switches
    )


def _reject_dead_random_region(nodes: Iterable[torch.fx.Node], region_id: str) -> None:
    if any(_draws_random_numbers(node) for node in nodes):
        raise RegionCaptureError(
            f"A random fused_op region must have a live output; {region_id!r} is "
            "dead because its outputs are unused or used only for their shape or dtype"
        )


# operator.iadd and the like, which torch.compile records for ``x += y``.
_IN_PLACE_OPERATORS = frozenset(
    {
        "iadd",
        "iand",
        "iconcat",
        "ifloordiv",
        "ilshift",
        "imatmul",
        "imod",
        "imul",
        "ior",
        "ipow",
        "irshift",
        "isub",
        "itruediv",
        "ixor",
        "setitem",
        "delitem",
    }
)


def _call_argument(node: torch.fx.Node, name: str) -> Any:
    """The value a torch.compile graph node passes for a Python parameter."""
    if name in node.kwargs:
        return node.kwargs[name]
    try:
        bound = inspect.signature(node.target).bind_partial(*node.args, **node.kwargs)
    except (TypeError, ValueError):
        return None
    return bound.arguments.get(name)


def _is_traced_python_call(node: torch.fx.Node) -> bool:
    """Whether the node is a Python call torch.compile recorded, without a schema."""
    return node.op == "call_method" or (
        node.op == "call_function"
        and not isinstance(
            node.target,
            (
                torch._ops.OpOverload,
                torch._ops.OpOverloadPacket,
                torch._ops.HigherOrderOperator,
            ),
        )
    )


def _mutates_in_place(node: torch.fx.Node) -> bool:
    """Whether a torch.compile graph node, which has no operator schema, mutates."""
    if not _is_traced_python_call(node):
        return False
    if node.op == "call_method":
        name = str(node.target)
        if name.endswith("__"):
            # x.__iadd__(y), x.__setitem__(...): the methods behind "x += y".
            return name.strip("_") in _IN_PLACE_OPERATORS
        return name.endswith("_")
    name = getattr(node.target, "__name__", "")
    if getattr(node.target, "__module__", None) in ("_operator", "operator"):
        # operator.and_ and operator.or_ end with an underscore but write nothing.
        return name in _IN_PLACE_OPERATORS
    if name.endswith("_") and not name.endswith("__"):
        return True
    return node.kwargs.get("out") is not None or _call_argument(node, "inplace") is True


def _written_tensor(node: torch.fx.Node) -> Any:
    """The argument an in-place call writes."""
    out = node.kwargs.get("out")
    return out if out is not None else (node.args[0] if node.args else None)


def _is_fresh(node: torch.fx.Node) -> bool:
    """Whether the node made a new tensor that shares storage with none of its inputs."""
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor) or node.op not in (
        "call_function",
        "call_method",
    ):
        return False
    if isinstance(node.target, torch._ops.OpOverload) and (
        node.target._schema.is_mutable
        or any(result.alias_info is not None for result in node.target._schema.returns)
    ):
        return False
    if _mutates_in_place(node) or _returns_its_input(node):
        return False
    return not any(
        isinstance(other := source.meta.get("val"), torch.Tensor)
        and (value is other or torch._C._is_alias_of(value, other))
        for source in node.all_input_nodes
    )


def _write_out_of_place(node: torch.fx.Node) -> bool:
    """Turn an in-place call into the call that returns a new tensor instead."""
    target = node.target
    if isinstance(target, torch._ops.OpOverload):
        name = target._schema.name.partition("::")[2]
        functional = getattr(
            getattr(torch.ops.aten, name[:-1], None), target._overloadname, None
        )
        if (
            target.namespace != "aten"
            or not name.endswith("_")
            or not target._schema.is_mutable
            or not isinstance(functional, torch._ops.OpOverload)
            or functional._schema.is_mutable
        ):
            return False
        node.target = functional
        return True
    if not _mutates_in_place(node) or node.kwargs.get("out") is not None:
        return False
    name = str(target) if node.op == "call_method" else getattr(target, "__name__", "")
    if node.op == "call_method":
        # relu_ becomes relu, and __iadd__ becomes __add__.
        plain = f"__{name[3:]}" if name.endswith("__") else name[:-1]
        if name in ("__setitem__", "__delitem__") or not hasattr(torch.Tensor, plain):
            return False
        node.target = plain
        return True
    if getattr(target, "__module__", None) in ("_operator", "operator"):
        functional = getattr(operator, name[1:], None)
    elif name.endswith("_"):
        functional = next(
            (
                found
                for module in (
                    inspect.getmodule(target),
                    torch,
                    torch.nn.functional,
                )
                if (found := getattr(module, name[:-1], None)) is not None
            ),
            None,
        )
    else:
        # F.relu(x, inplace=True): the flag may be passed by position.
        try:
            bound = inspect.signature(target).bind(*node.args, **node.kwargs)
        except (TypeError, ValueError):
            return False
        bound.arguments["inplace"] = False
        node.args, node.kwargs = bound.args, bound.kwargs
        return True
    if functional is None:
        return False
    node.target = functional
    return True


def _functionalize_local_writes(members: list[torch.fx.Node]) -> None:
    """Make in-place writes to a region's own new tensors out of place.

    ``nn.ReLU(inplace=True)`` after a convolution writes a tensor that only
    the activation reads, so the write changes nothing a reader could see.
    """
    member_set = set(members)
    position: dict[torch.fx.Node, int] = {}
    for node in members:
        written = node.args[0] if node.args else None
        if (
            not isinstance(written, torch.fx.Node)
            or written not in member_set
            or not _is_fresh(written)
        ):
            continue
        others = [user for user in written.users if user is not node]
        if others:
            # A torch.compile graph keeps what reads ``y`` after ``y.relu_()``
            # on the node from before the write. Those reads see the written
            # values. A read before the write may hold a view of the old ones.
            if not _is_traced_python_call(node):
                continue
            position = position or {
                other: index for index, other in enumerate(node.graph.nodes)
            }
            if any(position[user] < position[node] for user in others):
                continue
        if not _write_out_of_place(node):
            continue
        for user in others:
            user.replace_input_with(written, node)
        # The write returned the written tensor itself; the copy is a new one.
        node.meta["val"] = node.meta["val"].clone()


# What to do instead of a grad-mode block, and instead of an autocast block.
_BLOCK_ADVICE = (
    "Enter the block around the call of the compiled model",
    "Use explicit casts such as .float() instead of torch.autocast, or enter "
    "the block around the call of the compiled model",
)


def _context_switch(node: torch.fx.Node) -> Optional[str]:
    """Name the block whose entry or exit torch.compile recorded as this node."""
    if node.op != "call_function":
        return None
    if node.target is torch._C._set_grad_enabled:
        return "torch.no_grad() or torch.enable_grad()"
    name = getattr(node.target, "__name__", "")
    if name in ("_enter_autocast", "_exit_autocast"):
        return "torch.autocast"
    if name in ("_enter_inference_mode", "_exit_inference_mode"):
        return "torch.inference_mode()"
    return None


def _has_effects(node: torch.fx.Node) -> bool:
    if _mutates_in_place(node):
        return True
    if not node.is_impure(impure_random=False):
        return False
    # FX counts Python random functions as effects. Whether a region may draw
    # random numbers is decided on the operators they lower to.
    return not _is_traced_python_call(node) or node.target not in getattr(
        torch._library.utils, "_RANDOM_FUNCTIONS", ()
    )


def _validate_members(
    members: Iterable[torch.fx.Node], region_id: str, *, allow_random: bool
) -> None:
    for node in members:
        target = node.target
        block = _context_switch(node)
        if block is not None:
            raise RegionCaptureError(
                f"A {block} block inside a region is unsupported: {region_id!r}. "
                + _BLOCK_ADVICE[block == "torch.autocast"]
            )
        if _has_effects(node):
            raise RegionCaptureError(
                f"Mutation or effects inside regions are unsupported: {target} "
                f"in {region_id!r}"
            )
        if not allow_random and _draws_random_numbers(node):
            raise RegionCaptureError(
                "Random operations are supported only in fused_op regions: "
                f"{target} in {region_id!r}"
            )


def _same_tensor_layout(value: torch.Tensor, other: torch.Tensor) -> bool:
    return bool(
        value.dtype == other.dtype
        and value.device == other.device
        and value.layout == other.layout
        and value.shape == other.shape
        and (value.layout != torch.strided or value.stride() == other.stride())
    )


def _returns_its_input(node: torch.fx.Node) -> bool:
    """Whether the op returns its first argument itself, as eval-mode dropout does."""
    if not node.args or not isinstance(node.args[0], torch.fx.Node):
        return False
    source = node.args[0].meta.get("val")
    value = node.meta.get("val")
    if not isinstance(source, torch.Tensor) or not isinstance(value, torch.Tensor):
        return False
    if _is_traced_python_call(node):
        # A torch.compile graph records the returned object itself. An in-place
        # call returns its argument too, after writing it.
        return value is source and not _mutates_in_place(node)
    # Views share their input's storage too, and a program loaded from a file
    # has lost which values share storage, so neither can decide this. An op
    # that returns its input keeps its layout; its own kernel tells the rest.
    if (
        not isinstance(node.target, torch._ops.OpOverload)
        or node.target._schema.is_mutable
        or not _same_tensor_layout(value, source)
    ):
        return False
    args, kwargs = torch.fx.map_arg(
        (node.args, node.kwargs), lambda arg: arg.meta.get("val")
    )
    try:
        return bool(node.target.decompose(*args, **kwargs) is args[0])
    except Exception:
        return False


def _bypass_members_returning_inputs(
    signature: Any,
    members: list[torch.fx.Node],
    fused: Optional[str],
) -> set[torch.fx.Node]:
    """Make reads after a region bypass members that return their input itself.

    ``fused`` names a fused_op or fused_kernel scope written in the model's
    source, whose operator or kernel may produce such a member as an output,
    so a bypass to an input raises. A region selected by a rule is a whole
    module call: a tensor it passes through is the caller's input, not an
    output the replacement has to produce.
    """
    # Export binds later reads of a tensor to the last op that returned it, so
    # eval-mode dropout or a no-op cast would make the region output its input.
    # Reverse order resolves chains such as x.float().float().
    member_set = set(members)
    bypassed = set()
    for node in reversed(members):
        if not _returns_its_input(node):
            continue
        bypassed.add(node)
        source = node.args[0]
        if (
            fused
            and source not in member_set
            and any(user not in member_set for user in node.users)
        ):
            # A later read may mean the input or one of the replacement's outputs.
            raise RegionCaptureError(
                f"{fused} would output its input {source.name!r} unchanged, "
                f"because {node.target} returns that tensor itself; clone "
                f"{source.name!r} inside the region before that operation"
            )
        node.replace_all_uses_with(
            source, delete_user_cb=lambda user: user not in member_set
        )
        if signature is not None:
            signature.replace_all_uses(node.name, source.name)
    return bypassed


def _functionalize_tensor_constructor_detaches(members: list[torch.fx.Node]) -> bool:
    changed = False
    member_set = set(members)
    for node in members:
        if (
            node.op != "call_function"
            or node.target != torch.ops.aten.detach_.default
            or len(node.args) != 1
            or node.kwargs
        ):
            continue
        source = node.args[0]
        if (
            isinstance(source, torch.fx.Node)
            and source in member_set
            and source.op == "call_function"
            and source.target == torch.ops.aten.lift_fresh_copy.default
            and set(source.users) == {node}
        ):
            # A tensor literal's fresh, single-use copy can detach out of place.
            node.target = torch.ops.aten.detach.default
            changed = True
    return changed


def _reject_complex_regions(gm: torch.fx.GraphModule) -> None:
    if not region_hop_records(gm):
        return
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if any(
                isinstance(value, torch.Tensor) and value.is_complex()
                for value in torch.utils._pytree.tree_leaves(node.meta.get("val"))
            ):
                # Whole-graph complex lowering cannot preserve region boundaries.
                raise RegionCaptureError(
                    "Region capture currently does not support complex tensors "
                    "in a model containing nonempty regions; complex lowering "
                    "cannot preserve region boundaries"
                )


def _boundary(begin: torch.fx.Node) -> dict[str, Any]:
    # Export drops trailing arguments that equal their schema defaults.
    arguments = normalize_function(
        begin.target, begin.args, begin.kwargs, normalize_to_only_use_kwargs=True
    )
    return dict(arguments.kwargs)


def _describe(nodes: Sequence[torch.fx.Node]) -> str:
    # torch.compile records a node's value as "example_value".
    return ", ".join(
        f"{node.name}"
        f"{list(getattr(node.meta.get('val', node.meta.get('example_value')), 'shape', ()))}"
        for node in nodes
    )


def _bind_inputs(
    live_ins: tuple[torch.fx.Node, ...],
    listed: Optional[Sequence[torch.fx.Node]],
    region: str,
    *,
    allow_repeats: bool = False,
) -> tuple[torch.fx.Node, ...]:
    """Order a replacement's arguments as its ``inputs=`` lists them.

    A rule may list one tensor twice, for a module call that passes it twice.
    """
    # Traced first use is not source order: ``up * silu(gate)`` reads gate first.
    if listed is None:
        if len(live_ins) <= 1:
            return live_ins
        raise RegionCaptureError(
            f"{region} reads {len(live_ins)} tensors ({_describe(live_ins)}); "
            "pass inputs=(...) to list them in argument order"
        )
    bound = tuple(listed)
    repeats = len(set(bound)) != len(bound)
    if (repeats and not allow_repeats) or set(bound) != set(live_ins):
        raise RegionCaptureError(
            f"{region} inputs= must list each tensor the region reads exactly "
            f"once; it reads ({_describe(live_ins)}) but inputs= lists "
            f"({_describe(bound)})"
        )
    return bound


def _copy_shared_inputs(
    gm: torch.fx.GraphModule, inputs: Sequence[torch.fx.Node]
) -> tuple[torch.fx.Node, ...]:
    """Copy inputs as needed so that no two region operands share storage.

    hints_wrapper rejects operands that share storage, such as chunk results or
    cuDNN-flattened RNN weights. Regions only read their inputs, so a copy
    leaves the results unchanged. Bodies captured before decomposition do not
    need the copies (_resolve_alias_copies).
    """
    operands: list[torch.fx.Node] = []
    for node in inputs:
        value = node.meta["val"]
        if any(torch._C._is_alias_of(value, other.meta["val"]) for other in operands):
            node = gm.graph.call_function(torch.ops.aten.clone.default, (node,))
            node.meta["val"] = value.clone()
            node.meta["custom"] = {_ALIAS_COPY_KEY: True}
        operands.append(node)
    return tuple(operands)


def _state_sharing_storage(ep: torch.export.ExportedProgram) -> list[list[str]]:
    """Placeholders of parameters, buffers and constants that share storage."""
    signature = ep.graph_signature
    targets = {
        **signature.inputs_to_parameters,
        **signature.inputs_to_buffers,
        **signature.inputs_to_lifted_tensor_constants,
    }
    groups: dict[Any, list[str]] = {}
    for placeholder, target in targets.items():
        tensor = ep.state_dict.get(target, ep.constants.get(target))
        if isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided:
            groups.setdefault(tensor.untyped_storage()._cdata, []).append(placeholder)
    return [group for group in groups.values() if len(group) > 1]


def _lost_shared_storage(ep: torch.export.ExportedProgram) -> bool:
    """Whether the recorded values forgot which tensors share storage.

    torch.export.load rebuilds every value on its own, so a view no longer
    shares its input's storage and capture cannot see what a region aliases.
    """
    nodes = {node.name: node for node in ep.graph.nodes}
    for group in _state_sharing_storage(ep):
        first, second = (nodes[name].meta.get("val") for name in group[:2])
        if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
            return not torch._C._is_alias_of(first, second)
    # Operators whose schema says the result may share the input's storage. If
    # the program has some and none of their results does, the record is lost.
    candidates = False
    for node in ep.graph.nodes:
        if (
            node.op != "call_function"
            or not isinstance(node.target, torch._ops.OpOverload)
            or not node.args
            or not isinstance(node.args[0], torch.fx.Node)
        ):
            continue
        returns = node.target._schema.returns
        if (
            not returns
            or returns[0].alias_info is None
            or returns[0].alias_info.is_write
        ):
            continue
        values = torch.utils._pytree.tree_leaves(node.meta.get("val"))
        source = node.args[0].meta.get("val")
        if (
            values
            and isinstance(values[0], torch.Tensor)
            and isinstance(source, torch.Tensor)
        ):
            if torch._C._is_alias_of(values[0], source):
                return False
            candidates = True
    return candidates


def _recompute_values(ep: torch.export.ExportedProgram) -> None:
    """Record the program's values again so that shared storage shows.

    Parameters that share storage, such as an RNN's flattened weights, get
    values that share it; running the graph on them does the same for views.
    """
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp

    gm = ep.graph_module
    try:
        fake_mode = torch._export.utils._detect_fake_mode_from_gm(gm)
        nodes = {node.name: node for node in gm.graph.nodes}
        targets = {
            **ep.graph_signature.inputs_to_parameters,
            **ep.graph_signature.inputs_to_buffers,
            **ep.graph_signature.inputs_to_lifted_tensor_constants,
        }
        for group in _state_sharing_storage(ep):
            for name in group:
                target = targets[name]
                # One converter maps tensors of one storage to values of one.
                nodes[name].meta["val"] = fake_mode.from_tensor(
                    ep.state_dict.get(target, ep.constants.get(target)),
                    static_shapes=True,
                )
        FakeTensorProp(gm, fake_mode).propagate_dont_convert_inputs(
            *(
                node.meta.get("val")
                for node in gm.graph.nodes
                if node.op == "placeholder"
            )
        )
    except Exception:
        logger.debug("Could not recompute the program's values", exc_info=True)


def _copy_aliased_outputs(
    members: list[torch.fx.Node], signature: Any, *, replaced: bool
) -> tuple[list[torch.fx.Node], dict[torch.fx.Node, torch.fx.Node]]:
    """Give the results of a rule-selected region storage of their own.

    A module may return a view of its input, or several views of one tensor,
    which a region cannot. A scope in the model's source clones them; a rule
    selects a module that cannot be edited, so capture copies them. When an
    operator or kernel replaces the module (``replaced``), results that are
    not laid out contiguously are copied too: a replacement returns ordinary
    tensors. Returns the members with the copies, and each result's copy.
    """
    member_set = set(members)
    copies: dict[torch.fx.Node, torch.fx.Node] = {}
    kept = [
        value
        for node in members
        for source in node.all_input_nodes
        if source not in member_set
        and isinstance(value := source.meta.get("val"), torch.Tensor)
    ]
    lowered = any(isinstance(node.target, torch._ops.OpOverload) for node in members)
    layout = {"memory_format": torch.contiguous_format} if replaced else {}
    result = []
    for node in members:
        result.append(node)
        value = node.meta.get("val")
        readers = [user for user in node.users if user not in member_set]
        if not readers or not isinstance(value, torch.Tensor):
            continue
        shares = any(
            value is other or torch._C._is_alias_of(value, other) for other in kept
        )
        if shares or (
            replaced and value.layout == torch.strided and not value.is_contiguous()
        ):
            with node.graph.inserting_after(node):
                copied = node.graph.call_function(
                    torch.ops.aten.clone.default if lowered else torch.clone,
                    (node,),
                    dict(layout),
                )
            copied.meta = copy.copy(node.meta)
            copied.meta["val"] = value = value.clone(**layout)
            for reader in readers:
                reader.replace_input_with(node, copied)
            if signature is not None:
                signature.replace_all_uses(node.name, copied.name)
            result.append(copied)
            member_set.add(copied)
            copies[node] = copied
        kept.append(value)
    return result, copies


def _order_rule_outputs(
    child: torch.fx.GraphModule,
    members: list[torch.fx.Node],
    outputs: tuple[torch.fx.Node, ...],
    selected: dict[str, Any],
    bypassed: set[torch.fx.Node],
    copied: dict[torch.fx.Node, torch.fx.Node],
    fused: str,
) -> tuple[torch.fx.Node, ...]:
    """Order a rule-selected region's results as its module call returns them.

    A replacement returns what the module returns. Extraction lists the
    results in the order the module computes them, and only those read later.
    """
    returned = selected.get("outputs")
    if returned is None:
        if len(outputs) > 1:
            raise RegionCaptureError(
                f"{fused}: the selected module returns {len(outputs)} tensors "
                f"({_describe(outputs)}), and this program does not record "
                f"the order it returns them. {selected.get('hint', '')}".strip()
            )
        return outputs
    by_name = {node.name: node for node in members}
    # A tensor the module hands through unchanged is the caller's input.
    ordered = tuple(
        dict.fromkeys(
            copied.get(by_name[name], by_name[name])
            for name in returned
            if name in by_name and by_name[name] not in bypassed
        )
    )
    escaped = [node for node in outputs if node not in ordered]
    if escaped:
        raise RegionCaptureError(
            f"{fused}: the program reads {_describe(escaped)}, which the "
            "selected module computes but does not return"
        )
    if ordered != outputs:
        copies = {node.name: node for node in child.graph.nodes}
        output = next(node for node in child.graph.nodes if node.op == "output")
        try:
            output.args = (tuple(copies[node.name] for node in ordered),)
        except KeyError as error:
            raise RegionCaptureError(
                f"{fused}: cannot find the module's result {error} in its body"
            ) from error
        child.graph.lint()
        child.recompile()
    return ordered


def _reject_scopes_in_blocks(gm: torch.fx.GraphModule) -> None:
    """Reject a scope inside an autocast or grad-mode block torch.compile recorded.

    Lowering applies such a block to the operations around the region, but
    the region's body runs later, outside the block.
    """
    blocks: list[str] = []
    grad_mode_changed = False
    for node in gm.graph.nodes:
        if _marker_kind(node) == "begin" and (blocks or grad_mode_changed):
            around = blocks[-1] if blocks else "torch.no_grad() or torch.enable_grad()"
            raise RegionCaptureError(
                f"A region inside a {around} block is unsupported under "
                "torch.compile. " + _BLOCK_ADVICE[around == "torch.autocast"]
            )
        block = _context_switch(node)
        if block is None:
            continue
        if node.target is torch._C._set_grad_enabled:
            # Recorded only when the block changes the grad mode.
            grad_mode_changed = not grad_mode_changed
        elif getattr(node.target, "__name__", "").startswith("_enter"):
            blocks.append(block)
        elif blocks:
            blocks.pop()


def _state_writes(gm: torch.fx.GraphModule) -> list[str]:
    """Describe the writes of a torch.compile graph to its inputs and buffers."""
    inputs = [
        (node.name, value)
        for node in gm.graph.nodes
        if node.op == "placeholder"
        and isinstance(value := node.meta.get("val"), torch.Tensor)
    ]
    writes = []
    for node in gm.graph.nodes:
        if _mutates_in_place(node):
            written = _written_tensor(node)
            value = (
                written.meta.get("val") if isinstance(written, torch.fx.Node) else None
            )
            if not isinstance(value, torch.Tensor):
                continue
            name = next(
                (
                    name
                    for name, other in inputs
                    if value is other or torch._C._is_alias_of(value, other)
                ),
                None,
            )
            if name is not None:
                writes.append(f"mutation of input or buffer {name!r}")
        elif (
            node.op == "call_function"
            and getattr(node.target, "__name__", "") == "batch_norm"
            and _call_argument(node, "training") is True
            and _call_argument(node, "running_mean") is not None
        ):
            writes.append(f"the running statistics of batch norm {node.name!r}")
    return writes


def _tag_scopes_by_position(gm: torch.fx.GraphModule) -> None:
    """Tag the members of scopes whose trace recorded only the markers.

    Non-strict export runs the model's Python, so it records the markers and
    the operations between them in order, but not the scope's tags.
    """
    active: Optional[torch.fx.Node] = None
    members: list[torch.fx.Node] = []
    for node in gm.graph.nodes:
        kind = _marker_kind(node)
        if kind == "begin":
            active, members = node, []
        elif kind == "end":
            if (
                active is not None
                and node.args == (active,)
                and TOKEN_KEY not in node.meta.get("custom", {})
            ):
                boundary = _boundary(active)
                tags = {
                    TOKEN_KEY: -next(_POSITION_TOKENS),
                    NAME_KEY: boundary["name"],
                    BACKEND_KEY: boundary["backend"],
                    ROUTE_KEY: boundary["route"],
                    PLUGIN_KEY: boundary["plugin"],
                    OPERATOR_KEY: boundary["operator"],
                }
                for member in (*members, node):
                    member.meta["custom"] = {**member.meta.get("custom", {}), **tags}
            active = None
        elif active is not None and node.op in (
            "call_function",
            "call_method",
            "call_module",
        ):
            members.append(node)


def normalize_region_scopes(
    ep: torch.export.ExportedProgram,
) -> torch.export.ExportedProgram:
    """Validate exact membership and replace scopes with durable HOP children."""
    gm = ep.graph_module
    if has_region_markers(gm) and _lost_shared_storage(ep):
        _recompute_values(ep)
    if _normalize_scopes(gm, ep.graph_signature, dynamic=bool(ep.range_constraints)):
        ep.validate()
    return ep


def normalize_traced_scopes(gm: torch.fx.GraphModule) -> bool:
    """Normalize the scopes of a graph torch.compile traced; report any region."""
    dynamic = False
    for node in gm.graph.nodes:
        # torch.compile records each node's value under another name.
        if "val" not in node.meta and "example_value" in node.meta:
            node.meta["val"] = node.meta["example_value"]
        if node.op == "placeholder":
            value = node.meta.get("val")
            dynamic = dynamic or isinstance(value, torch.SymInt)
            if isinstance(value, torch.Tensor):
                dynamic = dynamic or any(
                    isinstance(size, torch.SymInt) for size in value.shape
                )
    _reject_scopes_in_blocks(gm)
    writes = _state_writes(gm)
    _normalize_scopes(gm, None, dynamic=dynamic)
    live = bool(region_hop_records(gm))
    if live and writes:
        raise RegionCaptureError(
            "Explicit regions require a functional model: "
            + ", ".join(writes)
            + " is unsupported anywhere in a model containing live regions, "
            "including outside the annotated scope."
        )
    return live


def _normalize_scopes(
    gm: torch.fx.GraphModule, signature: Any, *, dynamic: bool
) -> bool:
    """Replace the scopes of ``gm`` with HOP children; report whether it had any.

    ``signature`` is the exported program's graph signature, kept in step with
    renamed outputs, or None for a graph that has none.
    """
    _reject_effect_tokens(gm, signature)
    _reject_higher_order_nesting(gm)
    for child in gm.modules():
        if not isinstance(child, torch.fx.GraphModule):
            continue
        has_markers = any(_marker_kind(node) for node in child.graph.nodes)
        # Placeholders and export's cast guards copy their input's tags, as in a
        # torch.autocast block that reads a region output; neither is a member.
        if not has_markers and any(
            TOKEN_KEY in node.meta.get("custom", {})
            for node in child.graph.nodes
            if node.op != "placeholder"
            and node.target != torch.ops.aten._assert_tensor_metadata.default
        ):
            raise RegionCaptureError(
                "Region membership survived without its boundary markers"
            )
    if not any(_marker_kind(node) for node in gm.graph.nodes):
        _reject_complex_regions(gm)
        return False
    try:
        from torch._higher_order_ops.hints_wrap import hints_wrapper
    except ImportError as error:
        raise RegionCaptureError(
            "Region capture requires PyTorch's experimental hints_wrapper HOP"
        ) from error

    # Drop export's asserts now rather than in pre-lowering: the guard on each
    # aten.to copies the region tags of the cast's input, not the cast's, so the
    # membership checks below would misplace or reject it.
    apply_remove_assert_nodes(gm)
    _tag_scopes_by_position(gm)

    scopes = []
    active = None
    members: list[torch.fx.Node] = []
    tokens: set[int] = set()
    for node in list(gm.graph.nodes):
        kind = _marker_kind(node)
        custom = node.meta.get("custom", {})
        if kind == "begin":
            if active is not None:
                raise RegionCaptureError("Nested region scopes are unsupported")
            active = node
            members = []
        elif kind == "end":
            if active is None or node.args != (active,):
                raise RegionCaptureError("Unmatched or crossed region markers")
            token = custom.get(TOKEN_KEY)
            boundary = _boundary(active)
            name = boundary["name"]
            backend = boundary["backend"]
            if backend not in SUPPORTED_BACKENDS:
                raise RegionCaptureError(f"Unsupported region backend {backend!r}")
            route = boundary["route"]
            plugin = boundary["plugin"]
            operator_key = boundary["operator"]
            if route not in ("torch", "plugin") or (route == "plugin" and not plugin):
                raise RegionCaptureError("Invalid region plugin boundary")
            if operator_key and (route != "torch" or backend != "eager" or plugin):
                raise RegionCaptureError("Invalid fused_op region boundary")
            if not isinstance(token, int) or token in tokens:
                raise RegionCaptureError("Missing or reused region capture token")
            expected_tags = (
                (TOKEN_KEY, token, None),
                (NAME_KEY, name, None),
                (BACKEND_KEY, backend, "eager"),
                (ROUTE_KEY, route, "torch"),
                (PLUGIN_KEY, plugin, ""),
                (OPERATOR_KEY, operator_key, ""),
            )
            for member in (node, *members):
                tag = member.meta.get("custom", {})
                if any(
                    tag.get(key, default) != expected
                    for key, expected, default in expected_tags
                ):
                    raise RegionCaptureError(
                        f"Region membership disagrees with markers at {member.name!r}"
                    )
            if set(active.users) != {node}:
                raise RegionCaptureError(
                    "Region sentinel escaped into model computation"
                )
            tokens.add(token)
            scopes.append(
                (
                    active,
                    node,
                    members,
                    name,
                    backend,
                    route,
                    plugin,
                    operator_key,
                    custom.get(SELECTED_KEY),
                )
            )
            active = None
        elif node.op in ("call_function", "call_method", "call_module"):
            if active is not None:
                members.append(node)
            elif TOKEN_KEY in custom:
                raise RegionCaptureError(f"Operation {node.name!r} escaped its region")
    if active is not None:
        raise RegionCaptureError("Unclosed region scope")

    for index, (
        begin,
        end,
        members,
        name,
        backend,
        route,
        plugin,
        operator_key,
        selected,
    ) in enumerate(scopes):
        region_id = f"{name or 'region'}#{index}"
        fused = None
        if route == "plugin" or operator_key:
            api = "fused_kernel" if route == "plugin" else "fused_op"
            fused = f"{api} region {region_id!r}"
        # A rule selected the region: its module cannot be edited.
        rule = selected is not None
        functionalized_constructor = _functionalize_tensor_constructor_detaches(members)
        _functionalize_local_writes(members)
        _validate_members(members, region_id, allow_random=bool(operator_key))
        member_set = set(members)
        was_read = any(
            user not in member_set for node in members for user in node.users
        )
        bypassed = _bypass_members_returning_inputs(
            signature, members, None if rule else fused
        )
        if not any(user not in member_set for node in members for user in node.users):
            if rule and fused is not None and was_read:
                # The replacement would never run, and readers of the module's
                # result cannot be told from readers of its input.
                raise RegionCaptureError(
                    f"{fused}: the selected module returns its input "
                    "unchanged, so its operator or kernel would never run. "
                    "Select a module that computes something, or remove the rule"
                )
            _reject_dead_random_region(members, region_id)
            erase_nodes(gm, members)
            gm.graph.erase_node(end)
            gm.graph.erase_node(begin)
            logger.info("Region %s is an empty/dead pure scope", region_id)
            continue
        if dynamic:
            raise RegionCaptureError("Region capture currently requires static shapes")
        copied: dict[torch.fx.Node, torch.fx.Node] = {}
        if rule:
            members, copied = _copy_aliased_outputs(
                members, signature, replaced=fused is not None
            )
        child_name = f"_ttrt_region_body_{index}"
        while hasattr(gm, child_name):
            child_name += "_"
        try:
            child, inputs, outputs = fuse_as_graphmodule(
                gm, members, child_name, always_return_tuple=True
            )
        except (AssertionError, RuntimeError) as error:
            raise RegionCaptureError(
                f"Cannot extract region {region_id!r}: {error}"
            ) from error
        if fused is not None:
            # Read the marker now: an earlier region may have replaced a listed tensor.
            ordered_inputs = _bind_inputs(
                inputs, _boundary(begin)["inputs"], fused, allow_repeats=rule
            )
            if inputs != ordered_inputs:
                placeholders = dict(
                    zip(inputs, (n for n in child.graph.nodes if n.op == "placeholder"))
                )
                first_op = next(n for n in child.graph.nodes if n.op != "placeholder")
                placed = set()
                for arg in ordered_inputs:
                    placeholder = placeholders[arg]
                    if arg in placed:
                        # The call passes this tensor again: one more argument.
                        with child.graph.inserting_before(first_op):
                            again = child.graph.placeholder(f"{placeholder.name}_again")
                        again.meta = copy.copy(placeholder.meta)
                    else:
                        placed.add(arg)
                        first_op.prepend(placeholder)
                inputs = ordered_inputs
                child.graph.lint()
                child.recompile()
            if selected is not None:
                outputs = _order_rule_outputs(
                    child, members, outputs, selected, bypassed, copied, fused
                )
        output_metadata = tuple(node.meta.get("val") for node in outputs)
        if functionalized_constructor:
            from torch.fx.passes.fake_tensor_prop import FakeTensorProp

            # Refresh stale literal aliases locally; decomposition updates the parent.
            fake_mode = torch._export.utils._detect_fake_mode_from_gm(child)
            output_metadata = FakeTensorProp(child, fake_mode).propagate(
                *(node.meta["val"] for node in inputs)
            )
        input_values = []
        for node in inputs:
            if not isinstance(node.meta.get("val"), torch.Tensor):
                raise RegionCaptureError("Region capture requires Tensor live-ins")
            input_values.extend(_tensor_values(node.meta["val"]))
        output_values = []
        for value in output_metadata:
            output_values.extend(_tensor_values(value))
        # As in PyTorch, 0-dim CPU tensors such as torch.tensor(2.0) join any device.
        devices = {
            value.device
            for value in input_values + output_values
            if not is_cpu_scalar_tensor(value)
        }
        if len(devices) > 1:
            raise RegionCaptureError(
                f"Region {region_id!r} cannot span multiple devices: "
                + ", ".join(sorted(map(str, devices)))
            )
        validate_output_aliases(input_values, output_values, region_id)
        # Capture tokens are ephemeral and must not leak to later exports.
        for node in child.graph.nodes:
            node.meta = _without_capture_tags(node.meta)
        gm.add_submodule(child_name, child)
        with gm.graph.inserting_before(begin):
            body = gm.graph.get_attr(child_name)
        # Insert after all live-ins are available, before any external consumer.
        with gm.graph.inserting_before(end):
            operands = _copy_shared_inputs(gm, inputs)
            hop = gm.graph.call_function(
                hints_wrapper,
                args=(body, operands, {}),
                kwargs={
                    "hints": {
                        HOP_SCHEMA_KEY: 1,
                        "region_id": region_id,
                        "region_name": name,
                        "route": route,
                        "plugin_key": plugin,
                        "operator_key": operator_key,
                        "backend": backend,
                    }
                },
            )
            hop.meta = _without_capture_tags(outputs[0].meta)
            hop.meta["val"] = tuple(node.meta["val"] for node in outputs)
            for output_index, original in enumerate(outputs):
                replacement = gm.graph.call_function(
                    operator.getitem, (hop, output_index)
                )
                replacement.meta = _without_capture_tags(original.meta)
                original.replace_all_uses_with(replacement)
                if signature is not None:
                    signature.replace_all_uses(original.name, replacement.name)
        erase_nodes(gm, members)
        gm.graph.erase_node(end)
        gm.graph.erase_node(begin)
    _reject_complex_regions(gm)
    gm.graph.lint()
    gm.recompile()
    # Lifted constant placeholders can also inherit the scope's transient tags.
    for node in gm.graph.nodes:
        node.meta = _without_capture_tags(node.meta)
    return True


def validate_decomposed_bodies(gm: torch.fx.GraphModule) -> None:
    """Check region bodies again once torch.compile has lowered them to ATen.

    A torch.compile graph calls Python functions, which do not say whether
    they draw random numbers; the lowered operators do.
    """
    for node in gm.graph.nodes:
        if is_region_hop(node):
            hints = node.kwargs["hints"]
            body = gm.get_submodule(str(node.args[0].target))
            _validate_members(
                [
                    n
                    for n in body.graph.nodes
                    if n.op == "call_function" and n.target not in _GUARDS
                ],
                hints["region_id"],
                allow_random=bool(hints.get("operator_key")),
            )


def check_decomposed_regions(
    captured: torch.fx.GraphModule, decomposed: torch.fx.GraphModule
) -> None:
    """Let decomposition drop dead regions, but not change or reorder live ones."""
    regions = region_hop_records(decomposed)
    live = {record.id for record in regions}
    kept = []
    for record in region_hop_records(captured):
        if record.id in live:
            kept.append(record.capture_signature)
            continue
        # Decomposition removes regions left without users, for example when
        # full_like(y) becomes full(y.shape) or a consumer is unused.
        body = captured.get_submodule(record.child_target)
        _reject_dead_random_region(body.graph.nodes, record.id)
        logger.info("Region %s is dead after decomposition", record.id)
    if kept != [record.capture_signature for record in regions]:
        raise RegionError("Decomposition changed a region boundary or execution policy")


def _custom_op_body(
    child: torch.fx.GraphModule, op: torch._ops.OpOverload, region_id: str
) -> torch.fx.GraphModule:
    """Check the fake contract and build a Torch leaf containing one custom op."""
    inputs = [node for node in child.graph.nodes if node.op == "placeholder"]
    output = next(node for node in child.graph.nodes if node.op == "output")
    expected = tuple(node.meta.get("val") for node in output.args[0])
    schema = op._schema
    if len(schema.arguments) != len(inputs):
        raise RegionCaptureError(
            f"fused_op region {region_id!r} input count does not match {schema}"
        )
    if any(not isinstance(value, torch.Tensor) for value in expected):
        raise RegionCaptureError(
            f"fused_op region {region_id!r} requires flat Tensor reference outputs"
        )
    input_values = tuple(node.meta["val"] for node in inputs)
    fake_mode = torch._guards.detect_fake_mode((*input_values, *expected))
    if fake_mode is None:
        raise RegionCaptureError(
            f"fused_op region {region_id!r} is missing fake tensor metadata"
        )
    try:
        with fake_mode:
            result = op(*input_values)
    except Exception as error:
        raise RegionCaptureError(
            f"fused_op region {region_id!r} requires a valid fake implementation: {error}"
        ) from error
    actual = (result,) if isinstance(result, torch.Tensor) else result
    if not isinstance(actual, tuple) or len(actual) != len(expected):
        returned = len(actual) if isinstance(actual, tuple) else type(actual).__name__
        raise RegionCaptureError(
            f"fused_op region {region_id!r} output count or flat Tensor structure "
            f"does not match its reference: the operator returns {returned} and "
            f"the region produces {len(expected)} tensor(s)"
        )
    for index, (value, reference) in enumerate(zip(actual, expected)):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != reference.shape
            or value.dtype != reference.dtype
            or value.device != reference.device
            or value.layout != reference.layout
            or value.stride() != reference.stride()
        ):
            differs = (
                [
                    f"{name} {got} instead of {wanted}"
                    for name, got, wanted in (
                        ("shape", tuple(value.shape), tuple(reference.shape)),
                        ("dtype", value.dtype, reference.dtype),
                        ("device", value.device, reference.device),
                        ("layout", value.layout, reference.layout),
                        ("strides", value.stride(), reference.stride()),
                    )
                    if got != wanted
                ]
                if isinstance(value, torch.Tensor)
                else ["not a Tensor"]
            )
            raise RegionCaptureError(
                f"fused_op region {region_id!r} output {index} metadata does not "
                f"match its reference: the operator's fake result has "
                + ", ".join(differs)
            )
    validate_output_aliases(input_values, actual, region_id)
    graph = torch.fx.Graph()
    call = graph.call_function(op, tuple(graph.node_copy(node) for node in inputs))
    call.meta["val"] = result
    if isinstance(result, torch.Tensor):
        values = (call,)
    else:
        values = tuple(
            graph.call_function(operator.getitem, (call, index))
            for index in range(len(actual))
        )
        for node, value in zip(values, actual):
            node.meta["val"] = value
    graph.output(values)
    return torch.fx.GraphModule({}, graph)


def _resolve_alias_copies(hop: torch.fx.Node, *, drop: bool) -> None:
    """Unmark the copies _copy_shared_inputs made for a region, or drop them."""
    for operand in hop.args[1]:
        if not operand.meta.get("custom", {}).get(_ALIAS_COPY_KEY):
            continue
        operand.meta = _without_capture_tags(operand.meta, (_ALIAS_COPY_KEY,))
        if drop:
            hop.replace_input_with(operand, operand.args[0])
            if not operand.users:
                hop.graph.erase_node(operand)


def materialize_regions(
    gm: torch.fx.GraphModule, captured: torch.fx.GraphModule
) -> torch.fx.GraphModule:
    """Replace captured HOPs with PyTorch leaves or custom-kernel operators.

    ``execute_in_torch`` regions run their bodies from ``captured``, the graph
    before decomposition.
    """
    region_nodes = [node for node in gm.graph.nodes if is_region_hop(node)]
    if not region_nodes:
        return gm
    records = region_hop_records(gm)
    bodies = {
        record.id: captured.get_submodule(record.child_target)
        for record in region_hop_records(captured)
    }
    for index, (node, record) in enumerate(zip(region_nodes, records)):
        body_node = node.args[0]
        child = gm.get_submodule(record.child_target)
        if not isinstance(child, torch.fx.GraphModule):
            raise RegionError("Region HOP does not reference a GraphModule")
        child_name = f"_ttrt_torch_region_{index}"
        while hasattr(gm, child_name):
            child_name += "_"
        route = record.route
        plugin_op = None
        # A body captured before decomposition reads the inputs, not their copies.
        _resolve_alias_copies(node, drop=route == "torch" and not record.operator_key)
        if route == "plugin":
            from ._plugins import register_region_plugin

            kernel = find_kernel(record.plugin_key)
            if kernel is None:
                # The kernel is a Python object; the traced program holds a key.
                raise RegionError(
                    f"fused_kernel region {record.id!r} has no kernel in this "
                    "process: saving an exported program does not save the "
                    "kernel. Compile in the process that exported the program, "
                    "or select the module with regions={...: FusedKernel(...)} "
                    "when compiling"
                )
            try:
                plugin_op = register_region_plugin(child, kernel, region_id=record.id)
            except ValueError as error:
                # Kernel registration names the generated operator, not the region.
                error.add_note(f"Raised for fused_kernel region {record.id!r}")
                raise
        else:
            if record.operator_key:
                op = find_operator(record.operator_key)
                if op is None:
                    raise RegionError(
                        f"fused_op region {record.id!r} uses the operator "
                        f"{record.operator_key!r}, which is not registered in "
                        "this process; import the code that defines it before "
                        "compiling"
                    )
                # A loaded program names the operator: check it again.
                child = _custom_op_body(child, check_operator(op), record.id)
            else:
                # Torch-TensorRT's decompositions target TensorRT, not PyTorch.
                child = bodies[record.id]
            child.meta[REGION_ID_META_KEY] = record.id
            gm.add_submodule(child_name, child)
        operands = tuple(node.args[1])
        with gm.graph.inserting_before(node):
            call = (
                gm.graph.call_module(child_name, operands)
                if route == "torch"
                else gm.graph.call_function(plugin_op, operands)
            )
            call.meta = copy.copy(node.meta)
            call.meta[REGION_META_KEY] = {
                "id": record.id,
                "route": route,
                "backend": record.backend,
            }
        if route == "plugin" and len(node.meta["val"]) == 1:
            call.meta["val"] = node.meta["val"][0]
            for user in list(node.users):
                if (
                    user.op != "call_function"
                    or user.target != operator.getitem
                    or user.args != (node, 0)
                ):
                    raise RegionError(
                        "Plugin region has an unsupported output boundary"
                    )
                user.replace_all_uses_with(call)
                gm.graph.erase_node(user)
        else:
            node.replace_all_uses_with(call)
        gm.graph.erase_node(node)
        if not body_node.users:
            gm.graph.erase_node(body_node)
            gm.delete_submodule(str(body_node.target))
        output = next(n for n in child.graph.nodes if n.op == "output")
        record.child_target = child_name
        record.input_names = tuple(
            n.name for n in child.graph.nodes if n.op == "placeholder"
        )
        record.output_names = tuple(n.name for n in output.all_input_nodes)
        record.reference_graph = str(child.graph) if route == "torch" else None
        record.plugin_op = plugin_op
    attach_region_records(gm, records)
    gm.graph.lint()
    gm.recompile()
    audit_region_placement(gm, records)
    return gm
