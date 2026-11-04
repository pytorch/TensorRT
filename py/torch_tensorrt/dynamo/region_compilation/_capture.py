# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Consume strict-export scope markers before decomposition and partitioning."""

import copy
import logging
import operator
import os
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

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
    SUPPORTED_BACKENDS,
    TOKEN_KEY,
)
from torch_tensorrt.region._errors import RegionCaptureError, RegionError

from ._types import (
    HOP_SCHEMA_KEY,
    REGION_ID_META_KEY,
    REGION_META_KEY,
    RegionRecord,
    attach_region_records,
    audit_region_placement,
)
from ._validation import validate_output_aliases

if TYPE_CHECKING:
    from torch_tensorrt.dynamo._settings import CompilationSettings
    from torch_tensorrt.region._session import RegionCompilationSession

logger = logging.getLogger(__name__)
# The metadata a region scope puts on its members.
_SCOPE_TAGS = (TOKEN_KEY, NAME_KEY, BACKEND_KEY, ROUTE_KEY, PLUGIN_KEY, OPERATOR_KEY)
# Marks the copies _copy_shared_inputs makes of inputs that share storage.
_ALIAS_COPY_KEY = "torch_tensorrt_region_alias_copy"


def _marker_kind(node: torch.fx.Node) -> Optional[str]:
    if node.op == "call_function" and isinstance(node.target, torch._ops.OpOverload):
        if node.target._schema.name == "torch_tensorrt_region::begin":
            return "begin"
        if node.target._schema.name == "torch_tensorrt_region::end":
            return "end"
    return None


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


def reject_region_capture(ep: torch.export.ExportedProgram, path: str) -> None:
    if _has_unresolved_regions(ep.graph_module):
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
                    plugin_key=hints.get("plugin_key", -1),
                    operator_key=hints.get("operator_key", -1),
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


def _validate_members(
    members: list[torch.fx.Node], region_id: str, *, allow_random: bool
) -> None:
    for node in members:
        target = node.target
        if node.is_impure(impure_random=False):
            raise RegionCaptureError(
                f"Mutation or effects inside regions are unsupported: {target} "
                f"in {region_id!r}"
            )
        if not allow_random and _draws_random_numbers(node):
            raise RegionCaptureError(
                "Random operations are supported only in fused_op regions: "
                f"{target} in {region_id!r}"
            )


def _returns_its_input(node: torch.fx.Node) -> bool:
    """Whether the op returns its first argument itself, as eval-mode dropout does."""
    if not isinstance(node.target, torch._ops.OpOverload):
        return False
    args, kwargs = torch.fx.map_arg(
        (node.args, node.kwargs), lambda arg: arg.meta.get("val")
    )
    value = node.meta.get("val")
    # Views alias their input too; only the op's own kernel tells them apart.
    return (
        bool(args)
        and isinstance(args[0], torch.Tensor)
        and isinstance(value, torch.Tensor)
        and torch._C._is_alias_of(value, args[0])
        and node.target.decompose(*args, **kwargs) is args[0]
    )


def _bypass_members_returning_inputs(
    ep: torch.export.ExportedProgram,
    members: list[torch.fx.Node],
    fused: Optional[str],
) -> None:
    """Make reads after a region bypass members that return their input itself.

    ``fused`` names a fused_op or fused_kernel region, whose operator or kernel
    may produce such a member as an output, so a bypass to an input raises.
    """
    # Export binds later reads of a tensor to the last op that returned it, so
    # eval-mode dropout or a no-op cast would make the region output its input.
    # Reverse order resolves chains such as x.float().float().
    member_set = set(members)
    for node in reversed(members):
        if not _returns_its_input(node):
            continue
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
        ep.graph_signature.replace_all_uses(node.name, source.name)


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
    return ", ".join(
        f"{node.name}{list(getattr(node.meta.get('val'), 'shape', ()))}"
        for node in nodes
    )


def _generated(plugin_key: int) -> bool:
    """Whether a fused_kernel region's kernel comes from a KernelGenerator.

    A generated kernel is written for the order the region first reads its
    inputs, so such a region needs no ``inputs=``.
    """
    from torch_tensorrt.region import KernelGenerator
    from torch_tensorrt.region._session import current_session

    session = current_session()
    return session is not None and isinstance(
        session.plugins.get(plugin_key), KernelGenerator
    )


def _bind_inputs(
    live_ins: tuple[torch.fx.Node, ...],
    listed: Optional[Sequence[torch.fx.Node]],
    region: str,
) -> tuple[torch.fx.Node, ...]:
    """Order a replacement's arguments as its ``inputs=`` lists them."""
    # Traced first use is not source order: ``up * silu(gate)`` reads gate first.
    if listed is None:
        if len(live_ins) <= 1:
            return live_ins
        raise RegionCaptureError(
            f"{region} reads {len(live_ins)} tensors ({_describe(live_ins)}); "
            "pass inputs=(...) to list them in argument order"
        )
    bound = tuple(listed)
    if len(set(bound)) != len(bound) or set(bound) != set(live_ins):
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


def normalize_region_scopes(
    ep: torch.export.ExportedProgram,
) -> torch.export.ExportedProgram:
    """Validate exact membership and replace scopes with durable HOP children."""
    gm = ep.graph_module
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
        return ep
    try:
        from torch._higher_order_ops.hints_wrap import hints_wrapper
    except ImportError as error:
        raise RegionCaptureError(
            "Region capture requires PyTorch's experimental hints_wrapper HOP"
        ) from error

    for spec in (*ep.graph_signature.input_specs, *ep.graph_signature.output_specs):
        if spec.kind.name == "TOKEN":
            raise RegionCaptureError(
                "Region markers reached effect-token functionalization"
            )
    # Drop export's asserts now rather than in pre-lowering: the guard on each
    # aten.to copies the region tags of the cast's input, not the cast's, so the
    # membership checks below would misplace or reject it.
    apply_remove_assert_nodes(gm)

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
            if route not in ("torch", "plugin") or (route == "plugin" and plugin < 0):
                raise RegionCaptureError("Invalid region plugin boundary")
            if operator_key >= 0 and (
                route != "torch" or backend != "eager" or plugin >= 0
            ):
                raise RegionCaptureError("Invalid fused_op region boundary")
            if not isinstance(token, int) or token in tokens:
                raise RegionCaptureError("Missing or reused region capture token")
            expected_tags = (
                (TOKEN_KEY, token, None),
                (NAME_KEY, name, None),
                (BACKEND_KEY, backend, "eager"),
                (ROUTE_KEY, route, "torch"),
                (PLUGIN_KEY, plugin, -1),
                (OPERATOR_KEY, operator_key, -1),
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
                (active, node, members, name, backend, route, plugin, operator_key)
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
    ) in enumerate(scopes):
        region_id = f"{name or 'region'}#{index}"
        fused = None
        if route == "plugin" or operator_key >= 0:
            api = "fused_kernel" if route == "plugin" else "fused_op"
            fused = f"{api} region {region_id!r}"
        functionalized_constructor = _functionalize_tensor_constructor_detaches(members)
        _validate_members(members, region_id, allow_random=operator_key >= 0)
        _bypass_members_returning_inputs(ep, members, fused)
        member_set = set(members)
        if not any(user not in member_set for node in members for user in node.users):
            _reject_dead_random_region(members, region_id)
            erase_nodes(gm, members)
            gm.graph.erase_node(end)
            gm.graph.erase_node(begin)
            logger.info("Region %s is an empty/dead pure scope", region_id)
            continue
        if ep.range_constraints:
            raise RegionCaptureError("Region capture currently requires static shapes")
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
        listed = _boundary(begin)["inputs"] if fused is not None else None
        if fused is not None and (listed is not None or not _generated(plugin)):
            # Read the marker now: an earlier region may have replaced a listed tensor.
            ordered_inputs = _bind_inputs(inputs, listed, fused)
            if inputs != ordered_inputs:
                placeholders = dict(
                    zip(inputs, (n for n in child.graph.nodes if n.op == "placeholder"))
                )
                first_op = next(n for n in child.graph.nodes if n.op != "placeholder")
                for arg in ordered_inputs:
                    first_op.prepend(placeholders[arg])
                inputs = ordered_inputs
                child.graph.lint()
                child.recompile()
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
                ep.graph_signature.replace_all_uses(original.name, replacement.name)
        erase_nodes(gm, members)
        gm.graph.erase_node(end)
        gm.graph.erase_node(begin)
    _reject_complex_regions(gm)
    gm.graph.lint()
    gm.recompile()
    ep.validate()
    # Lifted constant placeholders can also inherit the scope's transient tags.
    for node in gm.graph.nodes:
        node.meta = _without_capture_tags(node.meta)
    return ep


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
        raise RegionCaptureError(
            f"fused_op region {region_id!r} output count or flat Tensor structure does not match its reference"
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
            raise RegionCaptureError(
                f"fused_op region {region_id!r} output {index} metadata does not match its reference"
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


def _kernel_registration(
    child: torch.fx.GraphModule,
    record: RegionRecord,
    session: Optional["RegionCompilationSession"],
    settings: Optional["CompilationSettings"],
) -> Any:
    """Return a fused_kernel registration, or None to keep a dry run's reference."""
    from torch_tensorrt.region import KernelGenerator

    if session is None or record.plugin_key not in session.plugins:
        raise RegionError("fused_kernel requires its owning compilation session")
    registration = session.plugins[record.plugin_key]
    if not isinstance(registration, KernelGenerator):
        return registration
    # Generation calls a paid agent and builds trial engines; a dry run does neither.
    if settings is not None and settings.dryrun:
        logger.warning(
            "Dry run shows region %r as its PyTorch reference instead of a "
            "generated kernel",
            record.id,
        )
        return None
    if os.name != "posix":
        raise RegionError(
            f"Kernel generation for region {record.id!r} requires a POSIX host; "
            "use fused_kernel(kernel=...) on this platform"
        )
    from ._generation._factory import generate_registration

    return generate_registration(
        child,
        registration,
        region_id=record.id,
        settings=settings,
        memo=session.generated_kernels,
    )


def materialize_regions(
    gm: torch.fx.GraphModule,
    captured: torch.fx.GraphModule,
    settings: Optional["CompilationSettings"] = None,
) -> torch.fx.GraphModule:
    """Replace captured HOPs with PyTorch leaves or custom-kernel operators.

    ``execute_in_torch`` regions run their bodies from ``captured``, the graph
    before decomposition. A dry run does not generate kernels: a generated
    region keeps its PyTorch reference body, like an eager ``execute_in_torch``
    region.
    """
    region_nodes = [node for node in gm.graph.nodes if is_region_hop(node)]
    if not region_nodes:
        return gm
    from torch_tensorrt.region._session import current_session

    session = current_session()
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
        plugin_op = None
        if record.route == "plugin":
            registration = _kernel_registration(child, record, session, settings)
            if registration is None:
                record.route = "torch"
            else:
                from ._plugins import register_region_plugin

                try:
                    plugin_op = register_region_plugin(
                        child, registration, region_id=record.id
                    )
                except ValueError as error:
                    # Kernel registration names the generated operator, not the
                    # region.
                    error.add_note(f"Raised for fused_kernel region {record.id!r}")
                    raise
        route = record.route
        # A body captured before decomposition reads the inputs, not their copies.
        _resolve_alias_copies(node, drop=route == "torch" and record.operator_key < 0)
        if route == "torch":
            if record.operator_key >= 0:
                if session is None or record.operator_key not in session.operators:
                    raise RegionError(
                        "fused_op requires its owning compilation session"
                    )
                child = _custom_op_body(
                    child, session.operators[record.operator_key], record.id
                )
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
