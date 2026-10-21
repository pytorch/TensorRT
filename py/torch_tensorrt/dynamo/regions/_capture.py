# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Consume strict-export scope markers before decomposition and partitioning."""

import copy
import logging
import operator
from typing import Any, Optional

import torch
from torch.fx.passes.utils.fuser_utils import erase_nodes, fuse_as_graphmodule

from torch_tensorrt.region._constants import (
    BACKEND_KEY,
    NAME_KEY,
    PLUGIN_KEY,
    ROUTE_KEY,
    SUPPORTED_BACKENDS,
    TOKEN_KEY,
)
from torch_tensorrt.region._errors import RegionCaptureError, RegionError

from ._types import (
    HOP_SCHEMA_KEY,
    REGION_META_KEY,
    RegionRecord,
    attach_region_records,
    audit_region_placement,
    validate_region_settings,
)

logger = logging.getLogger(__name__)


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


def _without_capture_tags(meta: dict[str, Any]) -> dict[str, Any]:
    result = copy.copy(meta)
    if "custom" in result:
        custom = dict(result["custom"])
        for key in (TOKEN_KEY, NAME_KEY, BACKEND_KEY, ROUTE_KEY, PLUGIN_KEY):
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


def _validate_members(members: list[torch.fx.Node]) -> None:
    for node in members:
        target = node.target
        if isinstance(target, torch._ops.HigherOrderOperator):
            raise RegionCaptureError(
                "Higher-order control flow inside a region is unsupported"
            )
        schema = getattr(target, "_schema", None)
        if (schema is not None and schema.is_mutable) or node.is_impure():
            raise RegionCaptureError(
                f"Mutation or effects inside regions are unsupported: {target}"
            )
        tags = getattr(target, "tags", ())
        if torch.Tag.nondeterministic_seeded in tags:
            raise RegionCaptureError(
                f"Random operations inside a region are unsupported: {target}"
            )


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
            # Strict export implements torch.tensor literals with a fresh copy
            # followed by detach_. No other user can observe this fresh tensor's
            # metadata mutation, so an ordinary detach has the same semantics.
            # Input/state detaches and all other mutations remain unsupported.
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
                # Both complex lowering paths transform the whole graph. Even
                # complex work outside an otherwise real-valued region can
                # change its boundary representation or inline its child.
                raise RegionCaptureError(
                    "Region capture currently does not support complex tensors "
                    "in a model containing nonempty regions; complex lowering "
                    "cannot preserve region boundaries"
                )


def normalize_region_scopes(
    ep: torch.export.ExportedProgram, settings: Any = None
) -> torch.export.ExportedProgram:
    """Validate exact membership and replace scopes with durable HOP children."""
    gm = ep.graph_module
    for child in gm.modules():
        if not isinstance(child, torch.fx.GraphModule):
            continue
        has_markers = any(_marker_kind(node) for node in child.graph.nodes)
        if child is not gm and (
            has_markers or any(is_region_hop(node) for node in child.graph.nodes)
        ):
            raise RegionCaptureError(
                "Regions inside higher-order control flow are unsupported"
            )
        if not has_markers and any(
            TOKEN_KEY in node.meta.get("custom", {}) for node in child.graph.nodes
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
            name = active.args[0]
            backend = (
                active.args[1]
                if len(active.args) > 1
                else active.kwargs.get("backend", "eager")
            )
            if backend not in SUPPORTED_BACKENDS:
                raise RegionCaptureError(f"Unsupported region backend {backend!r}")
            route = (
                active.args[2]
                if len(active.args) > 2
                else active.kwargs.get("route", "torch")
            )
            plugin = (
                active.args[3]
                if len(active.args) > 3
                else active.kwargs.get("plugin", -1)
            )
            if route not in ("torch", "plugin") or (route == "plugin" and plugin < 0):
                raise RegionCaptureError("Invalid region plugin boundary")
            if not isinstance(token, int) or token in tokens:
                raise RegionCaptureError("Missing or reused region capture token")
            if custom.get(NAME_KEY) != name:
                raise RegionCaptureError("Region label and boundary disagree")
            if custom.get(BACKEND_KEY, "eager") != backend:
                raise RegionCaptureError("Region backend and boundary disagree")
            if (
                custom.get(ROUTE_KEY, "torch") != route
                or custom.get(PLUGIN_KEY, -1) != plugin
            ):
                raise RegionCaptureError("Region route and boundary disagree")
            for member in members:
                tag = member.meta.get("custom", {})
                if (
                    tag.get(TOKEN_KEY) != token
                    or tag.get(NAME_KEY) != name
                    or tag.get(BACKEND_KEY, "eager") != backend
                    or tag.get(ROUTE_KEY, "torch") != route
                    or tag.get(PLUGIN_KEY, -1) != plugin
                ):
                    raise RegionCaptureError(
                        f"Region membership disagrees with markers at {member.name!r}"
                    )
            if set(active.users) != {node}:
                raise RegionCaptureError(
                    "Region sentinel escaped into model computation"
                )
            tokens.add(token)
            scopes.append((active, node, members, name, backend, route, plugin))
            active = None
        elif node.op in ("call_function", "call_method", "call_module"):
            if active is not None:
                members.append(node)
            elif TOKEN_KEY in custom:
                raise RegionCaptureError(f"Operation {node.name!r} escaped its region")
    if active is not None:
        raise RegionCaptureError("Unclosed region scope")

    for index, (begin, end, members, name, backend, route, plugin) in enumerate(scopes):
        region_id = f"{name or 'region'}#{index}"
        functionalized_constructor = _functionalize_tensor_constructor_detaches(members)
        _validate_members(members)
        member_set = set(members)
        if not any(user not in member_set for node in members for user in node.users):
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
                f"Cannot extract region {region_id}: {error}"
            ) from error
        if route == "plugin":
            # FX may sort an all-placeholder boundary by model argument order
            # (including lifted parameters). Kernel pointers have a positional
            # ABI: consistently use first encounter in the captured region.
            ordered_inputs = tuple(
                dict.fromkeys(
                    arg
                    for member in members
                    for arg in member.all_input_nodes
                    if arg not in member_set
                )
            )
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

            # Export's tensor-literal metadata can alias the lifted constant even
            # though lift_fresh_copy allocates fresh storage. Recompute metadata
            # for this pure child so boundary checks see the true relationships
            # of views derived from the constructor. Keep the refreshed values
            # local: partially updating parent metadata would break its alias
            # relationships with later scopes until decomposition recomputes it.
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
        devices = {value.device for value in input_values + output_values}
        if len(devices) > 1:
            raise RegionCaptureError("A region cannot span multiple devices")
        for output_index, output in enumerate(output_values):
            if any(torch._C._is_alias_of(output, value) for value in input_values):
                raise RegionCaptureError(
                    f"Region {region_id} outputs must not alias region inputs; "
                    "clone the aliased outputs inside the region"
                )
            if any(
                torch._C._is_alias_of(output, value)
                for value in output_values[:output_index]
            ):
                raise RegionCaptureError(
                    f"Region {region_id} outputs must not alias each other; "
                    "clone outputs that share storage (for example, views or "
                    "chunk results) inside the region"
                )
        # Capture tokens are ephemeral and must not leak to later exports.
        for node in child.graph.nodes:
            node.meta = _without_capture_tags(node.meta)
        gm.add_submodule(child_name, child)
        with gm.graph.inserting_before(begin):
            body = gm.graph.get_attr(child_name)
        # All live-ins have been produced by the last member. Validated extraction
        # guarantees no external consumer has to run before this replacement.
        with gm.graph.inserting_before(end):
            hop = gm.graph.call_function(
                hints_wrapper,
                args=(body, inputs, {}),
                kwargs={
                    "hints": {
                        HOP_SCHEMA_KEY: 1,
                        "region_id": region_id,
                        "region_name": name,
                        "route": route,
                        "plugin_key": plugin,
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
    if settings is not None:
        validate_region_settings(settings, region_hop_records(gm))
    # Export can copy a scope's annotations onto lifted constant placeholders,
    # which remain in the parent after both live and dead scopes are extracted.
    # Remove that transient provenance too, while retaining unrelated metadata.
    for node in gm.graph.nodes:
        node.meta = _without_capture_tags(node.meta)
    return ep


def materialize_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Replace captured HOPs with PyTorch leaves or custom-kernel operators."""
    region_nodes = [node for node in gm.graph.nodes if is_region_hop(node)]
    if not region_nodes:
        return gm
    records: list[RegionRecord] = []
    for node in region_nodes:
        hints = node.kwargs["hints"]
        body_node = node.args[0]
        child = gm.get_submodule(str(body_node.target))
        if not isinstance(child, torch.fx.GraphModule):
            raise RegionError("Region HOP does not reference a GraphModule")
        child_name = f"_ttrt_torch_region_{len(records)}"
        while hasattr(gm, child_name):
            child_name += "_"
        # The reference remains independent of subsequent parent rewriting.
        reference = copy.deepcopy(child)
        route = hints.get("route", "torch")
        plugin_op = None
        if route == "plugin":
            from torch_tensorrt.region._session import current_session

            from ._plugins import register_region_plugin

            session = current_session()
            key = hints.get("plugin_key", -1)
            if session is None or key not in session.plugins:
                raise RegionError(
                    "fuse_in_plugin requires its owning compilation session"
                )
            plugin_op = register_region_plugin(
                reference, session.plugins[key], region_id=hints["region_id"]
            )
        if route == "torch":
            child._torch_tensorrt_region_id = hints["region_id"]
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
                "id": hints["region_id"],
                "route": route,
                "backend": hints.get("backend", "eager"),
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
        records.append(
            RegionRecord(
                id=hints["region_id"],
                name=hints["region_name"],
                child_target=child_name,
                input_names=tuple(
                    n.name for n in child.graph.nodes if n.op == "placeholder"
                ),
                output_names=tuple(n.name for n in output.all_input_nodes),
                reference=reference,
                backend=hints.get("backend", "eager"),
                route=route,
                plugin_key=hints.get("plugin_key", -1),
                plugin_op=plugin_op,
            )
        )
    attach_region_records(gm, records)
    gm.graph.lint()
    gm.recompile()
    audit_region_placement(gm, records)
    return gm
