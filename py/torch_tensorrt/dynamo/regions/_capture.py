# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Consume strict-export scope markers before decomposition and partitioning."""

import copy
import logging
import operator
from typing import Any, Optional

import torch
from torch.fx.passes.utils.fuser_utils import erase_nodes, fuse_as_graphmodule

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
TOKEN_KEY = "torch_tensorrt_region_token"
NAME_KEY = "torch_tensorrt_region_name"


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


def reject_region_capture(ep: torch.export.ExportedProgram, path: str) -> None:
    for module in ep.graph_module.modules():
        if isinstance(module, torch.fx.GraphModule) and any(
            _marker_kind(node) or is_region_hop(node) for node in module.graph.nodes
        ):
            raise RegionError(f"execute_in_torch regions do not support {path}")


def assert_no_unresolved_regions(gm: torch.fx.GraphModule) -> None:
    for module in gm.modules():
        if isinstance(module, torch.fx.GraphModule) and any(
            _marker_kind(node) or is_region_hop(node) for node in module.graph.nodes
        ):
            raise RegionError(
                "Region capture markers/HOPs must be resolved before support analysis"
            )


def _without_capture_tags(meta: dict[str, Any]) -> dict[str, Any]:
    result = copy.copy(meta)
    if "custom" in result:
        custom = dict(result["custom"])
        custom.pop(TOKEN_KEY, None)
        custom.pop(NAME_KEY, None)
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
                )
            )
    return records


def _tensor_values(value: Any) -> list[torch.Tensor]:
    values: list[torch.Tensor] = torch.utils._pytree.tree_leaves(value)
    if not values or any(not isinstance(item, torch.Tensor) for item in values):
        raise RegionCaptureError("execute_in_torch requires Tensor region boundaries")
    for tensor in values:
        if any(isinstance(size, torch.SymInt) for size in tensor.shape):
            raise RegionCaptureError(
                "execute_in_torch currently requires static shapes"
            )
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
                f"Mutation or effects inside execute_in_torch are unsupported: {target}"
            )
        tags = getattr(target, "tags", ())
        if torch.Tag.nondeterministic_seeded in tags:
            raise RegionCaptureError(
                f"Random operations inside a region are unsupported: {target}"
            )


def normalize_region_scopes(
    ep: torch.export.ExportedProgram, settings: Any = None
) -> torch.export.ExportedProgram:
    """Validate exact membership and replace scopes with durable HOP children."""
    gm = ep.graph_module
    for child in gm.modules():
        if (
            child is not gm
            and isinstance(child, torch.fx.GraphModule)
            and any(_marker_kind(node) for node in child.graph.nodes)
        ):
            raise RegionCaptureError(
                "execute_in_torch inside higher-order control flow is unsupported"
            )
    if not any(_marker_kind(node) for node in gm.graph.nodes):
        if any(TOKEN_KEY in node.meta.get("custom", {}) for node in gm.graph.nodes):
            raise RegionCaptureError(
                "Region membership survived without its boundary markers"
            )
        return ep
    try:
        from torch._higher_order_ops.hints_wrap import hints_wrapper
    except ImportError as error:
        raise RegionCaptureError(
            "execute_in_torch requires PyTorch's experimental hints_wrapper HOP"
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
                raise RegionCaptureError(
                    "Nested execute_in_torch scopes are unsupported"
                )
            active = node
            members = []
        elif kind == "end":
            if active is None or node.args != (active,):
                raise RegionCaptureError(
                    "Unmatched or crossed execute_in_torch markers"
                )
            token = custom.get(TOKEN_KEY)
            name = active.args[0]
            if not isinstance(token, int) or token in tokens:
                raise RegionCaptureError("Missing or reused region capture token")
            if custom.get(NAME_KEY) != name:
                raise RegionCaptureError("Region label and boundary disagree")
            for member in members:
                tag = member.meta.get("custom", {})
                if tag.get(TOKEN_KEY) != token or tag.get(NAME_KEY) != name:
                    raise RegionCaptureError(
                        f"Region membership disagrees with markers at {member.name!r}"
                    )
            if set(active.users) != {node}:
                raise RegionCaptureError(
                    "Region sentinel escaped into model computation"
                )
            tokens.add(token)
            scopes.append((active, node, members, name))
            active = None
        elif node.op in ("call_function", "call_method", "call_module"):
            if active is not None:
                members.append(node)
            elif TOKEN_KEY in custom:
                raise RegionCaptureError(f"Operation {node.name!r} escaped its region")
    if active is not None:
        raise RegionCaptureError("Unclosed execute_in_torch scope")

    for index, (begin, end, members, name) in enumerate(scopes):
        region_id = f"{name or 'region'}#{index}"
        _validate_members(members)
        live = [
            node for node in members if any(user not in members for user in node.users)
        ]
        if not live:
            erase_nodes(gm, members)
            gm.graph.erase_node(end)
            gm.graph.erase_node(begin)
            logger.info("execute_in_torch %s is an empty/dead pure scope", region_id)
            continue
        if ep.range_constraints:
            raise RegionCaptureError(
                "execute_in_torch currently requires static shapes"
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
                f"Cannot extract region {region_id}: {error}"
            ) from error
        input_values = []
        for node in inputs:
            if not isinstance(node.meta.get("val"), torch.Tensor):
                raise RegionCaptureError("execute_in_torch requires Tensor live-ins")
            input_values.extend(_tensor_values(node.meta["val"]))
        output_values = []
        for node in outputs:
            output_values.extend(_tensor_values(node.meta.get("val")))
        devices = {value.device for value in input_values + output_values}
        if len(devices) > 1:
            raise RegionCaptureError("A region cannot span multiple devices")
        for output in output_values:
            if any(torch._C._is_alias_of(output, value) for value in input_values):
                raise RegionCaptureError("Region outputs must not alias region inputs")
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
                        "route": "torch",
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
    gm.graph.lint()
    gm.recompile()
    ep.validate()
    if settings is not None:
        validate_region_settings(settings, region_hop_records(gm))
    return ep


def materialize_torch_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    records: list[RegionRecord] = []
    for node in list(gm.graph.nodes):
        if not is_region_hop(node):
            continue
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
        child._torch_tensorrt_region_id = hints["region_id"]
        gm.add_submodule(child_name, child)
        operands = tuple(node.args[1])
        with gm.graph.inserting_before(node):
            call = gm.graph.call_module(child_name, operands)
            call.meta = copy.copy(node.meta)
            call.meta[REGION_META_KEY] = {"id": hints["region_id"], "route": "torch"}
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
            )
        )
    attach_region_records(gm, records)
    gm.graph.lint()
    gm.recompile()
    audit_region_placement(gm, records)
    return gm
