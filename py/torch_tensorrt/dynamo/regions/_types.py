# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile-local records and invariants for explicit PyTorch regions."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

import torch

from torch_tensorrt.region._errors import RegionError

REGION_META_KEY = "torch_tensorrt_region"
RECORDS_META_KEY = "torch_tensorrt_regions"
HOP_SCHEMA_KEY = "torch_tensorrt.region_schema"


@dataclass
class RegionRecord:
    id: str
    name: str
    child_target: str
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    route: str = "torch"
    owner: Optional[str] = None
    reference: Optional[torch.fx.GraphModule] = field(default=None, repr=False)


def get_region_records(gm: torch.fx.GraphModule) -> list[RegionRecord]:
    return list(gm.meta.get(RECORDS_META_KEY, ()))


def attach_region_records(
    gm: torch.fx.GraphModule, records: Sequence[RegionRecord]
) -> None:
    if records:
        gm.meta[RECORDS_META_KEY] = list(records)


def is_torch_region_node(
    gm_or_modules: Union[torch.fx.GraphModule, Mapping[str, torch.nn.Module]],
    node: torch.fx.Node,
) -> bool:
    annotation = node.meta.get(REGION_META_KEY)
    if (
        node.op != "call_module"
        or not isinstance(annotation, dict)
        or annotation.get("route") != "torch"
    ):
        return False
    try:
        child = (
            gm_or_modules.get_submodule(str(node.target))
            if isinstance(gm_or_modules, torch.fx.GraphModule)
            else gm_or_modules[str(node.target)]
        )
    except (AttributeError, KeyError):
        return False
    return isinstance(child, torch.fx.GraphModule) and getattr(
        child, "_torch_tensorrt_region_id", None
    ) == annotation.get("id")


def validate_region_settings(settings: Any, records: Sequence[RegionRecord]) -> None:
    if not records:
        return
    conflicts = []
    for name in (
        "require_full_compilation",
        "offload_module_to_cpu",
        "enable_autocast",
        "enable_resource_partitioning",
        "dynamically_allocate_resources",
        "enable_weight_streaming",
        "strip_engine_weights",
        "enable_cross_compile_for_windows",
    ):
        if getattr(settings, name, False):
            conflicts.append(f"{name}=True")
    if not getattr(settings, "immutable_weights", True):
        conflicts.append("immutable_weights=False")
    if conflicts:
        raise RegionError(
            f"execute_in_torch region {records[0].id!r} is incompatible with "
            + ", ".join(conflicts)
            + ". The experimental region path requires these features disabled "
            "and immutable_weights=True."
        )


def audit_region_placement(
    gm: torch.fx.GraphModule, records: Sequence[RegionRecord]
) -> None:
    """Verify actual call ownership, including nested fallback partitions."""
    if not records:
        return
    expected = {record.id: record for record in records}
    if len(expected) != len(records):
        raise RegionError("Duplicate region IDs in the placement manifest")
    occurrences: dict[str, list[str]] = {key: [] for key in expected}

    def visit(module: torch.fx.GraphModule, path: str, accelerated: bool) -> None:
        for node in module.graph.nodes:
            # Splitters copy a producer's metadata onto partition inputs. A
            # placeholder carries provenance, not another region invocation.
            if node.op == "placeholder":
                continue
            annotation = node.meta.get(REGION_META_KEY)
            if annotation is not None:
                if not is_torch_region_node(module, node):
                    raise RegionError(
                        f"Region leaf {node.name!r} lost its child, target or route"
                    )
                region_id = annotation["id"]
                if region_id not in expected:
                    raise RegionError(f"Unexpected region {region_id!r}")
                if accelerated:
                    raise RegionError(
                        f"execute_in_torch region {region_id!r} entered a TensorRT partition"
                    )
                child = module.get_submodule(str(node.target))
                placeholders = [n for n in child.graph.nodes if n.op == "placeholder"]
                if len(placeholders) != len(node.args) or node.kwargs:
                    raise RegionError(f"Region {region_id!r} input boundary changed")
                record = expected[region_id]
                if record.reference is not None and str(child.graph) != str(
                    record.reference.graph
                ):
                    raise RegionError(f"Region {region_id!r} reference body changed")
                occurrences[region_id].append(path or "<root>")
                continue
            if node.op == "call_module":
                child = module.get_submodule(str(node.target))
                if isinstance(child, torch.fx.GraphModule):
                    child_path = f"{path}.{node.target}" if path else str(node.target)
                    visit(
                        child,
                        child_path,
                        accelerated or str(node.target).startswith("_run_on_acc"),
                    )

    visit(gm, "", False)
    for region_id, owners in occurrences.items():
        if len(owners) != 1:
            raise RegionError(
                f"Region {region_id!r} must have exactly one PyTorch call; found {len(owners)}"
            )
        expected[region_id].owner = owners[0]
