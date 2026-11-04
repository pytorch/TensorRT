# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile-local records and invariants for explicit execution regions."""

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

import torch

from torch_tensorrt.region._constants import SUPPORTED_BACKENDS
from torch_tensorrt.region._errors import RegionError

REGION_META_KEY = "torch_tensorrt_region"
RECORDS_META_KEY = "torch_tensorrt_regions"
HOP_SCHEMA_KEY = "torch_tensorrt.region_schema"
logger = logging.getLogger(__name__)


@dataclass
class RegionRecord:
    id: str
    name: str
    child_target: str
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    route: str = "torch"
    owner: Optional[str] = None
    reference_graph: Optional[str] = field(default=None, repr=False)
    backend: str = "eager"
    plugin_key: int = -1
    operator_key: int = -1
    plugin_op: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.route not in ("torch", "plugin"):
            raise RegionError(f"Unsupported region route {self.route!r}")
        if self.backend not in SUPPORTED_BACKENDS:
            raise RegionError(f"Unsupported region backend {self.backend!r}")

    @property
    def capture_signature(self) -> tuple[str, str, str, int, int]:
        """Identity and execution policy that lowering must preserve."""
        return self.id, self.backend, self.route, self.plugin_key, self.operator_key


def get_region_records(gm: torch.nn.Module) -> list[RegionRecord]:
    """Return region records, or none for an ordinary, unannotated module."""
    return list(getattr(gm, "meta", {}).get(RECORDS_META_KEY, ()))


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


def validate_region_settings(
    gm: torch.fx.GraphModule, settings: Any, records: Sequence[RegionRecord]
) -> bool:
    """Reject incompatible settings, or record a conflict and stop a dry run."""
    if not records:
        return True
    conflicts = []
    if getattr(settings, "require_full_compilation", False) and any(
        record.route == "torch" for record in records
    ):
        conflicts.append("require_full_compilation=True")
    for name in (
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
    if not conflicts:
        return True
    message = (
        f"Region {records[0].id!r} is incompatible with "
        + ", ".join(conflicts)
        + ". The experimental region path requires these features disabled "
        "and immutable_weights=True."
    )
    if not settings.dryrun:
        raise RegionError(message)
    gm.meta["torch_tensorrt_region_conflicts"] = [message]
    logger.warning("Region dry-run stopped: %s", message)
    return False


def audit_region_placement(
    gm: torch.fx.GraphModule,
    records: Sequence[RegionRecord],
    *,
    require_plugins_in_trt: bool = False,
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
            # Partition inputs inherit metadata but do not invoke the region.
            if node.op == "placeholder":
                continue
            annotation = node.meta.get(REGION_META_KEY)
            # Count the plugin inside its partition, not the copied wrapper metadata.
            if (
                annotation is not None
                and annotation.get("route") == "plugin"
                and node.op == "call_module"
            ):
                annotation = None
            if annotation is not None:
                region_id = annotation.get("id")
                record = expected.get(region_id)
                if record is None:
                    raise RegionError(f"Unexpected region {region_id!r}")
                if record.route == "plugin":
                    if (
                        annotation.get("route") != "plugin"
                        or node.op != "call_function"
                        or node.target != record.plugin_op
                    ):
                        raise RegionError(
                            f"Plugin region {region_id!r} lost its kernel"
                        )
                    if require_plugins_in_trt and not accelerated:
                        raise RegionError(
                            f"fused_kernel region {region_id!r} must run in a TensorRT "
                            "plugin; check converter support, excluded ops and min_block_size"
                        )
                    occurrences[region_id].append(path or "<root>")
                    continue
                if not is_torch_region_node(module, node):
                    raise RegionError(
                        f"Region leaf {node.name!r} lost its child, target or route"
                    )
                if accelerated:
                    raise RegionError(
                        f"execute_in_torch region {region_id!r} entered a TensorRT partition"
                    )
                child = module.get_submodule(str(node.target))
                placeholders = [n for n in child.graph.nodes if n.op == "placeholder"]
                if len(placeholders) != len(node.args) or node.kwargs:
                    raise RegionError(f"Region {region_id!r} input boundary changed")
                if (
                    record.backend not in SUPPORTED_BACKENDS
                    or annotation.get("backend", "eager") != record.backend
                ):
                    raise RegionError(f"Region {region_id!r} backend changed")
                if (
                    record.reference_graph is not None
                    and str(child.graph) != record.reference_graph
                ):
                    raise RegionError(f"Region {region_id!r} reference body changed")
                occurrences[region_id].append(path or "<root>")
                continue
            if node.op == "call_module":
                child = module.get_submodule(str(node.target))
                child_path = f"{path}.{node.target}" if path else str(node.target)
                plugin_ids = getattr(child, "_torch_tensorrt_plugin_regions", ())
                if plugin_ids:
                    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

                    if not isinstance(child, TorchTensorRTModule):
                        raise RegionError(
                            "Plugin regions must belong to a TensorRT engine"
                        )
                    for region_id in plugin_ids:
                        if (
                            region_id not in expected
                            or expected[region_id].route != "plugin"
                        ):
                            raise RegionError(f"Unexpected plugin region {region_id!r}")
                        occurrences[region_id].append(child_path)
                if isinstance(child, torch.fx.GraphModule):
                    visit(
                        child,
                        child_path,
                        accelerated or str(node.target).startswith("_run_on_acc"),
                    )

    visit(gm, "", False)
    for region_id, owners in occurrences.items():
        if len(owners) != 1:
            raise RegionError(
                f"Region {region_id!r} must have exactly one call; found {len(owners)}"
            )
        expected[region_id].owner = owners[0]
