# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile-local records and invariants for explicit execution regions."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

import torch

from torch_tensorrt.region._constants import SUPPORTED_BACKENDS
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
    backend: str = "eager"
    plugin_key: int = -1
    plugin_op: Any = field(default=None, repr=False)
    _reference_graph: Optional[str] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.route not in ("torch", "plugin"):
            raise RegionError(f"Unsupported region route {self.route!r}")
        if self.backend not in SUPPORTED_BACKENDS:
            raise RegionError(f"Unsupported region backend {self.backend!r}")
        # The reference is an immutable capture snapshot. Format it once, rather
        # than again after every lowering pass and partitioning stage.
        if self.route == "torch" and self.reference is not None:
            self._reference_graph = str(self.reference.graph)

    @property
    def capture_signature(self) -> tuple[str, str, str, int]:
        """Identity and execution policy that lowering must preserve."""
        return self.id, self.backend, self.route, self.plugin_key


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


def validate_region_settings(settings: Any, records: Sequence[RegionRecord]) -> None:
    if not records:
        return
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
    if conflicts:
        raise RegionError(
            f"Region {records[0].id!r} is incompatible with "
            + ", ".join(conflicts)
            + ". The experimental region path requires these features disabled "
            "and immutable_weights=True."
        )


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
            # Splitters copy a producer's metadata onto partition inputs. A
            # placeholder carries provenance, not another region invocation.
            if node.op == "placeholder":
                continue
            annotation = node.meta.get(REGION_META_KEY)
            # The global partitioner also copies a single-output producer's
            # metadata onto the enclosing partition call. A plugin executes
            # inside that child; count its actual operator (or built engine
            # ownership) below, rather than treating the wrapper as the op.
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
                            f"fuse_in_plugin region {region_id!r} must run in a TensorRT "
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
                    record._reference_graph is not None
                    and str(child.graph) != record._reference_graph
                ):
                    raise RegionError(f"Region {region_id!r} reference body changed")
                occurrences[region_id].append(path or "<root>")
                continue
            if node.op == "call_module":
                child = module.get_submodule(str(node.target))
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
                        occurrences[region_id].append(str(node.target))
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
                f"Region {region_id!r} must have exactly one call; found {len(owners)}"
            )
        expected[region_id].owner = owners[0]
