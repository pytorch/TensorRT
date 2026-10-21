# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Install region execution backends after all compiler graph rewriting."""

import torch

from ._types import REGION_META_KEY, audit_region_placement, get_region_records


def configure_region_backends(gm: torch.fx.GraphModule) -> None:
    """Keep FX leaves intact while configuring their lazy runtime compilation.

    Call only on the finished runtime module, never during lowering, partitioning,
    or dry-run analysis. Intermediate passes must use the ordinary FX bodies.
    """
    records = get_region_records(gm)
    if not records:
        return
    audit_region_placement(gm, records)
    backends = {
        record.id: record.backend for record in records if record.route == "torch"
    }

    def visit(module: torch.fx.GraphModule) -> None:
        for node in module.graph.nodes:
            if node.op != "call_module":
                continue
            child = module.get_submodule(str(node.target))
            annotation = node.meta.get(REGION_META_KEY)
            if annotation is not None and annotation.get("route") == "torch":
                backend = backends[annotation["id"]]
                if backend == "eager":
                    continue
                if (
                    getattr(child, "_torch_tensorrt_compiled_backend", None) == backend
                    and child._compiled_call_impl is not None
                ):
                    continue
                # Module.compile retains the GraphModule and reference graph.
                # Inductor compiles lazily on first real execution. fullgraph
                # makes graph breaks/compiler failures visible to the caller.
                # Disable its CUDA graphs to avoid nested runtime captures.
                child.compile(
                    backend=backend,
                    fullgraph=True,
                    dynamic=False,
                    options={"triton.cudagraphs": False},
                )
                child._torch_tensorrt_compiled_backend = backend
            elif isinstance(child, torch.fx.GraphModule):
                visit(child)

    visit(gm)
