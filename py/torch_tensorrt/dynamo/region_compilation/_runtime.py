# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Install region execution backends after all compiler graph rewriting."""

import functools

import torch

from ._types import REGION_META_KEY, audit_region_placement, get_region_records


def _compile_lazily(child: torch.fx.GraphModule, backend: str) -> None:
    # Keep the FX body; compile lazily without nesting CUDA captures.
    child.compile(
        backend=backend,
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False},
    )
    child._torch_tensorrt_compiled_backend = backend


def configure_region_backends(gm: torch.fx.GraphModule) -> None:
    """Enable lazy compilation on the finished runtime graph, after all rewriting."""
    records = get_region_records(gm)
    if not records:
        return
    audit_region_placement(gm, records)

    def visit(module: torch.fx.GraphModule) -> None:
        for node in module.graph.nodes:
            if node.op != "call_module":
                continue
            child = module.get_submodule(str(node.target))
            annotation = node.meta.get(REGION_META_KEY)
            if annotation is not None and annotation.get("route") == "torch":
                backend = annotation.get("backend", "eager")
                if backend == "eager" or (
                    getattr(child, "_torch_tensorrt_compiled_backend", None) == backend
                    and child._compiled_call_impl is not None
                ):
                    continue
                compile_child = functools.partial(_compile_lazily, backend=backend)
                compile_child(child)
                # GraphModule deepcopy drops the compiled callable; compile copies too.
                child._register_deepcopy_hook(compile_child)
            elif isinstance(child, torch.fx.GraphModule):
                visit(child)

    visit(gm)
