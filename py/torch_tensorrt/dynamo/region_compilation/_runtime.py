# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Install region execution backends after all compiler graph rewriting."""

import functools
from typing import Any

import torch
from torch_tensorrt.region._errors import RegionError

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


def _compile_directly(child: torch.fx.GraphModule, backend: str) -> None:
    """Compile the body with Inductor itself on first call.

    The result of a torch.compile backend runs where PyTorch does not trace,
    so the body cannot start a trace of its own as ``_compile_lazily`` does.
    """
    if backend != "inductor":
        raise RegionError(f"Unsupported region backend {backend!r}")
    compiled: list[Any] = []

    def run(*args: Any) -> Any:
        if not compiled:
            # Compile once, and not while torch.compile compiles on another
            # thread: Inductor and Dynamo share tracing state.
            with torch._dynamo.convert_frame.compile_lock:
                if not compiled:
                    compiled.append(
                        torch._inductor.compile(
                            child, list(args), options={"triton.cudagraphs": False}
                        )
                    )
        return compiled[0](*args)

    child._compiled_call_impl = run
    child._torch_tensorrt_compiled_backend = backend


def configure_region_backends(
    gm: torch.fx.GraphModule, *, inside_torch_compile: bool = False
) -> None:
    """Enable lazy compilation on the finished runtime graph, after all rewriting.

    ``inside_torch_compile`` marks a graph that a torch.compile backend returns.
    """
    records = get_region_records(gm)
    if not records:
        return
    audit_region_placement(gm, records)
    compile_body = _compile_directly if inside_torch_compile else _compile_lazily

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
                compile_child = functools.partial(compile_body, backend=backend)
                compile_child(child)
                # GraphModule deepcopy drops the compiled callable; compile copies too.
                child._register_deepcopy_hook(compile_child)
            elif isinstance(child, torch.fx.GraphModule):
                visit(child)

    visit(gm)
