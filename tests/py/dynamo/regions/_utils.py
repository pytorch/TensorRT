# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared region capture, compilation, and subprocess helpers."""

import subprocess
import sys

import torch

import torch_tensorrt
from torch_tensorrt.dynamo.lowering._buffer_lifting import erase_export_guards
from torch_tensorrt.dynamo.region_compilation import (
    get_region_records,
    materialize_regions,
    normalize_region_scopes,
)
from torch_tensorrt.region._session import RegionCompilationSession


def capture_regions(model, *inputs):
    with RegionCompilationSession(strict=True):
        program = torch.export.export(model.eval(), inputs, strict=True)
        program = normalize_region_scopes(program).run_decompositions()
        graph = materialize_regions(program.module())
    erase_export_guards(graph)
    graph.graph.lint()
    graph.recompile()
    return graph


def compile_regions(model, inputs, **overrides):
    options = dict(
        ir="dynamo", strict=True, offload_module_to_cpu=False, min_block_size=1
    )
    options.update(overrides)
    return torch_tensorrt.compile(model, inputs=list(inputs), **options)


def graph_snapshot(graph):
    state = dict(graph.named_parameters()) | dict(graph.named_buffers())
    return (
        {
            name: str(child.graph)
            for name, child in graph.named_modules()
            if isinstance(child, torch.fx.GraphModule)
        },
        {name: id(value) for name, value in state.items()},
        [(r.id, r.backend, r.owner) for r in get_region_records(graph)],
    )


def run_python(*args, timeout=120, cwd=None):
    result = subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout
