# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared region capture, compilation, serialization, and subprocess helpers."""

import subprocess
import sys

import torch
import torch_tensorrt
from torch_tensorrt.dynamo.lowering._buffer_lifting import erase_export_guards
from torch_tensorrt.dynamo.region_compilation import (
    check_decomposed_regions,
    get_region_records,
    materialize_regions,
    normalize_region_scopes,
)
from torch_tensorrt.region._session import RegionCompilationSession


def capture_regions(model, *inputs):
    with RegionCompilationSession(strict=True):
        program = torch.export.export(model.eval(), inputs, strict=True)
        program = normalize_region_scopes(program)
        captured = program.graph_module
        program = program.run_decompositions()
        check_decomposed_regions(captured, program.graph_module)
        graph = materialize_regions(program.module(), captured)
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


def save_and_load(module, directory):
    path = str(directory / "region.pt2")
    torch_tensorrt.save(
        module,
        path,
        output_format="exported_program",
        retrace=False,
        use_legacy_exporter=True,
    )
    return torch_tensorrt.load(path).module()


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
