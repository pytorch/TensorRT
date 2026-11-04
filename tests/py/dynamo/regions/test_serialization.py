# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU coverage for region serialization; engine roundtrips live in integration tests."""

import copy
import io

import pytest
import torch

import torch_tensorrt
from torch_tensorrt import region
from torch_tensorrt.dynamo._exporter import export, transform
from torch_tensorrt.dynamo.region_compilation import (
    RegionRecord,
    attach_region_records,
    audit_region_placement,
    get_region_records,
)
from torch_tensorrt.dynamo.region_compilation._types import REGION_META_KEY

from ._utils import capture_regions as _captured_module
from ._utils import graph_snapshot


class _TwoOutputs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3, 3), requires_grad=False)
        self.register_buffer("bias", torch.randn(3))

    def forward(self, x):
        h = x + 1
        with region.execute_in_torch(name="branches"):
            y = torch.relu(h @ self.weight)
            z = torch.sigmoid(h + self.bias)
        return y + x, z


@pytest.mark.unit
@pytest.mark.parametrize("nested", [False, True], ids=["direct", "fallback_child"])
def test_legacy_export_keeps_region_state_outputs_and_source(nested):
    model, x = _TwoOutputs().eval(), torch.randn(2, 3)
    source = _captured_module(model, x)
    if nested:
        graph = torch.fx.Graph()
        arg = graph.placeholder("x")
        arg.meta = copy.copy(
            next(n for n in source.graph.nodes if n.op == "placeholder").meta
        )
        root = torch.nn.Module()
        root.add_module("_run_on_gpu_0", source)
        call = graph.call_module("_run_on_gpu_0", (arg,))
        # A tuple returned directly must survive just like individual getitems.
        graph.output(call)
        parent = torch.fx.GraphModule(root, graph)
        attach_region_records(parent, get_region_records(source))
        source = parent

    before = graph_snapshot(source)
    result = export(source, use_legacy_exporter=True)
    artifact = io.BytesIO()
    torch.export.save(result, artifact)
    artifact.seek(0)
    restored = torch.export.load(artifact)

    torch.testing.assert_close(restored.module()(x), model(x))
    assert not any(n.op == "call_module" for n in restored.graph.nodes)
    assert "torch_tensorrt_regions" not in restored.graph_module.meta
    provenance = restored.graph_module.meta["custom"]["torch_tensorrt_regions"]
    assert provenance == [
        {
            "id": "branches#0",
            "name": "branches",
            "route": "torch",
            "owner": "_run_on_gpu_0" if nested else "<root>",
        }
    ]
    assert graph_snapshot(source) == before
    audit_region_placement(source, get_region_records(source))


@pytest.mark.unit
def test_region_child_constants_do_not_overwrite_sibling_or_parent_state():
    class Body(torch.nn.Module):
        def __init__(self, bias):
            super().__init__()
            self.register_buffer("bias", torch.tensor(bias))

        def forward(self, x):
            return x + self.bias, x - self.bias

    root = torch.nn.Module()
    root.register_buffer("_ttrt_region_export_state_0", torch.tensor(99.0))
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    calls, records = [], []
    for index, bias in enumerate((2.0, 7.0)):
        region_id = f"region#{index}"
        child = torch.fx.symbolic_trace(Body(bias))
        # Child-owned constants are allowed to be non-persistent before export.
        child._non_persistent_buffers_set.add("bias")
        child._torch_tensorrt_region_id = region_id
        fallback_root = torch.nn.Module()
        fallback_root.add_module("region_body", child)
        fallback_graph = torch.fx.Graph()
        arg = fallback_graph.placeholder("x")
        region_call = fallback_graph.call_module("region_body", (arg,))
        region_call.meta[REGION_META_KEY] = {"id": region_id, "route": "torch"}
        fallback_graph.output(region_call)
        fallback = torch.fx.GraphModule(fallback_root, fallback_graph)
        name = f"_run_on_gpu_{index}"
        root.add_module(name, fallback)
        calls.append(graph.call_module(name, (x,)))
        records.append(RegionRecord(region_id, "", "region_body"))
    graph.output(tuple(calls))
    source = torch.fx.GraphModule(root, graph)
    # Retain an unused name to test collisions during state promotion.
    source.register_buffer("_ttrt_region_export_state_0", torch.tensor(99.0))
    attach_region_records(source, records)
    inputs = torch.tensor([1.0, 3.0])
    expected = source(inputs)

    flattened = transform(source)

    torch.testing.assert_close(flattened(inputs), expected)
    assert not any(n.op == "call_module" for n in flattened.graph.nodes)
    assert flattened._ttrt_region_export_state_0.item() == 99.0
    assert {value.item() for _, value in flattened.named_buffers()} == {2.0, 7.0, 99.0}
    audit_region_placement(source, get_region_records(source))


@pytest.mark.unit
@pytest.mark.parametrize(
    "options",
    [
        {"use_legacy_exporter": False},
        {"use_legacy_exporter": True, "cross_compile_module": True},
    ],
)
def test_exporter_rejects_unqualified_region_paths(options):
    x = torch.randn(2, 3)
    source = _captured_module(_TwoOutputs().eval(), x)
    with pytest.raises(region.RegionError, match="legacy exported_program"):
        export(source, arg_inputs=(x,), **options)


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"output_format": "torchscript"},
        {"retrace": True},
        {"use_legacy_exporter": False},
    ],
)
def test_save_rejects_unqualified_region_paths(overrides, tmp_path):
    x = torch.randn(2, 3)
    source = _captured_module(_TwoOutputs(), x)
    options = dict(
        output_format="exported_program", retrace=False, use_legacy_exporter=True
    )
    options.update(overrides)
    destination = tmp_path / "unsupported.ep"
    before = graph_snapshot(source)
    with pytest.raises(region.RegionError, match="Saving annotated regions"):
        torch_tensorrt.save(source, str(destination), inputs=[x], **options)
    assert not destination.exists()
    assert graph_snapshot(source) == before


@pytest.mark.unit
@pytest.mark.parametrize(
    "corruption, message",
    [
        ("identity", "lost its child, target or route"),
        ("route", "lost its child, target or route"),
        ("bindings", "input boundary changed"),
        ("body", "reference body changed"),
    ],
)
def test_export_audits_region_clone_without_modifying_source(corruption, message):
    source = _captured_module(_TwoOutputs().eval(), torch.randn(2, 3))
    call = next(n for n in source.graph.nodes if n.meta.get(REGION_META_KEY))
    child = source.get_submodule(call.target)
    if corruption == "identity":
        child._torch_tensorrt_region_id = "wrong-region"
    elif corruption == "route":
        call.meta[REGION_META_KEY]["route"] = "tensorrt"
    elif corruption == "bindings":
        call.args = call.args[:-1]
    else:
        relu = next(
            n for n in child.graph.nodes if n.target == torch.ops.aten.relu.default
        )
        relu.target = torch.ops.aten.sigmoid.default
        child.recompile()

    before = graph_snapshot(source)
    with pytest.raises(region.RegionError, match=message):
        export(source, use_legacy_exporter=True)

    assert graph_snapshot(source) == before
