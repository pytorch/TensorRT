# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Region serialization, including ExecuTorch programs with TensorRT delegates."""

import copy
import importlib.util
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
from torch_tensorrt.dynamo.region_compilation._types import (
    REGION_ID_META_KEY,
    REGION_META_KEY,
)

from ._utils import capture_regions as _captured_module
from ._utils import compile_regions, graph_snapshot


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
@pytest.mark.parametrize("copied", [False, True], ids=["source", "deepcopy"])
@pytest.mark.parametrize("nested", [False, True], ids=["direct", "fallback_child"])
def test_legacy_export_keeps_region_state_outputs_and_source(nested, copied):
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
    if copied:
        # A deep copy keeps each region leaf's identity.
        source = copy.deepcopy(source)

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
        child.meta[REGION_ID_META_KEY] = region_id
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
        {"output_format": "aot_inductor"},
        {"output_format": "executorch", "retrace": True},
        {"output_format": "executorch", "use_legacy_exporter": False},
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
    "overrides",
    [{}, {"retrace": False, "use_legacy_exporter": True}],
    ids=["defaults", "explicit"],
)
def test_save_accepts_executorch_for_regions(overrides, monkeypatch, tmp_path):
    import torch_tensorrt._compile as compile_module

    # Report ExecuTorch as missing, so every lane can check that the region gate lets
    # the save through to the ExecuTorch requirement instead of rejecting the format.
    monkeypatch.setattr(compile_module, "_has_executorch_exir", lambda: False)
    source = _captured_module(_TwoOutputs().eval(), torch.randn(2, 3))
    destination = tmp_path / "region.pte"
    with pytest.raises(ImportError, match="requires executorch.exir"):
        torch_tensorrt.save(
            source, str(destination), output_format="executorch", **overrides
        )
    assert not destination.exists()


def _assert_pte_matches(path, model, x):
    from executorch.runtime import Runtime

    method = Runtime.get().load_program(path).load_method("forward")
    expected = tuple(output.cpu() for output in model(x))
    torch.testing.assert_close(tuple(method.execute([x.cpu()])), expected)


@pytest.mark.unit
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_save_executorch_runs_region_program(device, tmp_path):
    pytest.importorskip("executorch.exir")
    # Checked at run time: remote GPU runners collect ExecuTorch lanes off the GPU host.
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    # ExecuTorch runs the region on its CPU kernels, so the .pte must hold the
    # parameter and buffer the region reads as host data whatever their device.
    model = _TwoOutputs().eval().to(device)
    x = torch.randn(2, 3, device=device)
    source = _captured_module(model, x)
    before = graph_snapshot(source)
    destination = tmp_path / "region.pte"
    torch_tensorrt.save(source, str(destination), output_format="executorch")

    _assert_pte_matches(destination, model, x)
    assert graph_snapshot(source) == before


@pytest.mark.unit
def test_save_executorch_delegates_engines_around_region(tmp_path):
    pytest.importorskip("executorch.exir")
    # Checked at run time: remote GPU runners collect ExecuTorch lanes off the GPU host.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to build TensorRT engines")
    from executorch.exir._serialize._program import deserialize_pte_binary

    # TensorRT builds one engine before the region and one after it.
    model, x = _TwoOutputs().eval().cuda(), torch.randn(2, 3, device="cuda")
    compiled = compile_regions(model, (x,))
    destination = tmp_path / "region.pte"
    torch_tensorrt.save(compiled, str(destination), output_format="executorch")

    # Each engine becomes a delegate, and the region's operators, which TensorRT
    # could convert, stay ExecuTorch kernels.
    plan = deserialize_pte_binary(destination.read_bytes()).program.execution_plan[0]
    assert [delegate.id for delegate in plan.delegates] == ["TensorRTBackend"] * 2
    operators = {operator.name for operator in plan.operators}
    assert {"aten::mm", "aten::relu", "aten::sigmoid"} <= operators

    if importlib.util.find_spec("torch_tensorrt_executorch_runtime") is not None:
        import torch_tensorrt_executorch_runtime  # noqa: F401

        _assert_pte_matches(destination, model, x)


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
        child.meta[REGION_ID_META_KEY] = "wrong-region"
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
