# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify region backend selection, real Inductor execution, and TRT placement."""

import pytest
import torch
from torch.utils._triton import has_triton

from torch_tensorrt import region
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_manager import DynamoPassManager
from torch_tensorrt.dynamo.region_compilation import (
    audit_region_placement,
    configure_region_backends,
    get_region_records,
    is_region_hop,
    materialize_regions,
    normalize_region_scopes,
    region_hop_records,
)
from torch_tensorrt.dynamo.region_compilation._types import REGION_META_KEY
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
from torch_tensorrt.region._constants import BACKEND_KEY
from torch_tensorrt.region._session import RegionCompilationSession

from ._utils import capture_regions as capture
from ._utils import compile_regions


class RegionModel(torch.nn.Module):
    def __init__(self, backend="inductor", only_region=False):
        super().__init__()
        self.backend = backend
        self.only_region = only_region

    def forward(self, x):
        before = x if self.only_region else x + 1
        with region.execute_in_torch(name="activation", backend=self.backend):
            result = torch.relu(before + 0.25) * 2
        return result if self.only_region else result - 1


@pytest.mark.parametrize("backend,error", [(42, TypeError), ("unknown", ValueError)])
def test_invalid_backend_is_rejected(backend, error):
    with pytest.raises(error, match="backend"):
        with region.execute_in_torch(backend=backend):
            pass


@pytest.mark.parametrize("strict", [False, True])
def test_capture_checks_precede_kernel_factory(strict):
    calls = []

    def unexpected_factory():
        calls.append(True)
        raise AssertionError("Kernel factory ran before capture validation")

    class Model(torch.nn.Module):
        def forward(self, x):
            with region.fused_kernel(kernel=unexpected_factory):
                return x.sin()

    with RegionCompilationSession(strict=strict) as session:
        if strict:
            session.capture_error = "Missing marker capabilities"
        message = "Missing marker capabilities" if strict else "strict=True"
        with pytest.raises(Exception, match=message):
            torch.export.export(Model(), (torch.ones(2),), strict=strict)
    assert not calls


def test_default_and_unowned_annotations_do_not_compile(monkeypatch):
    def unexpected_compile(*args, **kwargs):
        raise AssertionError("Annotation unexpectedly invoked torch.compile")

    monkeypatch.setattr(torch, "compile", unexpected_compile)
    x = torch.randn(2, 8)
    with region.execute_in_torch():
        result = torch.relu(x)
    torch.testing.assert_close(result, torch.relu(x))
    model = RegionModel().eval()
    exported = torch.export.export(model, (x,), strict=True)
    assert not get_region_records(exported.module())
    torch.testing.assert_close(exported.module()(x), model(x))
    eager_graph = capture(RegionModel(backend="eager"), x)
    configure_region_backends(eager_graph)
    record = get_region_records(eager_graph)[0]
    assert record.backend == "eager"
    assert eager_graph.get_submodule(record.child_target)._compiled_call_impl is None


def test_backend_metadata_survives_capture_with_repeated_names():
    class Mixed(torch.nn.Module):
        def forward(self, x):
            with region.execute_in_torch(name="shared"):
                first = torch.relu(x)
            with region.execute_in_torch(name="shared", backend="inductor"):
                second = torch.sigmoid(first)
            return second

    x = torch.randn(2, 8)
    with RegionCompilationSession(strict=True):
        exported = normalize_region_scopes(
            torch.export.export(Mixed(), (x,), strict=True)
        )
        assert [r.backend for r in region_hop_records(exported.graph_module)] == [
            "eager",
            "inductor",
        ]
        graph = materialize_regions(exported.run_decompositions().module())
    records = get_region_records(graph)
    assert [r.backend for r in records] == ["eager", "inductor"]
    assert len({r.id for r in records}) == 2
    assert {r.name for r in records} == {"shared"}
    assert [
        node.meta[REGION_META_KEY]["backend"]
        for node in graph.graph.nodes
        if REGION_META_KEY in node.meta
    ] == ["eager", "inductor"]
    audit_region_placement(graph, records)


@pytest.mark.parametrize("stage", ["capture", "hop", "call"])
def test_backend_metadata_loss_or_change_is_rejected(stage):
    with RegionCompilationSession(strict=True):
        exported = torch.export.export(RegionModel(), (torch.randn(2, 8),), strict=True)
        if stage == "capture":
            member = next(
                node
                for node in exported.graph.nodes
                if node.target == torch.ops.aten.relu.default
            )
            del member.meta["custom"][BACKEND_KEY]
            with pytest.raises(region.RegionError, match="membership"):
                normalize_region_scopes(exported)
            return
        exported = normalize_region_scopes(exported)
        if stage == "hop":

            def corrupt_backend(graph, settings):
                hop = next(node for node in graph.graph.nodes if is_region_hop(node))
                hints = dict(hop.kwargs["hints"], backend="eager")
                hop.kwargs = dict(hop.kwargs, hints=hints)
                return graph

            with pytest.raises(region.RegionError, match="backend"):
                DynamoPassManager([corrupt_backend])(
                    exported.graph_module, CompilationSettings()
                )
            return
        graph = materialize_regions(exported.run_decompositions().module())
    call = next(node for node in graph.graph.nodes if REGION_META_KEY in node.meta)
    del call.meta[REGION_META_KEY]["backend"]
    with pytest.raises(region.RegionError, match="backend"):
        configure_region_backends(graph)


def test_real_inductor_is_lazy_reused_and_keeps_graph(observed_inductor):
    x = torch.randn(2, 8)
    model = RegionModel().eval()
    graph = capture(model, x)
    record = get_region_records(graph)[0]
    child = graph.get_submodule(record.child_target)
    before = str(child.graph)
    assert child._compiled_call_impl is None
    configure_region_backends(graph)
    configure_region_backends(graph)
    assert len(observed_inductor.requests) == 1
    assert observed_inductor.counter.frame_count == 0
    assert observed_inductor.executions == 0
    with torch.no_grad():
        for value in (x, x + 1):
            torch.testing.assert_close(graph(value), model(value))
    assert observed_inductor.counter.frame_count == 1
    assert observed_inductor.executions == 2
    assert isinstance(child, torch.fx.GraphModule)
    assert str(child.graph) == before
    audit_region_placement(graph, [record])


@pytest.mark.parametrize("suppress_errors", [False, True])
def test_backend_failure_propagates_without_eager_fallback(
    monkeypatch, suppress_errors
):
    torch._dynamo.reset()
    real_compile = torch.compile

    def broken_backend(graph, inputs):
        raise RuntimeError("region backend deliberately failed")

    def compile_spy(model, **kwargs):
        kwargs.pop("options")
        kwargs["backend"] = broken_backend
        return real_compile(model, **kwargs)

    x = torch.randn(2, 8)
    graph = capture(RegionModel(), x)
    monkeypatch.setattr(torch, "compile", compile_spy)
    configure_region_backends(graph)
    try:
        with (
            torch._dynamo.config.patch(suppress_errors=suppress_errors),
            pytest.raises(RuntimeError, match="region backend deliberately failed"),
        ):
            graph(x)
    finally:
        torch._dynamo.reset()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton(),
    reason="CUDA Inductor requires Triton support on the device",
)
@pytest.mark.parametrize("use_fast_partitioner", [True, False], ids=["fast", "global"])
def test_cuda_inductor_region_between_engines(observed_inductor, use_fast_partitioner):
    x = torch.randn(3, 257, device="cuda")
    model = RegionModel().cuda().eval()
    compiled = compile_regions(
        model, [x], min_block_size=1, use_fast_partitioner=use_fast_partitioner
    )
    assert sum(isinstance(m, TorchTensorRTModule) for m in compiled.modules()) == 2
    records = get_region_records(compiled)
    assert len(records) == 1 and records[0].backend == "inductor"
    assert observed_inductor.counter.frame_count == 0
    with torch.no_grad():
        for value in (x, x + 1):
            torch.testing.assert_close(compiled(value), model(value))
    assert observed_inductor.counter.frame_count == 1
    assert observed_inductor.executions == 2
    audit_region_placement(compiled, records)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton(),
    reason="CUDA Inductor requires Triton support on the device",
)
@pytest.mark.parametrize("only_region,min_block_size", [(True, 1), (False, 100)])
def test_public_compile_installs_backend_without_engines(
    observed_inductor, only_region, min_block_size
):
    x = torch.randn(2, 8, device="cuda")
    model = RegionModel(only_region=only_region).cuda().eval()
    compiled = compile_regions(model, [x], min_block_size=min_block_size)
    assert not any(isinstance(m, TorchTensorRTModule) for m in compiled.modules())
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), model(x))
    assert observed_inductor.counter.frame_count == 1
    assert observed_inductor.executions == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dryrun_does_not_install_backend(observed_inductor):
    x = torch.randn(2, 8, device="cuda")
    model = RegionModel().cuda().eval()
    compiled = compile_regions(model, [x], min_block_size=1, dryrun=True)
    assert not observed_inductor.requests
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), model(x))
    assert observed_inductor.counter.frame_count == 0
    assert observed_inductor.executions == 0
