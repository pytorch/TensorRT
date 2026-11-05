# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Guard region manifests against mutation and exercise CUDA graph execution."""

import operator
from contextlib import contextmanager

import pytest
import torch
import torch_tensorrt
from torch.utils._triton import has_triton
from torch_tensorrt import region
from torch_tensorrt.dynamo.region_compilation import (
    get_region_records,
    normalize_region_scopes,
)
from torch_tensorrt.dynamo.region_compilation._types import (
    RECORDS_META_KEY,
    RegionRecord,
    attach_region_records,
)
from torch_tensorrt.dynamo.region_compilation._validation import (
    validate_region_mutations,
)
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
from torch_tensorrt.dynamo.runtime._CudaGraphsTorchTensorRTModule import (
    CudaGraphsTorchTensorRTModule,
)
from torch_tensorrt.region._session import RegionCompilationSession
from torch_tensorrt.runtime import _cudagraphs

from ._utils import compile_regions, save_and_load


class CaptureStub(torch.nn.Module):
    def __init__(self, compiled):
        super().__init__()
        self.compiled = compiled
        self.resets = 0

    def _reset_captured_graph(self):
        self.resets += 1


class MutationModel(torch.nn.Module):
    def __init__(
        self, target="buffer", placement="before", annotation="live", backend="eager"
    ):
        super().__init__()
        self.register_buffer("state", torch.zeros(2, 8))
        self.target = target
        self.placement = placement
        self.annotation = annotation
        self.backend = backend

    def forward(self, x):
        mutated = self.state if self.target == "buffer" else x
        if self.target == "temporary":
            mutated = x.clone()
        if self.placement == "before":
            mutated.add_(1)
        if self.annotation == "live":
            with region.execute_in_torch(backend=self.backend):
                result = torch.relu(x + mutated)
        else:
            if self.annotation == "empty":
                with region.execute_in_torch():
                    pass
            result = torch.relu(x + mutated)
        if self.placement == "after":
            mutated.add_(1)
        return result + mutated


def functional_export(model):
    with RegionCompilationSession(strict=True):
        exported = torch.export.export(model, (torch.randn(2, 8),), strict=True)
        return normalize_region_scopes(exported).run_decompositions()


@pytest.mark.parametrize("annotation,target", [("none", "buffer"), ("empty", "input")])
def test_no_live_region_preserves_existing_mutation_support(annotation, target):
    exported = functional_export(MutationModel(target=target, annotation=annotation))
    validate_region_mutations(exported)


@pytest.mark.parametrize("placement", ["before", "after"])
def test_local_temporary_mutation_remains_supported(placement):
    exported = functional_export(MutationModel("temporary", placement))
    validate_region_mutations(exported)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "target,placement,backend",
    [
        ("buffer", "before", "eager"),
        ("buffer", "after", "inductor"),
        ("input", "after", "eager"),
        ("input", "before", "inductor"),
    ],
)
def test_public_compile_rejects_mutation_before_buffer_lifting(
    monkeypatch, target, placement, backend
):
    from torch_tensorrt.dynamo import _compiler

    def unexpected_lifting(*args, **kwargs):
        raise AssertionError("Mutation reached buffer lifting before region validation")

    monkeypatch.setattr(_compiler, "lift_mutated_buffers", unexpected_lifting)
    model = MutationModel(target, placement, backend=backend).cuda().eval()
    with pytest.raises(region.RegionError, match=f"mutation of {target}"):
        torch_tensorrt.compile(
            model,
            inputs=[torch.randn(2, 8, device="cuda")],
            ir="dynamo",
            strict=True,
            offload_module_to_cpu=False,
        )


@pytest.mark.parametrize(
    "parent_work,route,serialized",
    [
        (False, None, False),
        (True, None, False),
        (False, "torch", False),
        (False, "plugin", False),
        (True, "torch", True),
        (True, "plugin", True),
    ],
)
def test_global_partitioner_whole_graph_mode_requires_torch_region(
    monkeypatch, parent_work, route, serialized
):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    result = graph.call_module("_run_on_acc_0", (x,))
    result = graph.call_function(operator.getitem, (result, 0))
    if parent_work:
        result = graph.call_function(torch.relu, (result,))
    graph.output(result)
    root = torch.nn.Module()
    root.add_module("_run_on_acc_0", torch.nn.Identity())
    compiled = torch.fx.GraphModule(root, graph)
    if serialized:
        compiled.meta["custom"] = {RECORDS_META_KEY: [{"route": route}]}
    elif route is not None:
        attach_region_records(
            compiled,
            [RegionRecord("region0", "region0", "_run_on_acc_0", route=route)],
        )
    monkeypatch.setattr(_cudagraphs, "CudaGraphsTorchTensorRTModule", CaptureStub)
    previous_mode = _cudagraphs._PY_RT_CUDAGRAPHS
    expected_whole_graph = route == "torch"
    with torch_tensorrt.runtime.enable_cudagraphs(compiled) as selected:
        if expected_whole_graph:
            assert isinstance(selected, CaptureStub)
            assert selected.compiled is compiled
            assert selected.resets == 0
        else:
            assert selected is compiled
        assert (
            torch_tensorrt.runtime.get_whole_cudagraphs_mode() == expected_whole_graph
        )
    assert _cudagraphs._PY_RT_CUDAGRAPHS == previous_mode
    if expected_whole_graph:
        assert selected.resets == 1


def test_failed_cudagraph_entry_restores_mode_and_runtime_override(monkeypatch):
    from torch_tensorrt.runtime import _runtime_config

    changes = []

    @contextmanager
    def runtime_config(*args, **kwargs):
        changes.append("enter")
        try:
            yield
        finally:
            changes.append("exit")

    def failed_wrapper(module):
        assert torch_tensorrt.runtime.get_whole_cudagraphs_mode()
        raise RuntimeError("warmup failed")

    monkeypatch.setattr(_runtime_config, "runtime_config", runtime_config)
    monkeypatch.setattr(_cudagraphs, "CudaGraphsTorchTensorRTModule", failed_wrapper)
    compiled = torch.fx.symbolic_trace(torch.nn.ReLU())
    compiled.meta["custom"] = {RECORDS_META_KEY: [{"route": "torch"}]}
    previous_mode = _cudagraphs._PY_RT_CUDAGRAPHS
    # Test the strategy override without requiring an engine or CUDA.
    context = _cudagraphs._CudagraphsContextManager(
        compiled, cuda_graph_strategy="whole_graph_capture"
    )
    with pytest.raises(RuntimeError, match="warmup failed"):
        with context:
            pytest.fail("A failed wrapper must not enter the user context")
    assert changes == ["enter", "exit"]
    assert _cudagraphs._PY_RT_CUDAGRAPHS == previous_mode
    monkeypatch.setattr(_cudagraphs, "CudaGraphsTorchTensorRTModule", CaptureStub)
    with context as retried:
        assert torch_tensorrt.runtime.get_whole_cudagraphs_mode()
    assert retried.resets == 1
    assert changes == ["enter", "exit", "enter", "exit"]
    assert _cudagraphs._PY_RT_CUDAGRAPHS == previous_mode


def test_cudagraph_context_restores_mode_active_when_entered(monkeypatch):
    monkeypatch.setattr(_cudagraphs, "CudaGraphsTorchTensorRTModule", CaptureStub)
    compiled = torch.fx.symbolic_trace(torch.nn.ReLU())
    compiled.meta["custom"] = {RECORDS_META_KEY: [{"route": "torch"}]}
    previous_mode = _cudagraphs._PY_RT_CUDAGRAPHS
    pending = torch_tensorrt.runtime.enable_cudagraphs(compiled)
    with torch_tensorrt.runtime.enable_cudagraphs(compiled):
        with pending:
            assert torch_tensorrt.runtime.get_whole_cudagraphs_mode()
        assert torch_tensorrt.runtime.get_whole_cudagraphs_mode()
    assert _cudagraphs._PY_RT_CUDAGRAPHS == previous_mode


def test_cudagraph_context_rejects_nested_self_entry_and_allows_reuse(monkeypatch):
    monkeypatch.setattr(_cudagraphs, "CudaGraphsTorchTensorRTModule", CaptureStub)
    compiled = torch.fx.symbolic_trace(torch.nn.ReLU())
    compiled.meta["custom"] = {RECORDS_META_KEY: [{"route": "torch"}]}
    previous_mode = _cudagraphs._PY_RT_CUDAGRAPHS
    context = torch_tensorrt.runtime.enable_cudagraphs(compiled)
    with context as outer:
        with pytest.raises(RuntimeError, match="cannot re-enter itself"):
            with context:
                pytest.fail("The same context cannot be active twice")
        assert torch_tensorrt.runtime.get_whole_cudagraphs_mode()
        assert context.cudagraphs_module is outer
        assert outer.resets == 0
    assert outer.resets == 1
    assert _cudagraphs._PY_RT_CUDAGRAPHS == previous_mode
    with context as reused:
        assert reused is not outer
        assert torch_tensorrt.runtime.get_whole_cudagraphs_mode()
    assert reused.resets == 1
    assert _cudagraphs._PY_RT_CUDAGRAPHS == previous_mode


class InductorRegion(torch.nn.Module):
    def forward(self, x):
        with region.execute_in_torch(backend="inductor"):
            return torch.relu(x + 0.25) * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_eager_region_cudagraph_recaptures_for_autocast_and_dtype():
    class Matmul(torch.nn.Module):
        def forward(self, x, y):
            with region.execute_in_torch():
                return x @ y

    x, y = (torch.randn(8, 8, device="cuda") for _ in range(2))
    model = Matmul().eval()
    compiled = compile_regions(model, [x, y])
    executions = []

    def observe(module, args):
        executions.append(
            (
                torch.cuda.is_current_stream_capturing(),
                torch.is_autocast_enabled("cuda"),
                torch.cuda.current_stream().cuda_stream,
            )
        )

    hook = compiled.register_forward_pre_hook(observe)
    try:
        with (
            torch.no_grad(),
            torch_tensorrt.runtime.enable_cudagraphs(compiled) as wrapped,
        ):
            assert not executions
            previous_graph = None
            for dtype, autocast in (
                (torch.float32, False),
                (torch.float32, True),
                (torch.float32, False),
                (torch.float64, False),
            ):
                value, weight = x.to(dtype), y.to(dtype)
                with torch.autocast("cuda", enabled=autocast, dtype=torch.float16):
                    start = len(executions)
                    result = wrapped(value, weight)
                    torch.testing.assert_close(result, model(value, weight))
                    assert result.dtype == (torch.float16 if autocast else dtype)
                    assert wrapped.cudagraph is not previous_graph
                    previous_graph = wrapped.cudagraph
                    calls = executions[start:]
                    # Warmup must use the capture stream and autocast context.
                    assert len(calls) > 1 and calls[-1][0]
                    assert not any(capturing for capturing, _, _ in calls[:-1])
                    assert all(enabled == autocast for _, enabled, _ in calls)
                    assert all(
                        stream == wrapped._engine_stream.cuda_stream
                        for _, _, stream in calls
                    )
                    count = len(executions)
                    torch.testing.assert_close(
                        wrapped(value + 1, weight), model(value + 1, weight)
                    )
                    assert wrapped.cudagraph is previous_graph
                    assert len(executions) == count
    finally:
        hook.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_eager_region_cudagraph_replays_autocast_casts_across_scopes():
    class LinearRegion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, x):
            with region.execute_in_torch():
                return self.linear(x)

    x = torch.randn(2, 8, device="cuda")
    model = LinearRegion().cuda().eval()
    compiled = compile_regions(model, [x])
    with (
        torch.no_grad(),
        torch_tensorrt.runtime.enable_cudagraphs(compiled) as wrapped,
    ):
        graph = None
        for step in range(3):
            # Autocast caches casts of leaves that require grad until its scope exits.
            with torch.autocast("cuda", dtype=torch.float16):
                value = (x + step).requires_grad_()
                torch.testing.assert_close(wrapped(value), model(value))
                assert torch.is_autocast_cache_enabled()
            assert graph is None or wrapped.cudagraph is graph
            graph = wrapped.cudagraph
            # The region shares these parameters, so replays must cast their
            # current values rather than a cached copy freed with the scope.
            model.linear.weight.add_(1)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton(),
    reason="CUDA Inductor requires Triton support on the device",
)
def test_inductor_cudagraph_warms_up_actual_context_and_stream(observed_inductor):
    x = torch.randn(2, 8, device="cuda")
    model = InductorRegion().eval()
    compiled = compile_regions(model, [x])
    # Compile lazily in the caller's context, before stream capture.
    with torch_tensorrt.runtime.enable_cudagraphs(compiled) as wrapped:
        assert isinstance(wrapped, CudaGraphsTorchTensorRTModule)
        assert observed_inductor.counter.frame_count == 0
        with torch.no_grad():
            torch.testing.assert_close(wrapped(x), model(x))
            graph = wrapped.cudagraph
            torch.testing.assert_close(wrapped(x + 1), model(x + 1))
            assert wrapped.cudagraph is graph
        with torch.inference_mode():
            torch.testing.assert_close(wrapped(x), model(x))
            assert wrapped.cudagraph is not graph
            graph = wrapped.cudagraph
            caller = torch.cuda.current_stream()
            stream = torch.cuda.Stream()
            stream.wait_stream(caller)
            with torch.cuda.stream(stream):
                torch.testing.assert_close(wrapped(x + 2), model(x + 2))
                assert wrapped.cudagraph is not graph
            caller.wait_stream(stream)
            torch.testing.assert_close(wrapped(x + 3), model(x + 3))
            graph = wrapped.cudagraph
            result = wrapped(x.double())
            torch.testing.assert_close(result, model(x.double()))
            assert result.dtype == torch.float64
            assert wrapped.cudagraph is not graph
        # Warmup must compile with the autocast cache flag that capture uses.
        with torch.autocast("cuda", dtype=torch.float16):
            torch.testing.assert_close(wrapped(x), model(x))
    assert wrapped.cudagraph is None
    assert observed_inductor.capture_states and not any(
        observed_inductor.capture_states
    )
    frames = observed_inductor.counter.frame_count
    assert frames >= 2
    # Without autocast the cache flag is untouched, so plain calls reuse the region.
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), model(x))
    assert observed_inductor.counter.frame_count == frames


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("nested", [True, False], ids=["nested", "single_list"])
def test_region_cudagraph_preserves_output_pytree_and_previous_results(nested):
    class StructuredOutputs(torch.nn.Module):
        def forward(self, x):
            with region.execute_in_torch():
                activated = torch.relu(x)
                gated = torch.sigmoid(x)
            if nested:
                return {"pair": (activated, [gated]), "constant": 7}
            return [activated]

    x = torch.randn(2, 8, device="cuda")
    model = StructuredOutputs().eval()
    compiled = compile_regions(model, [x])
    with torch.no_grad(), torch_tensorrt.runtime.enable_cudagraphs(compiled) as wrapped:
        first = wrapped(x)
        expected = model(x)
        torch.testing.assert_close(first, expected)
        captured = wrapped.cudagraph
        torch.testing.assert_close(wrapped(x + 1), model(x + 1))
        assert wrapped.cudagraph is captured
        # Later replay must not overwrite tensors already returned to callers.
        torch.testing.assert_close(first, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_fast_partitioner", [True, False], ids=["fast", "global"])
@pytest.mark.parametrize(
    "backend,serialized",
    [("eager", False), ("eager", True), ("inductor", False)],
    ids=["eager-compiled", "eager-loaded", "inductor-compiled"],
)
def test_region_cudagraph_captures_tensorrt_engines_around_region(
    tmp_path, backend, serialized, use_fast_partitioner
):
    if backend == "inductor" and not has_triton():
        pytest.skip("CUDA Inductor requires Triton support on the device")

    class EngineRegionEngine(torch.nn.Module):
        def forward(self, x):
            x = x * 3 + 1
            with region.execute_in_torch(backend=backend):
                x = torch.relu(x) * 2
            return x - 1

    storage = torch.randn(16, device="cuda")
    x = storage.view(8, 2).t()
    model = EngineRegionEngine().eval()
    compiled = compile_regions(model, [x], use_fast_partitioner=use_fast_partitioner)
    assert sum(isinstance(m, TorchTensorRTModule) for m in compiled.modules()) == 2
    if serialized:
        compiled = save_and_load(compiled, tmp_path)
        assert sum("execute_engine" in str(n.target) for n in compiled.graph.nodes) == 2
    with torch_tensorrt.runtime.enable_cudagraphs(compiled) as wrapped:
        with torch.no_grad():
            first, expected = wrapped(x), model(x)
            torch.testing.assert_close(first, expected)
            # The first engine reads the transposed input through a static view.
            assert wrapped._input_buffers[0].stride() == x.stride()
            graph = wrapped.cudagraph
            storage.add_(1)
            torch.testing.assert_close(wrapped(x), model(x))
            assert wrapped.cudagraph is graph
            torch.testing.assert_close(first, expected)
        with torch.inference_mode():
            torch.testing.assert_close(wrapped(x), model(x))
            assert wrapped.cudagraph is not graph
            graph = wrapped.cudagraph
            caller, stream = torch.cuda.current_stream(), torch.cuda.Stream()
            stream.wait_stream(caller)
            with torch.cuda.stream(stream):
                torch.testing.assert_close(wrapped(x), model(x))
                assert wrapped.cudagraph is not graph
            caller.wait_stream(stream)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_unannotated_global_fallback_cudagraph_preserves_input_layout():
    class StorageOrder(torch.nn.Module):
        def forward(self, x):
            return x.as_strided((2, 2), (1, 2)).clone()

    storage = torch.arange(32, device="cuda", dtype=torch.float32)
    x = storage.as_strided((4, 4), (1, 4))
    model = StorageOrder().eval()
    compiled = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=[x],
        min_block_size=1,
        use_fast_partitioner=False,
        offload_module_to_cpu=False,
        torch_executed_ops={torch.ops.aten.as_strided.default},
    )
    assert not get_region_records(compiled)
    torch.testing.assert_close(compiled(x), model(x))
    with torch.no_grad(), torch_tensorrt.runtime.enable_cudagraphs(compiled) as wrapped:
        torch.testing.assert_close(wrapped(x), model(x))
        storage.add_(100)
        torch.testing.assert_close(wrapped(x), model(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("serialized", [False, True], ids=["compiled", "loaded"])
def test_region_cudagraph_preserves_input_strides_across_storage_offsets(
    tmp_path, serialized
):
    class StorageOrder(torch.nn.Module):
        def forward(self, x):
            with region.execute_in_torch():
                return x.as_strided((2, 2), (1, 2)).clone()

    storage = torch.arange(32, device="cuda", dtype=torch.float32)
    x = storage.as_strided((4, 4), (1, 4))
    model = StorageOrder().eval()
    compiled = compile_regions(model, [x])
    if serialized:
        compiled = save_and_load(compiled, tmp_path)
        assert not get_region_records(compiled)
        torch.testing.assert_close(compiled(x), model(x))
    with torch.no_grad(), torch_tensorrt.runtime.enable_cudagraphs(compiled) as wrapped:
        torch.testing.assert_close(wrapped(x), model(x))
        graph, buffer = wrapped.cudagraph, wrapped._input_buffers[0]
        assert buffer.stride() == x.stride()
        storage.add_(100)
        torch.testing.assert_close(wrapped(x), model(x))
        assert wrapped.cudagraph is graph and wrapped._input_buffers[0] is buffer

        # New strides need a new capture; a new storage offset alone does not.
        for stride, offset, recaptured in (
            ((4, 1), 0, True),
            ((4, 1), 5, False),
            ((0, 1), 5, True),
        ):
            x = storage.as_strided((4, 4), stride, offset)
            torch.testing.assert_close(wrapped(x), model(x))
            assert (wrapped.cudagraph is not graph) == recaptured
            graph = wrapped.cudagraph
            assert wrapped._input_buffers[0].stride() == stride


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_region_cudagraph_copies_only_the_bytes_inputs_reach():
    class ReadsPastView(torch.nn.Module):
        def forward(self, x, y):
            with region.execute_in_torch():
                return x.as_strided((12,), (1,))[8:].clone() + y[:4]

    storage = torch.arange(16, device="cuda", dtype=torch.float32)
    x, y = storage[:8], storage[4:12]
    model = ReadsPastView().eval()
    compiled = compile_regions(model, [x, y])
    torch.testing.assert_close(compiled(x, y), model(x, y))
    with torch.no_grad(), torch_tensorrt.runtime.enable_cudagraphs(compiled) as wrapped:
        # y overlaps x, so their shared copy holds the bytes x reads past its view.
        torch.testing.assert_close(wrapped(x, y), model(x, y))
        # x's own copy ends at its last element, so reading past it fails.
        with pytest.raises(RuntimeError, match="out of bounds"):
            wrapped(x, y.clone())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_region_cudagraph_replays_slices_of_larger_tensors():
    class Rows(torch.nn.Module):
        def forward(self, x, y):
            with region.execute_in_torch():
                return torch.relu(x) + y

    batch = torch.randn(64, 8, device="cuda")
    other = torch.randn(9, 8, device="cuda")
    model = Rows().eval()
    compiled = compile_regions(model, [batch[:2].clone(), batch[2:4].clone()])
    with torch.no_grad(), torch_tensorrt.runtime.enable_cudagraphs(compiled) as wrapped:
        torch.testing.assert_close(
            wrapped(batch[:2], batch[2:4]), model(batch[:2], batch[2:4])
        )
        graph = wrapped.cudagraph
        # Offsets, the distance between disjoint inputs, and whether they share
        # storage vary; shapes and strides do not.
        pairs = [(batch[i : i + 2], batch[j : j + 2]) for i, j in ((60, 3), (7, 9))]
        for x, y in pairs + [(other[3:5], batch[40:42])]:
            torch.testing.assert_close(wrapped(x, y), model(x, y))
        assert wrapped.cudagraph is graph
        # Each input copies only its own two rows.
        assert [b.numel() for b in wrapped._input_storage_buffers] == [2 * 8 * 4] * 2

        # Inputs that overlap, such as interleaved chunks, share one copy of
        # the bytes they reach, with the distance between them in the layout.
        wide = batch.view(32, 16)
        for x, y in [
            (batch[7:9], batch[8:10]),
            (batch[5:7],) * 2,
            wide[1:3].chunk(2, -1),
        ]:
            torch.testing.assert_close(wrapped(x, y), model(x, y))
            assert torch._C._is_alias_of(*wrapped._input_buffers)
        assert [b.numel() for b in wrapped._input_storage_buffers] == [2 * 16 * 4]
        # The copy keeps the inputs' alignment, which kernels can depend on.
        alignments = [t.data_ptr() % 256 for t in (x, y, *wrapped._input_buffers)]
        assert alignments[:2] == alignments[2:]
        graph = wrapped.cudagraph
        x, y = wide[30:].chunk(2, -1)
        torch.testing.assert_close(wrapped(x, y), model(x, y))
        assert wrapped.cudagraph is graph


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("annotated", [True, False], ids=["region", "global_fallback"])
def test_cudagraph_binds_keyword_inputs_by_name(annotated):
    class KeywordInputs(torch.nn.Module):
        def forward(self, x, *, scale, bias, extra):
            if annotated:
                with region.execute_in_torch():
                    return (x * scale + bias) * extra["gain"] - extra["shift"]
            return (x * scale + bias) * extra["gain"] - extra["shift"]

    def inputs():
        x, scale, bias, gain, shift = (torch.randn(2, 8).cuda() for _ in range(5))
        extra = {"gain": gain, "shift": shift}
        return x, {"scale": scale, "bias": bias, "extra": extra}

    def reverse(kwargs):
        # The same keywords and dict keys, inserted in the opposite order.
        return {
            key: dict(reversed(value.items())) if isinstance(value, dict) else value
            for key, value in reversed(kwargs.items())
        }

    model = KeywordInputs().eval()
    fallback = {} if annotated else {"torch_executed_ops": {torch.ops.aten.mul.Tensor}}
    x, kwargs = inputs()
    compiled = compile_regions(model, [x], kwarg_inputs=kwargs, **fallback)
    with torch.no_grad(), torch_tensorrt.runtime.enable_cudagraphs(compiled) as wrapped:
        assert isinstance(wrapped, CudaGraphsTorchTensorRTModule)
        x, kwargs = inputs()
        torch.testing.assert_close(wrapped(x, **reverse(kwargs)), model(x, **kwargs))
        captured = wrapped.cudagraph
        for call in (
            lambda x, kw: wrapped(x, **kw),
            lambda x, kw: wrapped(x, **reverse(kw)),
            # Inputs bind by position or by keyword, as for compiled(...).
            lambda x, kw: wrapped(x, kw["scale"], extra=kw["extra"], bias=kw["bias"]),
            lambda x, kw: wrapped(**reverse(kw), x=x),
        ):
            x, kwargs = inputs()
            torch.testing.assert_close(call(x, kwargs), model(x, **kwargs))
            assert wrapped.cudagraph is captured


@torch.library.custom_op("ttrt_region_runtime_test::noise", mutates_args=())
def _region_noise(x: torch.Tensor) -> torch.Tensor:
    return x + torch.rand_like(x)


@_region_noise.register_fake
def _region_noise_fake(x):
    return torch.empty_like(x)


class RandomRegion(torch.nn.Module):
    def forward(self, x):
        with region.fused_op(torch.ops.ttrt_region_runtime_test.noise.default):
            return x + torch.rand_like(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_op_cudagraph_preserves_rng_sequence_on_capture_replay_and_recapture():
    x = torch.zeros(2, 8, device="cuda")
    model = RandomRegion().eval()
    compiled = compile_regions(model, [x])
    with torch.random.fork_rng(devices=[x.device]):
        torch.manual_seed(2026)
        expected, states = [], []
        for step in range(4):
            expected.append(model(x + step))
            states.append(torch.cuda.get_rng_state(x.device))
        torch.manual_seed(2026)
        with torch_tensorrt.runtime.enable_cudagraphs(compiled) as wrapped:
            for step in range(4):
                with torch.inference_mode(step >= 2), torch.no_grad():
                    previous_graph = wrapped.cudagraph
                    torch.testing.assert_close(wrapped(x + step), expected[step])
                    assert torch.equal(torch.cuda.get_rng_state(x.device), states[step])
                    if step in (0, 2):
                        assert wrapped.cudagraph is not previous_graph
                    else:
                        assert wrapped.cudagraph is previous_graph


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_failed_region_cudagraph_warmup_restores_rng_state():
    x = torch.zeros(2, 8, device="cuda")
    compiled = compile_regions(RandomRegion().eval(), [x])

    def fail_after_random_draw(module, args, output):
        torch.rand(())
        raise RuntimeError("random warmup failed")

    with torch.random.fork_rng(devices=[x.device]):
        torch.manual_seed(2026)
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(x.device)
        hook = compiled.register_forward_hook(fail_after_random_draw)
        try:
            with torch_tensorrt.runtime.enable_cudagraphs(compiled) as wrapped:
                with pytest.raises(RuntimeError, match="random warmup failed"):
                    wrapped(x)
                assert wrapped.cudagraph is None
                assert torch.equal(torch.random.get_rng_state(), cpu_state)
                assert torch.equal(torch.cuda.get_rng_state(x.device), cuda_state)
        finally:
            hook.remove()
