# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Guard region manifests against mutation and exercise CUDA graph execution."""

import operator
from contextlib import contextmanager

import pytest
import torch
from torch.utils._triton import has_triton

import torch_tensorrt
from torch_tensorrt import region
from torch_tensorrt.dynamo.region_compilation import normalize_region_scopes
from torch_tensorrt.dynamo.region_compilation._types import (
    RegionRecord,
    attach_region_records,
)
from torch_tensorrt.dynamo.region_compilation._validation import (
    validate_region_mutations,
)
from torch_tensorrt.dynamo.runtime._CudaGraphsTorchTensorRTModule import (
    CudaGraphsTorchTensorRTModule,
)
from torch_tensorrt.region._session import RegionCompilationSession
from torch_tensorrt.runtime import _cudagraphs

from ._utils import compile_regions


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
    "parent_work,route",
    [(False, None), (True, None), (False, "torch"), (False, "plugin")],
)
def test_global_partitioner_parent_work_selects_whole_graph_mode(
    monkeypatch, parent_work, route
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
    if route is not None:
        attach_region_records(
            compiled,
            [RegionRecord("region0", "region0", "_run_on_acc_0", route=route)],
        )
    monkeypatch.setattr(_cudagraphs, "CudaGraphsTorchTensorRTModule", CaptureStub)
    previous_mode = _cudagraphs._PY_RT_CUDAGRAPHS
    expected_whole_graph = parent_work or route == "torch"
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
    assert wrapped.cudagraph is None
    assert observed_inductor.capture_states and not any(
        observed_inductor.capture_states
    )
    assert observed_inductor.counter.frame_count >= 2


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
def test_region_cudagraph_preserves_input_strides_and_storage_offset():
    class StorageOrder(torch.nn.Module):
        def forward(self, x):
            with region.execute_in_torch():
                return x.as_strided((2, 2), (1, 2)).clone()

    storage = torch.arange(32, device="cuda", dtype=torch.float32)
    x = storage.as_strided((4, 4), (1, 4))
    model = StorageOrder().eval()
    compiled = compile_regions(model, [x])
    with torch.no_grad(), torch_tensorrt.runtime.enable_cudagraphs(compiled) as wrapped:
        torch.testing.assert_close(wrapped(x), model(x))
        graph, buffer = wrapped.cudagraph, wrapped._input_buffers[0]
        assert buffer.stride() == x.stride()
        storage.add_(100)
        torch.testing.assert_close(wrapped(x), model(x))
        assert wrapped.cudagraph is graph and wrapped._input_buffers[0] is buffer

        for stride, offset in (((4, 1), 0), ((4, 1), 5), ((0, 1), 5)):
            x = storage.as_strided((4, 4), stride, offset)
            torch.testing.assert_close(wrapped(x), model(x))
            assert wrapped.cudagraph is not graph
            graph = wrapped.cudagraph
            assert wrapped._input_buffers[0].stride() == stride
            assert wrapped._input_buffers[0].storage_offset() == offset


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_region_cudagraph_preserves_shared_storage_and_copies_hidden_values():
    class StorageViews(torch.nn.Module):
        def forward(self, x, y):
            with region.execute_in_torch():
                return x.as_strided((4,), (1,), storage_offset=12).clone() + y[:4]

    storage = torch.arange(16, device="cuda", dtype=torch.float32)
    x, y = storage[:8], storage[4:12]
    model = StorageViews().eval()
    # hints_wrapper disallows aliased capture inputs; runtime views can still alias.
    compiled = compile_regions(model, [x, storage.clone()[4:12]])
    with torch.no_grad(), torch_tensorrt.runtime.enable_cudagraphs(compiled) as wrapped:
        first = wrapped(x, y)
        torch.testing.assert_close(first, model(x, y))
        assert torch._C._is_alias_of(*wrapped._input_buffers)
        graph = wrapped.cudagraph
        # The model reads backing storage outside either visible input view.
        storage[12:].add_(100)
        torch.testing.assert_close(wrapped(x, y), model(x, y))
        assert wrapped.cudagraph is graph
        torch.testing.assert_close(model(x, y), first + 100)

        independent_y = storage.clone()[4:12]
        torch.testing.assert_close(wrapped(x, independent_y), model(x, independent_y))
        assert wrapped.cudagraph is not graph
        assert not torch._C._is_alias_of(*wrapped._input_buffers)


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
