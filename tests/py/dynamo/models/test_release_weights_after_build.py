# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import weakref

import pytest
import torch
import torch_tensorrt
from torch_tensorrt.dynamo import _compiler
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo._weight_release import (
    WeightReleaser,
    release_storage,
    weight_storages,
)
from torch_tensorrt.dynamo.conversion import _conversion


def test_release_weights_after_build_defaults_to_mapped_weights_only():
    assert CompilationSettings().release_weights_after_build is None


def test_old_pickle_without_field_defaults_to_none():
    state = CompilationSettings().__dict__.copy()
    state.pop("release_weights_after_build", None)
    restored = CompilationSettings.__new__(CompilationSettings)
    restored.__setstate__(state)
    assert restored.release_weights_after_build is None


class _Linear(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return x @ self.weight


def _traced(weight: torch.nn.Parameter) -> torch.fx.GraphModule:
    return torch.fx.symbolic_trace(_Linear(weight))


def _param(n: int = 4) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.randn(n, n), requires_grad=False)


def test_releaser_releases_after_the_last_reader():
    own, shared, torch_side = _param(), _param(), _param()
    first, second = _traced(own), _traced(shared)
    also_shared, in_torch = _traced(shared), _traced(torch_side)
    releaser = WeightReleaser(torch.nn.ParameterList([own, shared, torch_side]))
    releaser.count_readers([first, second, also_shared, in_torch])

    releaser.release_storages_after_build(weight_storages(first), "first")
    # Its tensor is emptied, so using it raises a shape error instead of reading freed memory.
    assert own.shape == (0,)
    assert shared.untyped_storage().nbytes() > 0

    releaser.release_storages_after_build(weight_storages(second), "second")
    assert shared.untyped_storage().nbytes() > 0  # also_shared still reads it
    releaser.release_storages_after_build(weight_storages(also_shared), "also_shared")
    assert shared.shape == (0,)

    # in_torch runs in PyTorch: it is counted but never built, so its weight stays.
    assert torch_side.untyped_storage().nbytes() > 0


def test_storage_rebound_after_counting_is_kept():
    shared = _param()
    first, second = _traced(shared), _traced(shared)
    releaser = WeightReleaser(torch.nn.ParameterList([shared]))
    releaser.count_readers([first, second])
    read = weight_storages(first)
    shared.data = shared.data.clone()  # e.g. moved between devices after partitioning
    releaser.release_storages_after_build(read, "first")
    assert (
        shared.untyped_storage().nbytes() > 0
    )  # unknown to the releaser: second may still read it


def test_releaser_releases_unread_parameters_only():
    used, unread = _param(), _param()
    holder = torch.nn.Module()
    holder.used, holder.unread = used, unread
    holder.register_buffer("buf", torch.randn(4))
    releaser = WeightReleaser(holder)
    releaser.count_readers([_traced(used)])
    unread_bytes = unread.numel() * unread.element_size()
    assert releaser.release_unread_storages() == unread_bytes
    assert unread.shape == (0,)
    assert used.untyped_storage().nbytes() > 0
    assert holder.buf.untyped_storage().nbytes() > 0  # buffers are never candidates


def test_tensor_the_module_does_not_own_is_kept():
    weight, constant = _param(), torch.randn(4, 4)
    reader = _traced(weight)
    # A partition registers a lifted constant it reads as its buffer.
    reader.register_buffer("constant", constant)
    releaser = WeightReleaser(torch.nn.ParameterList([weight]))
    releaser.count_readers([reader])
    releaser.release_storages_after_build(set(weight_storages(reader)), "reader")
    assert weight.shape == (0,)
    assert constant.untyped_storage().nbytes() > 0


def test_non_resizable_storage_is_skipped():
    weight = torch.frombuffer(bytearray(64), dtype=torch.float32)
    assert release_storage(weight.untyped_storage()) == 0
    assert weight.untyped_storage().nbytes() == 64


def test_sparse_tensor_reads_the_storages_of_its_indices_and_values():
    values = torch.randn(4)
    root = torch.nn.Module()
    # Built from a dense tensor without copying it, as from a dense weight.
    indices = torch.arange(4).repeat(2, 1)
    root.register_buffer(
        "s", torch.sparse_coo_tensor(indices, values, (4, 4), check_invariants=True)
    )
    graph = torch.fx.Graph()
    x, s = graph.placeholder("x"), graph.get_attr("s")
    graph.output(graph.call_function(torch.add, (x, s)))
    read = weight_storages(torch.fx.GraphModule(root, graph))
    assert values.untyped_storage().data_ptr() in read
    assert root.s._indices().untyped_storage().data_ptr() in read


def test_subgraph_reads_are_counted():
    weight = _param()
    root = torch.nn.Module()
    root.branch = _traced(weight)  # fetched by a get_attr, as a torch.cond branch
    graph = torch.fx.Graph()
    graph.output(graph.get_attr("branch"))
    assert weight.untyped_storage().data_ptr() in weight_storages(
        torch.fx.GraphModule(root, graph)
    )


class _TwoEngines(torch.nn.Module):
    """Two TensorRT partitions split by ops forced to PyTorch, a weight shared by both, one read only
    by the PyTorch partition, and one the forward never reads."""

    def __init__(self):
        super().__init__()
        p = lambda *s: torch.nn.Parameter(
            torch.randn(*s), requires_grad=False
        )  # noqa: E731
        self.w_a, self.w_b, self.w_shared = p(16, 16), p(16, 16), p(16)
        self.w_torch, self.unused_head = p(16), p(16, 1000)
        self.register_buffer("bias", torch.randn(16))

    def forward(self, x):
        y = x @ self.w_a + self.w_shared + self.bias
        y = torch.sin(y * self.w_torch)  # aten.mul / aten.sin run in PyTorch
        return y @ self.w_b + self.w_shared


_TORCH_EXECUTED_OPS = {torch.ops.aten.sin.default, torch.ops.aten.mul.Tensor}


@pytest.mark.parametrize("use_fast_partitioner", [True, False])
@pytest.mark.parametrize("release", [True, None, False])
def test_compile_releases_weights(release, use_fast_partitioner):
    torch.manual_seed(0)
    model = _TwoEngines().eval().cuda()
    x = torch.randn(4, 16, device="cuda")
    with torch.no_grad():
        expected = model(x)
    ep = torch.export.export(model, (x,))
    trt_gm = torch_tensorrt.dynamo.compile(
        ep,
        arg_inputs=[x],
        min_block_size=1,
        torch_executed_ops=_TORCH_EXECUTED_OPS,
        use_fast_partitioner=use_fast_partitioner,
        release_weights_after_build=release,
    )
    with torch.no_grad():
        for _ in range(2):  # TF32 matmuls in the engines
            torch.testing.assert_close(trt_gm(x), expected, rtol=1e-2, atol=1e-2)

    state = ep.state_dict
    # A read-only buffer read by an engine is a weight like any other: released after its build.
    # None (the default) releases only mapped weights, and these aren't mapped.
    for name in ("w_a", "w_b", "w_shared", "unused_head", "bias"):
        assert (state[name].numel() == 0) == (release is True), name
    assert state["w_torch"].numel() > 0  # read by the PyTorch partition
    # Released tensors are emptied, not left over freed memory: copying the model and the compiled
    # module reads none.
    copy.deepcopy(model)
    copy.deepcopy(trt_gm)


def test_weights_are_released_before_the_next_build(monkeypatch):
    torch.manual_seed(0)
    model = _TwoEngines().eval().cuda()
    x = torch.randn(4, 16, device="cuda")
    ep = torch.export.export(model, (x,))
    state = ep.state_dict
    released_at_build = []
    convert_module = _compiler.convert_module

    def record_released_weights(*args, **kwargs):
        released_at_build.append(
            {
                name
                for name in ("w_a", "w_b", "w_shared", "unused_head")
                if not state[name].untyped_storage().nbytes()
            }
        )
        return convert_module(*args, **kwargs)

    monkeypatch.setattr(_compiler, "convert_module", record_released_weights)
    torch_tensorrt.dynamo.compile(
        ep,
        arg_inputs=[x],
        min_block_size=1,
        torch_executed_ops=_TORCH_EXECUTED_OPS,
        release_weights_after_build=True,
    )
    # The head before the first build; w_a, read only by the first engine, before the second.
    assert released_at_build == [{"unused_head"}, {"unused_head", "w_a"}]


@pytest.mark.parametrize("release", [True, False])
@pytest.mark.parametrize("single_engine", [True, False])
def test_weights_are_released_before_the_engine_is_serialized(
    release, single_engine, monkeypatch
):
    torch.manual_seed(0)
    if single_engine:
        model, names = torch.nn.Linear(16, 16), ("weight", "bias")
        options = {"require_full_compilation": True}
    else:
        model, names = _TwoEngines(), ("w_a", "w_b", "w_shared")
        options = {"torch_executed_ops": _TORCH_EXECUTED_OPS}
    model = model.eval().cuda()
    x = torch.randn(4, 16, device="cuda")
    with torch.no_grad():
        expected = model(x)
    ep = torch.export.export(model, (x,))
    state = ep.state_dict
    released_at_serialize = []

    class _RecordingEngine:
        def __init__(self, engine):
            self._engine = engine

        def __getattr__(self, attr):
            return getattr(self._engine, attr)

        def serialize(self):
            released_at_serialize.append(
                {n for n in names if not state[n].untyped_storage().nbytes()}
            )
            return self._engine.serialize()

    run = _conversion.TRTInterpreter.run

    def run_and_record(self, *args, **kwargs):
        result = run(self, *args, **kwargs)
        return result._replace(engine=_RecordingEngine(result.engine))

    monkeypatch.setattr(_conversion.TRTInterpreter, "run", run_and_record)
    trt_gm = torch_tensorrt.dynamo.compile(
        ep,
        arg_inputs=[x],
        min_block_size=1,
        release_weights_after_build=release,
        **options,
    )
    with torch.no_grad():
        torch.testing.assert_close(trt_gm(x), expected, rtol=1e-2, atol=1e-2)
    if not release:
        expected_released = [set()] * (1 if single_engine else 2)
    elif single_engine:
        expected_released = [{"weight", "bias"}]
    else:
        # w_shared is read by both engines, so it goes only before the second is serialized.
        expected_released = [{"w_a"}, {"w_a", "w_b", "w_shared"}]
    assert released_at_serialize == expected_released


def test_engine_cache_hit_releases_weights(tmp_path, monkeypatch):
    hits = []
    pull_cached_engine = _conversion.pull_cached_engine

    def record_hit(*args, **kwargs):
        result = pull_cached_engine(*args, **kwargs)
        hits.append(result is not None)
        return result

    monkeypatch.setattr(_conversion, "pull_cached_engine", record_hit)
    x = torch.randn(4, 16, device="cuda")
    for _ in range(2):
        # A fresh instance each time: the cached engine is refit with its weights.
        model = torch.nn.Linear(16, 16).eval().cuda()
        with torch.no_grad():
            expected = model(x)
        ep = torch.export.export(model, (x,))
        trt_gm = torch_tensorrt.dynamo.compile(
            ep,
            arg_inputs=[x],
            min_block_size=1,
            immutable_weights=False,
            cache_built_engines=True,
            reuse_cached_engines=True,
            engine_cache_dir=str(tmp_path),
            release_weights_after_build=True,
        )
        with torch.no_grad():
            torch.testing.assert_close(trt_gm(x), expected, rtol=1e-2, atol=1e-2)
        for name, tensor in ep.state_dict.items():
            assert tensor.numel() == 0, name
    assert hits == [False, True]


@pytest.mark.parametrize("release", [True, None])
def test_build_frees_its_folded_constants(release, monkeypatch):
    class _Folded(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(64, 64), requires_grad=False)

        def forward(self, x):
            return x @ (self.w * 2)  # w * 2 is folded into a frozen parameter

    model = _Folded().eval().cuda()
    x = torch.randn(4, 64, device="cuda")
    folded_constants_alive = []
    interpret_module_to_result = _conversion.interpret_module_to_result

    def check_folded_constants_freed(module, *args, **kwargs):
        refs = [
            weakref.ref(getattr(module, name).untyped_storage())
            for name in dir(module)
            if name.startswith("_frozen_param")
        ]
        result = interpret_module_to_result(module, *args, **kwargs)
        folded_constants_alive.append(sum(ref() is not None for ref in refs))
        return result

    monkeypatch.setattr(
        _conversion, "interpret_module_to_result", check_folded_constants_freed
    )
    torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)),
        arg_inputs=[x],
        min_block_size=1,
        release_weights_after_build=release,
    )
    assert folded_constants_alive == [0]  # freed by conversion, as without the setting


def test_default_creates_no_releaser_without_mapped_weights(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("created a WeightReleaser")

    monkeypatch.setattr(_compiler, "has_mapped_weights", lambda: False)
    monkeypatch.setattr(_compiler, "WeightReleaser", fail)
    model = torch.nn.Linear(16, 16).eval().cuda()
    x = torch.randn(4, 16, device="cuda")
    torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)), arg_inputs=[x], min_block_size=1
    )
    assert model.weight.untyped_storage().nbytes() > 0


def test_lifted_constant_is_kept():
    scale = torch.randn(16, device="cuda")
    expected_scale = scale.clone()

    class _Scaled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16, 16)

        def forward(self, x):
            # A tensor the model doesn't own, lifted as a constant that shares its memory.
            return self.lin(x) * scale

    model = _Scaled().eval().cuda()
    x = torch.randn(4, 16, device="cuda")
    ep = torch.export.export(model, (x,))
    torch_tensorrt.dynamo.compile(
        ep, arg_inputs=[x], min_block_size=1, release_weights_after_build=True
    )
    # The caller's tensor, not the model's: kept.
    torch.testing.assert_close(scale, expected_scale)
    assert model.lin.weight.numel() == 0


def test_constant_folded_in_a_cond_branch_is_kept_until_built():
    class _Cond(torch.nn.Module):
        def forward(self, x):
            return torch.cond(
                x.sum() > 0,
                lambda x: x + torch.ones(4, 16, device=x.device),
                lambda x: x - 1,
                (x,),
            )

    model = _Cond().eval().cuda()
    x = torch.ones(4, 16, device="cuda")
    trt_gm = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)),
        arg_inputs=[x],
        min_block_size=1,
        release_weights_after_build=True,
    )
    with torch.no_grad():
        torch.testing.assert_close(trt_gm(x), model(x))


@pytest.mark.skipif(
    not torch_tensorrt.ENABLED_FEATURES.refit, reason="refit is not available"
)
def test_refit_after_release_with_the_global_partitioner():
    class _Layers(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1, self.lin2 = torch.nn.Linear(16, 16), torch.nn.Linear(16, 16)

        def forward(self, x):
            return self.lin2(torch.sin(self.lin1(x)))  # aten.sin runs in PyTorch

    x = torch.randn(4, 16, device="cuda")
    torch.manual_seed(0)
    trt_gm = torch_tensorrt.dynamo.compile(
        torch.export.export(_Layers().eval().cuda(), (x,)),
        arg_inputs=[x],
        min_block_size=1,
        torch_executed_ops={torch.ops.aten.sin.default},
        use_fast_partitioner=False,
        immutable_weights=False,
        release_weights_after_build=True,
    )
    torch.manual_seed(1)
    fresh = _Layers().eval().cuda()
    # The global partitioner left lin1 and lin2 in the compiled module, emptied: refit skips them.
    refitted = torch_tensorrt.dynamo.refit_module_weights(
        trt_gm, torch.export.export(fresh, (x,)), arg_inputs=[x]
    )
    with torch.no_grad():
        torch.testing.assert_close(refitted(x), fresh(x), rtol=1e-2, atol=1e-2)


def test_single_engine_fast_path_releases_weights():
    model = torch.nn.Linear(16, 16).eval().cuda()
    model.unused_head = torch.nn.Linear(16, 1000).cuda()
    x = torch.randn(4, 16, device="cuda")
    with torch.no_grad():
        expected = model(x)
    ep = torch.export.export(model, (x,))
    trt_gm = torch_tensorrt.dynamo.compile(
        ep,
        arg_inputs=[x],
        min_block_size=1,
        require_full_compilation=True,
        release_weights_after_build=True,
    )
    with torch.no_grad():
        torch.testing.assert_close(trt_gm(x), expected, rtol=1e-3, atol=1e-3)
    for name, tensor in ep.state_dict.items():
        assert tensor.numel() == 0, name


def test_mutated_buffer_survives_release():
    class _State(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(16, 16), requires_grad=False)
            self.register_buffer("state", torch.zeros(4, 16))

        def forward(self, x):
            y = x @ self.w + self.state
            self.state.copy_(y)
            return y * 2  # returning y itself trips an unrelated lowering error on main

    torch.manual_seed(0)
    model = _State().eval().cuda()
    reference = copy.deepcopy(model)
    x = torch.randn(4, 16, device="cuda")
    ep = torch.export.export(model, (x,))
    trt_gm = torch_tensorrt.dynamo.compile(
        ep, arg_inputs=[x], min_block_size=1, release_weights_after_build=True
    )
    assert ep.state_dict["w"].numel() == 0
    # Lifted into an engine input, not read through get_attr: never released.
    assert ep.state_dict["state"].untyped_storage().nbytes() > 0
    with torch.no_grad():
        out = trt_gm(x)
        # The compiled module returns a 1-tuple for this model, with or without the release.
        out = out[0] if isinstance(out, (tuple, list)) else out
        torch.testing.assert_close(out, reference(x), rtol=1e-2, atol=1e-2)


def test_mutable_module_rejects_release():
    with pytest.raises(ValueError, match="release_weights_after_build"):
        torch_tensorrt.MutableTorchTensorRTModule(
            torch.nn.Linear(4, 4).cuda(), release_weights_after_build=True
        )
