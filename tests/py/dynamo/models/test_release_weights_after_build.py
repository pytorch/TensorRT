# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import torch_tensorrt
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo._weight_release import (
    WeightReleaser,
    parameter_storages,
    release_storage,
    weight_storages,
)


def test_release_weights_after_build_default_off():
    assert CompilationSettings().release_weights_after_build is False


def test_old_pickle_without_field_defaults_to_false():
    state = CompilationSettings().__dict__.copy()
    state.pop("release_weights_after_build", None)
    restored = CompilationSettings.__new__(CompilationSettings)
    restored.__setstate__(state)
    assert restored.release_weights_after_build is False


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
    releaser = WeightReleaser([first, second, also_shared, in_torch])

    releaser.built(weight_storages(first), "first")
    assert own.untyped_storage().nbytes() == 0
    assert shared.untyped_storage().nbytes() > 0

    releaser.built(weight_storages(second), "second")
    assert shared.untyped_storage().nbytes() > 0  # also_shared still reads it
    releaser.built(weight_storages(also_shared), "also_shared")
    assert shared.untyped_storage().nbytes() == 0

    # in_torch runs in PyTorch: it is counted but never built, so its weight stays.
    assert torch_side.untyped_storage().nbytes() > 0


def test_storage_rebound_after_counting_is_kept():
    shared = _param()
    first, second = _traced(shared), _traced(shared)
    releaser = WeightReleaser([first, second])
    shared.data = shared.data.clone()  # e.g. moved between devices after partitioning
    releaser.built(weight_storages(first), "first")
    assert (
        shared.untyped_storage().nbytes() > 0
    )  # unknown to the releaser: second may still read it


def test_releaser_releases_unread_parameters_only():
    used, unread = _param(), _param()
    holder = torch.nn.Module()
    holder.used, holder.unread = used, unread
    holder.register_buffer("buf", torch.randn(4))
    releaser = WeightReleaser([_traced(used)])
    released = releaser.release_unread(parameter_storages(holder))
    assert released == unread.numel() * unread.element_size()
    assert unread.untyped_storage().nbytes() == 0
    assert used.untyped_storage().nbytes() > 0
    assert holder.buf.untyped_storage().nbytes() > 0  # buffers are never candidates


def test_non_resizable_storage_is_skipped():
    weight = torch.frombuffer(bytearray(64), dtype=torch.float32)
    assert release_storage(weight.untyped_storage()) == 0
    assert weight.untyped_storage().nbytes() == 64


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


@pytest.mark.parametrize("use_fast_partitioner", [True, False])
@pytest.mark.parametrize("release", [True, False])
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
        torch_executed_ops={torch.ops.aten.sin.default, torch.ops.aten.mul.Tensor},
        use_fast_partitioner=use_fast_partitioner,
        release_weights_after_build=release,
    )
    with torch.no_grad():
        for _ in range(2):  # TF32 matmuls in the engines
            torch.testing.assert_close(trt_gm(x), expected, rtol=1e-2, atol=1e-2)

    state = ep.state_dict
    # A read-only buffer read by an engine is a weight like any other: released after its build.
    for name in ("w_a", "w_b", "w_shared", "unused_head", "bias"):
        assert (state[name].untyped_storage().nbytes() == 0) == release, name
    assert (
        state["w_torch"].untyped_storage().nbytes() > 0
    )  # read by the PyTorch partition


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
        assert tensor.untyped_storage().nbytes() == 0, name


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

    x = torch.randn(4, 16, device="cuda")
    ep = torch.export.export(_State().eval().cuda(), (x,))
    trt_gm = torch_tensorrt.dynamo.compile(
        ep, arg_inputs=[x], min_block_size=1, release_weights_after_build=True
    )
    assert ep.state_dict["w"].untyped_storage().nbytes() == 0
    # Lifted into an engine input, not read through get_attr: never released.
    buffers = list(trt_gm.buffers())
    assert buffers and all(b.untyped_storage().nbytes() > 0 for b in buffers)


def test_mutable_module_rejects_release():
    with pytest.raises(ValueError, match="release_weights_after_build"):
        torch_tensorrt.MutableTorchTensorRTModule(
            torch.nn.Linear(4, 4).cuda(), release_weights_after_build=True
        )
