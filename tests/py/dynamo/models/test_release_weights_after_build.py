# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import torch_tensorrt
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo._weight_release import (
    WeightReleaser,
    parameter_storages,
    register_release_handler,
    release_storage,
    unregister_release_handler,
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


def test_release_handler_takes_precedence():
    weight = _param()
    seen = []

    def handler(storage):
        seen.append(storage.data_ptr())
        return True

    register_release_handler(handler)
    try:
        assert release_storage(weight.untyped_storage()) == weight.numel() * 4
    finally:
        unregister_release_handler(handler)
    assert seen == [weight.untyped_storage().data_ptr()]
    assert (
        weight.untyped_storage().nbytes() > 0
    )  # the handler, not resize_, owned the release


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

    def forward(self, x):
        y = x @ self.w_a + self.w_shared
        y = torch.sin(y * self.w_torch)  # aten.mul / aten.sin run in PyTorch
        return y @ self.w_b + self.w_shared


@pytest.mark.parametrize("release", [True, False])
def test_compile_releases_weights(release):
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
        release_weights_after_build=release,
    )
    with torch.no_grad():
        actual = trt_gm(x)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    state = ep.state_dict
    for name in ("w_a", "w_b", "w_shared", "unused_head"):
        assert (state[name].untyped_storage().nbytes() == 0) == release, name
    assert (
        state["w_torch"].untyped_storage().nbytes() > 0
    )  # read by the PyTorch partition
