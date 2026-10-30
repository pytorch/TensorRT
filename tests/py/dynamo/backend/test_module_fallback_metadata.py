# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU regressions for module provenance before TensorRT engine construction."""

import pytest
import torch

from torch_tensorrt.dynamo.backend import backends
from torch_tensorrt.dynamo.partitioning.common import node_in_torch_executed_module


@pytest.mark.parametrize("class_metadata", [False, True])
@pytest.mark.parametrize("selected_type", [torch.nn.Sequential, torch.nn.Linear])
def test_module_fallback_matches_export_and_dynamo_metadata(
    class_metadata, selected_type
):
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    node = graph.call_function(torch.ops.aten.relu.default, (value,))
    stack = {
        "parent": ("parent", torch.nn.Sequential),
        "child": ("parent.linear", torch.nn.Linear),
    }
    if not class_metadata:
        stack = {
            key: (path, f"{cls.__module__}.{cls.__qualname__}")
            for key, (path, cls) in stack.items()
        }
    node.meta["nn_module_stack"] = stack
    before = dict(stack)
    selected_name = f"{selected_type.__module__}.{selected_type.__qualname__}"

    assert node_in_torch_executed_module(node, {selected_name})
    assert not node_in_torch_executed_module(node, {selected_type.__name__})
    assert not node_in_torch_executed_module(node, {"missing.Module"})
    assert not node_in_torch_executed_module(value, {selected_name})
    assert node.meta["nn_module_stack"] == before


@pytest.mark.parametrize("mode", ["default", "dynamic", "graph_break", "aot_autograd"])
@torch.no_grad()
def test_module_fallback_metadata_reaches_backend_partitioning(monkeypatch, mode):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, x):
            x = x * 0.5 + 0.25
            if mode == "graph_break":
                torch._dynamo.graph_break()
            return torch.sigmoid(self.linear(x))

    captured = []

    def capture_compile(gm, inputs, *, settings, **kwargs):
        # Stop before partitioning and engine construction. Backend capture and
        # lowering remain real; these regressions need no CUDA tensors or engines.
        captured.extend(
            (
                node.target,
                node_in_torch_executed_module(node, settings.torch_executed_modules),
            )
            for node in gm.graph.nodes
            if node.op == "call_function"
        )
        return gm

    monkeypatch.setattr(backends, "compile_module", capture_compile)
    torch._dynamo.reset()
    try:
        model = Model().eval()
        compiled = torch.compile(
            model,
            backend="torch_tensorrt",
            dynamic=mode == "dynamic",
            options={
                "torch_executed_modules": ["torch.nn.modules.linear.Linear"],
                "pass_through_build_failures": True,
                "use_distributed_mode_trace": mode == "aot_autograd",
            },
        )
        for batch in ([2, 5] if mode == "dynamic" else [2]):
            sample = torch.randn(batch, 8)
            torch.testing.assert_close(compiled(sample), model(sample))

        linear_targets = {
            torch.ops.aten.linear.default,
            torch.ops.aten.addmm.default,
            torch.ops.aten.mm.default,
        }
        linear_membership = [
            excluded for target, excluded in captured if target in linear_targets
        ]
        assert linear_membership and all(linear_membership)
        assert (torch.ops.aten.sigmoid.default, False) in captured
    finally:
        torch._dynamo.reset()
