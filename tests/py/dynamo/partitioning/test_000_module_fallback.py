# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Module fallback through the torch.compile backend, end to end."""

import logging
from unittest.mock import patch

import pytest
import torch
from torch_tensorrt.dynamo.backend import backends


class LinearBetweenEngines(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        return torch.sigmoid(self.linear(x * 0.5 + 0.25))


@pytest.mark.parametrize("use_fast_partitioner", [True, False], ids=["fast", "global"])
@torch.no_grad()
def test_module_fallback_torch_compile(use_fast_partitioner, caplog):
    model = LinearBetweenEngines().eval().cuda()
    sample = torch.randn(3, 8, device="cuda")
    compiled_graphs = []
    original_compile_module = backends.compile_module

    def capture_compile(*args, **kwargs):
        compiled_graphs.append(original_compile_module(*args, **kwargs))
        return compiled_graphs[-1]

    torch._dynamo.reset()
    try:
        with patch.object(backends, "compile_module", capture_compile):
            with caplog.at_level(logging.WARNING):
                compiled = torch.compile(
                    model,
                    backend="torch_tensorrt",
                    options={
                        "min_block_size": 1,
                        "torch_executed_modules": ["torch.nn.modules.linear.Linear"],
                        "use_fast_partitioner": use_fast_partitioner,
                        "pass_through_build_failures": True,
                    },
                )
                # Only the operations around the Linear run in TensorRT.
                torch.testing.assert_close(
                    compiled(sample), model(sample), rtol=1e-4, atol=1e-4
                )
    finally:
        torch._dynamo.reset()

    assert compiled_graphs, "The TensorRT backend was never invoked"
    for gm in compiled_graphs:
        assert any("_run_on_acc" in name for name, _ in gm.named_modules())
        targets = {
            str(node.target)
            for module in gm.modules()
            if isinstance(module, torch.fx.GraphModule)
            for node in module.graph.nodes
            if node.op == "call_function"
        }
        assert targets & {
            "aten.linear.default",
            "aten.addmm.default",
            "aten.mm.default",
        }, f"The excluded Linear was compiled into TensorRT: {targets}"
    # Dynamo records module classes, which must still match the qualified name.
    assert "were not found in the graph" not in caplog.text
