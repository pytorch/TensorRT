# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Interactions of module exclusions with compiler and partitioner settings."""

from unittest.mock import patch

import pytest
import torch
from torch.fx.passes.splitter_base import FxNetSplitterInternalError

import torch_tensorrt
from torch_tensorrt.dynamo import partitioning


class SettingsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        return torch.sigmoid(self.linear(torch.relu(x)))


def _torch_targets(gm):
    return {
        str(node.target)
        for name, child in gm.named_modules()
        if isinstance(child, torch.fx.GraphModule) and "_run_on_acc" not in name
        for node in child.graph.nodes
        if node.op == "call_function"
    }


@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("min_block_size", [1, 100])
@torch.no_grad()
def test_module_fallback_minimum_block_size(fast, min_block_size):
    model = SettingsModel().eval().cuda()
    x = torch.randn(3, 8, device="cuda")
    gm = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=[x],
        min_block_size=min_block_size,
        use_fast_partitioner=fast,
        torch_executed_modules=["torch.nn.modules.linear.Linear"],
    )
    assert "aten.linear.default" in _torch_targets(gm)
    assert any("_run_on_acc" in name for name, _ in gm.named_children()) == (
        min_block_size == 1
    )
    torch.testing.assert_close(gm(x), model(x), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("fast", [True, False])
@torch.no_grad()
def test_module_fallback_partial_graph_rejects_full_compilation(fast):
    model = SettingsModel().eval().cuda()
    x = torch.randn(3, 8, device="cuda")
    with pytest.raises(AssertionError, match="require_full_compilation"):
        torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=[x],
            min_block_size=1,
            use_fast_partitioner=fast,
            require_full_compilation=True,
            torch_executed_modules=["torch.nn.modules.linear.Linear"],
        )


@torch.no_grad()
def test_module_fallback_survives_partitioner_retry():
    model = SettingsModel().eval().cuda()
    x = torch.randn(3, 8, device="cuda")
    # A real fast-partitioner failure must pass the same exclusions to global.
    with (
        patch.object(
            partitioning,
            "fast_partition",
            side_effect=FxNetSplitterInternalError("test retry"),
        ) as fast,
        patch.object(
            partitioning, "global_partition", wraps=partitioning.global_partition
        ) as global_partition,
    ):
        gm = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=[x],
            min_block_size=1,
            use_fast_partitioner=True,
            torch_executed_modules=["torch.nn.modules.linear.Linear"],
        )
    fast.assert_called_once()
    global_partition.assert_called_once()
    assert "aten.linear.default" in _torch_targets(gm)
    assert any("_run_on_acc" in name for name, _ in gm.named_children())
    torch.testing.assert_close(gm(x), model(x), atol=1e-4, rtol=1e-4)
