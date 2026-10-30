# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Release regressions: lowering must preserve explicit module fallback."""

import pytest
import torch

import torch_tensorrt
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering import get_decompositions, post_lowering
from torch_tensorrt.dynamo.partitioning.common import node_in_torch_executed_module


class PadThenConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = torch.nn.ZeroPad2d((1, 1, 1, 1))
        self.conv = torch.nn.Conv2d(3, 4, 3)

    def forward(self, x):
        return self.conv(self.pad(x))


class UnequalHeadAttention(torch.nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


class AttentionThenAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = UnequalHeadAttention()

    def forward(self, q, k, v):
        return self.attention(q, k, v) + 1


def _export_with_default_decompositions(model, inputs):
    return (
        torch.export.export(model, inputs)
        .run_decompositions(get_decompositions(False, False, False))
        .module()
    )


@pytest.mark.parametrize("force_fallback", [False, True], ids=["baseline", "fallback"])
def test_padding_module_fallback_survives_convolution_fusion(force_fallback):
    torch.manual_seed(0)
    model = PadThenConv().eval()
    inputs = (torch.randn(1, 3, 8, 8),)
    module_types = ["torch.nn.modules.padding.ZeroPad2d"] if force_fallback else []
    settings = CompilationSettings(
        torch_executed_modules=module_types, min_block_size=1
    )

    with torch.no_grad():
        gm = _export_with_default_decompositions(model, inputs)
        pad = next(
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.aten.constant_pad_nd.default
        )
        assert node_in_torch_executed_module(pad, module_types) == force_fallback
        lowered = post_lowering(gm, settings)
        torch.testing.assert_close(lowered(*inputs), model(*inputs))

    if force_fallback:
        assert any(
            node_in_torch_executed_module(node, module_types)
            for node in lowered.graph.nodes
            if node.op == "call_function"
        ), "Padding fusion removed the explicitly excluded ZeroPad2d module boundary"
    else:
        assert any(
            node.target == torch.ops.tensorrt.conv_asym_pad.default
            for node in lowered.graph.nodes
        ), "The baseline must exercise padding/convolution fusion"


@pytest.mark.parametrize("force_fallback", [False, True], ids=["baseline", "fallback"])
def test_attention_module_fallback_survives_shape_decomposition(force_fallback):
    torch.manual_seed(0)
    model = AttentionThenAdd().eval()
    inputs = (
        torch.randn(1, 2, 4, 8),
        torch.randn(1, 2, 4, 8),
        torch.randn(1, 2, 4, 16),
    )
    module_type = (
        f"{UnequalHeadAttention.__module__}.{UnequalHeadAttention.__qualname__}"
    )
    module_types = [module_type] if force_fallback else []
    settings = CompilationSettings(
        torch_executed_modules=module_types, min_block_size=1
    )

    with torch.no_grad():
        gm = _export_with_default_decompositions(model, inputs)
        attention = next(
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.aten.scaled_dot_product_attention.default
        )
        assert node_in_torch_executed_module(attention, module_types) == force_fallback
        outer_add = next(
            node for node in gm.graph.nodes if node.target == torch.ops.aten.add.Tensor
        )
        lowered = post_lowering(gm, settings)
        torch.testing.assert_close(lowered(*inputs), model(*inputs))

    attention_nodes = [
        node
        for node in lowered.graph.nodes
        if node.op == "call_function" and node is not outer_add
    ]
    assert attention_nodes
    if force_fallback:
        leaked = [
            node.name
            for node in attention_nodes
            if not node_in_torch_executed_module(node, module_types)
        ]
        assert not leaked, f"Attention decomposition lost module fallback on: {leaked}"
        assert not node_in_torch_executed_module(outer_add, module_types)
    else:
        assert any(
            node.target == torch.ops.aten.bmm.default for node in attention_nodes
        ), "The baseline must exercise attention decomposition"


@pytest.mark.parametrize("use_fast_partitioner", [True, False], ids=["fast", "global"])
@pytest.mark.parametrize("force_fallback", [False, True], ids=["baseline", "fallback"])
@torch.no_grad()
def test_attention_module_fallback_end_to_end(use_fast_partitioner, force_fallback):
    torch.manual_seed(0)
    model = AttentionThenAdd().eval().cuda()
    inputs = (
        torch.randn(1, 2, 4, 8, device="cuda"),
        torch.randn(1, 2, 4, 8, device="cuda"),
        torch.randn(1, 2, 4, 16, device="cuda"),
    )
    module_type = (
        f"{UnequalHeadAttention.__module__}.{UnequalHeadAttention.__qualname__}"
    )
    compiled = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=list(inputs),
        torch_executed_modules=[module_type] if force_fallback else [],
        min_block_size=1,
        disable_tf32=True,
        use_fast_partitioner=use_fast_partitioner,
        cache_built_engines=False,
    )
    assert any("_run_on_acc" in name for name, _ in compiled.named_modules())
    if not force_fallback:
        torch.testing.assert_close(
            compiled(*inputs), model(*inputs), rtol=1e-4, atol=1e-4
        )
        return
    targets = {
        str(node.target)
        for name, module in compiled.named_modules()
        if isinstance(module, torch.fx.GraphModule) and "_run_on_acc" not in name
        for node in module.graph.nodes
        if node.op == "call_function"
    }
    assert targets & {
        "aten.scaled_dot_product_attention.default",
        "aten.bmm.default",
    }, f"Explicitly excluded attention was compiled into TensorRT: {targets}"
    torch.testing.assert_close(compiled(*inputs), model(*inputs), rtol=1e-4, atol=1e-4)
