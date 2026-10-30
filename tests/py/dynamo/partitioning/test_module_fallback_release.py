# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Release coverage for module fallback: check placement as well as outputs."""

import logging

import pytest
import torch

import torch_tensorrt
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo.partitioning.common import node_in_torch_executed_module


def _qualified_name(module_type):
    return f"{module_type.__module__}.{module_type.__qualname__}"


class FallbackBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU())

    def forward(self, x):
        return self.layers(x) + x


class NestedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = FallbackBlock()

    def forward(self, x):
        return torch.sigmoid(self.block(x * 0.5))


class SharedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(8, 8)

    def forward(self, x):
        return torch.sigmoid(self.shared(torch.relu(self.shared(x))))


class TupleBlock(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x), torch.cos(x)


class TupleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = TupleBlock()

    def forward(self, x):
        left, right = self.block(x * 0.5)
        return {"sum": left + right, "product": left * right}


class FunctionalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return torch.relu(x) + self.relu(x + 0.25)


class LoweredModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(8)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        return torch.sigmoid(self.activation(self.norm(x)))


def _fallback_nodes(compiled):
    """Fast partitioning has fallback children; global keeps ops at the root."""
    return [
        node
        for name, module in compiled.named_modules()
        if isinstance(module, torch.fx.GraphModule) and "_run_on_acc" not in name
        for node in module.graph.nodes
        if node.op == "call_function"
    ]


def _engine_count(compiled):
    return sum("_run_on_acc" in name for name, _ in compiled.named_children())


@pytest.mark.parametrize("use_fast_partitioner", [False, True], ids=["global", "fast"])
@pytest.mark.parametrize(
    "model_type,selected_type,expected_ops,minimum_counts,full_fallback",
    [
        (NestedModel, FallbackBlock, ("linear", "gelu", "add"), (1, 1, 1), False),
        (NestedModel, torch.nn.Sequential, ("linear", "gelu"), (1, 1), False),
        (SharedModel, torch.nn.Linear, ("linear",), (2,), False),
        (NestedModel, NestedModel, ("linear", "gelu", "sigmoid"), (1, 1, 1), True),
        (TupleModel, TupleBlock, ("sin", "cos"), (1, 1), False),
        (FunctionalModel, torch.nn.ReLU, ("relu",), (1,), False),
        (LoweredModel, torch.nn.LayerNorm, ("layer_norm",), (1,), False),
        (LoweredModel, torch.nn.GELU, ("gelu",), (1,), False),
    ],
    ids=[
        "custom-parent",
        "nested-container",
        "shared",
        "root",
        "tuple-output",
        "functional-boundary",
        "layernorm",
        "gelu",
    ],
)
def test_module_fallback_placement_and_accuracy(
    use_fast_partitioner,
    model_type,
    selected_type,
    expected_ops,
    minimum_counts,
    full_fallback,
):
    torch.manual_seed(42)
    model = model_type().eval().cuda()
    inputs = [torch.randn(2, 8, device="cuda")]
    selected = {_qualified_name(selected_type)}
    with torch.no_grad():
        expected = model(*inputs)
        compiled = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=inputs,
            min_block_size=1,
            disable_tf32=True,
            use_fast_partitioner=use_fast_partitioner,
            torch_executed_modules=selected,
        )
        torch.testing.assert_close(compiled(*inputs), expected, rtol=1e-4, atol=1e-4)
        # A second input catches accidental constant specialization as well.
        second_inputs = [inputs[0] * -0.75]
        torch.testing.assert_close(
            compiled(*second_inputs), model(*second_inputs), rtol=1e-4, atol=1e-4
        )

    nodes = _fallback_nodes(compiled)
    for target, count in zip(expected_ops, minimum_counts):
        matching = [node for node in nodes if target in str(node.target)]
        assert len(matching) >= count, (target, nodes)
        assert (
            sum(node_in_torch_executed_module(node, selected) for node in matching)
            >= count
        )
    assert (
        (_engine_count(compiled) == 0)
        if full_fallback
        else (_engine_count(compiled) > 0)
    )
    if model_type is FunctionalModel:
        # Selecting nn.ReLU must leave the unrelated functional ReLU accelerated.
        assert sum("relu" in str(node.target) for node in nodes) == 1


@pytest.mark.parametrize("use_fast_partitioner", [False, True], ids=["global", "fast"])
def test_module_fallback_combines_with_operator_fallback(use_fast_partitioner):
    model = NestedModel().eval().cuda()
    inputs = [torch.randn(2, 8, device="cuda")]
    with torch.no_grad():
        compiled = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=inputs,
            min_block_size=1,
            disable_tf32=True,
            use_fast_partitioner=use_fast_partitioner,
            torch_executed_modules={_qualified_name(FallbackBlock)},
            torch_executed_ops={torch.ops.aten.sigmoid.default},
        )
        torch.testing.assert_close(
            compiled(*inputs), model(*inputs), rtol=1e-4, atol=1e-4
        )
    targets = {str(node.target) for node in _fallback_nodes(compiled)}
    assert "aten.sigmoid.default" in targets
    assert any("linear" in target for target in targets)
    assert (
        _engine_count(compiled) > 0
    )  # The multiplication before the block remains TRT.


@pytest.mark.parametrize(
    "partitioner", [partitioning.global_partition, partitioning.fast_partition]
)
def test_unmatched_module_name_warns(partitioner, caplog):
    model = FunctionalModel().eval()
    graph = torch.export.export(model, (torch.randn(2, 8),)).module()
    with caplog.at_level(logging.WARNING):
        partitioner(graph, min_block_size=1, torch_executed_modules={"missing.Module"})
    assert "missing.Module" in caplog.text
    assert "were not found in the graph" in caplog.text


@pytest.mark.parametrize(
    "selected", ["ReLU", "relu", "torch.nn.ReLU", "torch.nn.modules.activation.ReLU"]
)
def test_selection_requires_fully_qualified_class_name(selected):
    # Exercise actual export metadata rather than constructing a synthetic stack.
    model = FunctionalModel().eval()
    graph = torch.export.export(model, (torch.randn(2, 8),)).module()
    matches = [
        node
        for node in graph.graph.nodes
        if node_in_torch_executed_module(node, {selected})
    ]
    if selected == _qualified_name(torch.nn.ReLU):
        assert len(matches) == 1
        assert matches[0].target == torch.ops.aten.relu.default
    else:
        assert matches == []


class PadConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = torch.nn.ZeroPad2d((1, 1, 1, 1))
        self.conv = torch.nn.Conv2d(3, 4, 3)

    def forward(self, x):
        return self.conv(self.pad(x))


@pytest.mark.parametrize("use_fast_partitioner", [False, True], ids=["global", "fast"])
def test_module_fallback_survives_pad_convolution_fusion(use_fast_partitioner):
    model = PadConvModel().eval().cuda()
    inputs = [torch.randn(1, 3, 8, 8, device="cuda")]
    selected = {_qualified_name(torch.nn.ZeroPad2d)}
    with torch.no_grad():
        compiled = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=inputs,
            min_block_size=1,
            disable_tf32=True,
            use_fast_partitioner=use_fast_partitioner,
            torch_executed_modules=selected,
        )
        torch.testing.assert_close(
            compiled(*inputs), model(*inputs), rtol=1e-4, atol=1e-4
        )
    assert any(
        "pad" in str(node.target) and node_in_torch_executed_module(node, selected)
        for node in _fallback_nodes(compiled)
    ), "Requested ZeroPad2d fallback was lost during padding/convolution fusion"
    assert _engine_count(compiled) > 0
