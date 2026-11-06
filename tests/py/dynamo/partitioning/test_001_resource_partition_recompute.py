# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import operator
import sys

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.splitter_base import Subgraph
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo.conversion import CompilationSettings
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
    pre_export_lowering,
)
from torch_tensorrt.dynamo.lowering.passes.reset_folded_constructors import (
    FOLDED_CONSTRUCTOR_META,
)
from torch_tensorrt.dynamo.partitioning._resource_partitioner import (
    ENGINE_COMPILATION_MEMORY_USAGE_MULTIPLIER,
    ResourcePartitioner,
)

LAYER_BYTES = 1 << 20


def _mask(pad):
    return torch.where(pad[:, None, :] & pad[:, :, None], 0.0, -1e4)


class _MaskedLayers(torch.nn.Module):
    """Attention-like layers that all read one 512x512 mask built from the padding input."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(256, LAYER_BYTES // (256 * 4), bias=False) for _ in range(8)
        )

    def forward(self, x, pad):
        mask = _mask(pad)
        for layer in self.layers:
            x = (
                torch.softmax(x @ x.transpose(1, 2) / 16 + mask, -1)
                @ layer(x)[..., :256]
            )
        return x


@pytest.mark.parametrize("dynamic", [False, True])
def test_subgraphs_recompute_the_input_only_mask(dynamic):
    """Each subgraph builds the mask from the padding input itself, which one subgraph
    reads, instead of receiving the 512x512 mask (1 MiB) from the first one. With a dynamic
    sequence length, sizes are symbolic and compared over the whole range."""
    torch.manual_seed(0)
    model = _MaskedLayers().cuda().eval()
    x = torch.randn(1, 512, 256, device="cuda") / 8
    pad = torch.ones(1, 512, dtype=torch.bool, device="cuda")
    pad[:, 400:] = False
    settings = CompilationSettings(min_block_size=1)
    seq = torch.export.Dim("seq", min=256, max=1024)
    dynamic_shapes = ({1: seq}, {1: seq}) if dynamic else None
    exported = pre_export_lowering(
        torch.export.export(model, (x, pad), dynamic_shapes=dynamic_shapes), settings
    )
    gm = post_lowering(
        exported.run_decompositions(get_decompositions(False)).module(), settings
    )
    partitioned, _ = partitioning.fast_partition(gm, min_block_size=1, skip_fusion=True)
    partitioner = ResourcePartitioner(
        partitioned._run_on_acc_0, cpu_memory_budget=None, submodule_name="acc"
    )
    # One 1 MiB layer per subgraph.
    partitioner.remaining_memory_budget = (
        ENGINE_COMPILATION_MEMORY_USAGE_MULTIPLIER * 3 * LAYER_BYTES // 2
    )
    partitioned._run_on_acc_0 = split = partitioner.partition_graph()

    pieces = [m for n, m in split.named_children() if "resource_split" in n]
    mask = _mask(pad)
    masks, pad_readers = [], []
    for piece in pieces:
        piece.register_forward_pre_hook(
            lambda _, args: masks.extend(a for a in args if a.equal(mask))
        )
        piece.register_forward_pre_hook(
            lambda m, args: pad_readers.extend(
                [m] if any(a is pad for a in args) else []
            )
        )
    with torch.no_grad():
        torch.testing.assert_close(partitioned(x, pad), model(x, pad))
    assert len(pieces) >= 4
    assert masks == []
    assert len(pad_readers) <= 1


def _recompute(module, inputs, cuts, folded=()):
    """Trace `module`, give its nodes fake values, and recompute across subgraphs that start at the call
    nodes with the indices in `cuts`. The attributes named in `folded` are marked as constant folding
    marks the constants it makes."""
    gm = torch.fx.symbolic_trace(module)
    for node in gm.graph.nodes:
        if node.op == "get_attr" and node.target in folded:
            node.meta[FOLDED_CONSTRUCTOR_META] = True
    FakeTensorProp(gm, FakeTensorMode(allow_non_fake_inputs=True)).propagate(*inputs)
    nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    bounds = [0, *cuts, len(nodes)]
    subgraphs = [
        Subgraph(is_acc=True, nodes=nodes[a:b]) for a, b in zip(bounds, bounds[1:])
    ]
    partitioner = ResourcePartitioner(gm, cpu_memory_budget=None, submodule_name="m")
    subgraphs = partitioner.recompute_input_only_nodes(subgraphs)
    gm.graph.lint()
    gm.recompile()
    return gm, nodes, subgraphs


def _inputs_of(subgraph):
    """The nodes a subgraph reads from outside itself."""
    inside = set(subgraph.nodes)
    return {i for n in subgraph.nodes for i in n._input_nodes} - inside


class _DerivedInALaterSubgraph(torch.nn.Module):
    def forward(self, x, pad):
        mask = _mask(pad)
        a = x + mask  # subgraph 0
        c = mask * 2  # subgraph 1, from the mask it recomputes
        b = a * c  # subgraph 1
        return b + c  # subgraph 2 reads c, which subgraph 1 computes from its copy


def test_a_value_derived_from_a_recomputed_one_reaches_a_third_subgraph():
    module = _DerivedInALaterSubgraph()
    x, pad = torch.randn(1, 256, 256), torch.rand(1, 256) > 0.2
    gm, nodes, subgraphs = _recompute(module, (x, pad), cuts=[-3, -1])
    assert not _inputs_of(subgraphs[2]) & {nodes[-3]}  # c is recomputed, not passed
    torch.testing.assert_close(gm(x, pad), module(x, pad))


class _UnbindInput(torch.nn.Module):
    def forward(self, x, pair):
        first, second = torch.unbind(pair)
        return (x + first) * second  # the multiply is in subgraph 1


def test_a_getitem_crosses_with_its_producer():
    module = _UnbindInput()
    x, pair = torch.randn(256, 256), torch.randn(2, 256, 256)
    gm, nodes, subgraphs = _recompute(module, (x, pair), cuts=[-1])
    # The getitem of an input's unbind isn't copied without the unbind, which would make subgraph 0
    # output a tuple.
    assert all(n.target is not operator.getitem for n in subgraphs[1].nodes)
    torch.testing.assert_close(gm(x, pair), module(x, pair))


class _RandomFromTheMask(torch.nn.Module):
    def forward(self, x, pad):
        noise = torch.rand_like(_mask(pad))
        return (x + noise) * noise  # the multiply is in subgraph 1


def test_a_random_value_is_passed_not_recomputed():
    x, pad = torch.randn(1, 256, 256), torch.rand(1, 256) > 0.2
    _, nodes, subgraphs = _recompute(_RandomFromTheMask(), (x, pad), cuts=[-1])
    noise = next(n for n in nodes if n.target is torch.rand_like)
    assert noise in _inputs_of(subgraphs[1])


class _TwoReadersOfTheMask(torch.nn.Module):
    def forward(self, x, pad):
        mask = _mask(pad)
        return torch.relu(x + mask) * mask  # the multiply is in subgraph 1


@pytest.mark.parametrize("length,recomputed", [(64, False), (256, True)])
def test_only_a_large_value_is_recomputed(length, recomputed):
    """A 64x64 float mask (16 KiB) crosses as it is; a 256x256 one (256 KiB) is recomputed."""
    x, pad = torch.randn(1, length, length), torch.rand(1, length) > 0.2
    _, nodes, subgraphs = _recompute(_TwoReadersOfTheMask(), (x, pad), cuts=[-1])
    mask = next(n for n in nodes if n.target is torch.where)
    assert (mask not in _inputs_of(subgraphs[1])) == recomputed


class _MaskTimesFolded(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.register_buffer("folded", torch.ones(256, 256, dtype=dtype))

    def forward(self, x, pad):
        y = _mask(pad) * self.folded
        return torch.relu(x + y) * y  # the multiply is in subgraph 1


@pytest.mark.parametrize(
    "dtype,recomputed", [(torch.bool, True), (torch.float32, False)]
)
def test_a_folded_float_constant_is_a_weight(dtype, recomputed):
    """A folded bool constant (a mask's fixed part, 64 KiB) may be recomputed with the chain; a folded
    float one (a weight transform, 256 KiB) is a weight, so the value it feeds is passed.
    """
    x, pad = torch.randn(1, 256, 256), torch.rand(1, 256) > 0.2
    _, nodes, subgraphs = _recompute(
        _MaskTimesFolded(dtype), (x, pad), cuts=[-1], folded=("folded",)
    )
    y = next(n for n in nodes if n.target is operator.mul)
    assert (y not in _inputs_of(subgraphs[1])) == recomputed


def test_an_input_only_chain_deeper_than_the_recursion_limit():
    """The chain is walked without recursion, so its depth isn't bounded by Python's stack."""
    depth = 2 * sys.getrecursionlimit()

    def chain(pad):
        mask = _mask(pad)
        for _ in range(depth):
            mask = torch.sin(mask)
        return mask

    class TwoReaders(torch.nn.Module):
        def forward(self, x, pad):
            mask = chain(pad)
            return torch.relu(x + mask) * mask

    gm = torch.fx.symbolic_trace(TwoReaders())
    nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    # The whole chain and the first reader in one subgraph, the second reader in the next.
    first, second = nodes[:-1], nodes[-1:]
    partitioner = ResourcePartitioner(gm, cpu_memory_budget=None, submodule_name="m")
    subgraphs = partitioner.recompute_input_only_nodes(
        [Subgraph(is_acc=True, nodes=first), Subgraph(is_acc=True, nodes=second)]
    )

    assert {i for n in subgraphs[1].nodes for i in n._input_nodes} & set(first) <= {
        n for n in first if any(i.op == "placeholder" for i in n._input_nodes)
    }  # only the nodes reading the padding input cross
    assert len(subgraphs[1].nodes) > depth
    gm.graph.lint()
    gm.recompile()
    x = torch.randn(1, 4, 4)
    pad = torch.tensor([[True, True, True, False]])
    torch.testing.assert_close(gm(x, pad), torch.relu(x + chain(pad)) * chain(pad))
