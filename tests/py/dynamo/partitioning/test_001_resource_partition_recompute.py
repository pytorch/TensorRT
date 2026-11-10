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
from torch_tensorrt.dynamo.lowering.passes.constant_folding import constant_fold
from torch_tensorrt.dynamo.lowering.passes.fuse_distributed_ops import (
    tensorrt_fused_nccl_all_reduce_op,
)
from torch_tensorrt.dynamo.partitioning._resource_partitioner import (
    ENGINE_COMPILATION_MEMORY_USAGE_MULTIPLIER,
    ResourcePartitioner,
)

LAYER_BYTES = 1 << 20


def _mask(pad):
    return torch.where(pad[:, None, :] & pad[:, :, None], 0.0, -1e4)


class _MaskedLayers(torch.nn.Module):
    """Attention-like layers that all read one 512x512 mask built from the padding input, with a fixed
    causal part if `causal`."""

    def __init__(self, causal=False):
        super().__init__()
        self.causal = causal
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(256, LAYER_BYTES // (256 * 4), bias=False) for _ in range(8)
        )

    def mask(self, x, pad):
        mask = _mask(pad)
        if self.causal:  # a float constant once folded, with a static sequence length
            n = x.shape[1]
            mask = mask + torch.full((n, n), -1e4, device=x.device).triu(1)
        return mask

    def forward(self, x, pad):
        mask = self.mask(x, pad)
        for layer in self.layers:
            x = (
                torch.softmax(x @ x.transpose(1, 2) / 16 + mask, -1)
                @ layer(x)[..., :256]
            )
        return x


@pytest.mark.parametrize(
    "dynamic,causal", [(False, False), (True, False), (False, True)]
)
def test_subgraphs_recompute_the_input_only_mask(dynamic, causal):
    """Each subgraph builds the mask from the padding input itself, which one subgraph
    reads, instead of receiving the 512x512 mask (1 MiB) from the first one. With a dynamic
    sequence length, sizes are symbolic and compared over the whole range. A fixed causal part
    folds into a 1 MiB float constant, which each subgraph reads too."""
    torch.manual_seed(0)
    model = _MaskedLayers(causal).cuda().eval()
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
    mask = model.mask(x, pad)
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


def _recompute(module, inputs, cuts, fold=False):
    """Trace `module`, give its nodes fake values, and recompute across subgraphs that start at the call
    nodes with the indices in `cuts`. With `fold`, the module is exported and constant-folded first, as
    compilation does, so that its constructors become constants."""
    if fold:
        gm = constant_fold(
            torch.export.export(module, inputs).module(), CompilationSettings()
        )
    else:
        gm = torch.fx.symbolic_trace(module)
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
    # The third subgraph recomputes c, and b from it, which empties the second: it's dropped, and the
    # copy of the mask made for its c is erased rather than left without a reader.
    assert len(subgraphs) == 2
    assert all(n.users for subgraph in subgraphs for n in subgraph.nodes)
    assert not _inputs_of(subgraphs[-1]) & {nodes[-3]}  # c is recomputed, not passed
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


class _AllReducedMask(torch.nn.Module):
    def forward(self, x, pad):
        reduced = tensorrt_fused_nccl_all_reduce_op(_mask(pad), "sum", "0")
        return torch.relu(x + reduced) * reduced  # the multiply is in subgraph 1


def test_a_collective_is_passed_not_recomputed():
    """Every rank must issue the same collectives, and each rank splits by its own memory reading."""
    x, pad = torch.randn(1, 256, 256), torch.rand(1, 256) > 0.2
    _, nodes, subgraphs = _recompute(_AllReducedMask(), (x, pad), cuts=[-1])
    reduced = next(n for n in nodes if n.target is tensorrt_fused_nccl_all_reduce_op)
    assert reduced in _inputs_of(subgraphs[1])


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


class _MaskPlusFolded(torch.nn.Module):
    """A mask plus a fixed part that constant folding turns into a 1 MiB float constant, built from
    constructors (a causal triangle) or from a parameter (a weight transform)."""

    def __init__(self, from_parameter):
        super().__init__()
        self.from_parameter = from_parameter
        self.weight = torch.nn.Parameter(torch.full((512, 512), -1e4))

    def forward(self, x, pad):
        fixed = self.weight if self.from_parameter else torch.full((512, 512), -1e4)
        y = _mask(pad) + fixed.triu(1)
        return torch.relu(x + y) * y  # the multiply is in subgraph 1


@pytest.mark.parametrize("from_parameter,recomputed", [(False, True), (True, False)])
def test_a_constant_folded_from_a_parameter_is_a_weight(from_parameter, recomputed):
    """A constant folded without reading a parameter, such as a mask's fixed part, may be recomputed
    with the chain up to 4 MiB, whatever its dtype; one folded from a parameter is a weight
    transform, so the value it feeds is passed."""
    x, pad = torch.randn(1, 512, 512), torch.rand(1, 512) > 0.2
    _, nodes, subgraphs = _recompute(
        _MaskPlusFolded(from_parameter), (x, pad), cuts=[-1], fold=True
    )
    y = next(n for n in nodes if n.target is torch.ops.aten.add.Tensor)
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
