# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys

import torch
from torch.fx.passes.splitter_base import Subgraph
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo.conversion import CompilationSettings
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
    pre_export_lowering,
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


def test_subgraphs_recompute_the_input_only_mask():
    """Each subgraph builds the mask from the padding input itself, which one subgraph
    reads, instead of receiving the 512x512 mask (1 MiB) from the first one."""
    torch.manual_seed(0)
    model = _MaskedLayers().cuda().eval()
    x = torch.randn(1, 512, 256, device="cuda") / 8
    pad = torch.ones(1, 512, dtype=torch.bool, device="cuda")
    pad[:, 400:] = False
    settings = CompilationSettings(min_block_size=1)
    exported = pre_export_lowering(torch.export.export(model, (x, pad)), settings)
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
