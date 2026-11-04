# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import operator
import sys
import unittest.mock as mock

import torch
import torch.nn as nn
from torch.fx.passes.splitter_base import Subgraph
from torch.ops import aten
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo.conversion import CompilationSettings
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
    pre_export_lowering,
)
from torch_tensorrt.dynamo.lowering.passes import post_lowering, pre_export_lowering
from torch_tensorrt.dynamo.partitioning._resource_partitioner import (
    ResourcePartitioner,
)

# Fixed RSS value to make memory-budget calculations deterministic.
_FIXED_RSS_BYTES = 512 * 1024 * 1024  # 512 MB


class TestResourcePartitioning(TestCase):
    def test_atomic_subgraph_correction(self):
        class net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(3)
                self.relu = nn.ReLU()
                self.fc = nn.Linear(3 * 224 * 224, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x

        torch.manual_seed(0)
        model = net().eval()
        model.to("cuda")
        inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]

        exp_program = torch.export.export(model, tuple(inputs))

        compilation_options = {
            "min_block_size": 1,
            "immutable_weights": True,
            "reuse_cached_engines": False,
            "enable_resource_partitioning": True,
        }
        settings = CompilationSettings(**compilation_options)

        exported_program = pre_export_lowering(exp_program, settings)
        exported_program = exported_program.run_decompositions(
            get_decompositions(False)
        )

        gm = exported_program.module()
        gm = post_lowering(gm, settings)

        partitioned_module, supported_ops = partitioning.fast_partition(
            gm,
            min_block_size=settings.min_block_size,
            torch_executed_ops=settings.torch_executed_ops,
            require_full_compilation=settings.require_full_compilation,
            skip_fusion=True,
        )

        for name, _ in partitioned_module.named_children():
            submodule = getattr(partitioned_module, name)
            if (
                not isinstance(submodule, torch.fx.graph_module.GraphModule)
                or "_run_on_acc" not in name
            ):
                continue
            _mock_mem = mock.MagicMock()
            _mock_mem.rss = _FIXED_RSS_BYTES
            with mock.patch("psutil.Process") as mock_proc:
                mock_proc.return_value.memory_info.return_value = _mock_mem
                partitioner = ResourcePartitioner(
                    submodule,
                    submodule_name=name,
                    cpu_memory_budget=2 * 1024 * 1024 * 1024,
                )
            subgraphs = partitioner.put_nodes_into_subgraphs()
            new_subgraphs = []
            current_subgraph = []
            # Split the subgraph into two subgraphs by the ReLU node, which breaks the fusion group.
            for node in subgraphs[0].nodes:
                if node.op == "call_function" and node.target == aten.relu.default:
                    new_subgraphs.append(Subgraph(is_acc=True, nodes=current_subgraph))
                    current_subgraph = []
                current_subgraph.append(node)
            if current_subgraph:
                new_subgraphs.append(Subgraph(is_acc=True, nodes=current_subgraph))

            leaf_node = partitioner.get_leaf_node(new_subgraphs[0].nodes)
            broken_fusion = partitioner.step_if_break_fusion(
                new_subgraphs,
                leaf_node,
                set(new_subgraphs[0].nodes),
                set(new_subgraphs[1].nodes),
            )
            # The fusion was broken
            assert broken_fusion

            # The fusion should be fixed after the step
            partitioner._verify_all_fusion_nodes_in_same_subgraph(new_subgraphs)

            break

    def test_fusion_with_a_long_chain_of_producers(self):
        class MulAddAfterChain(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.ones(4))

            def forward(self, x, y):
                z = x * self.w
                # The add of the mul + add fusion reads a chain longer than the recursion limit.
                for _ in range(2 * sys.getrecursionlimit()):
                    y = torch.sin(y)
                return z + y

        inputs = (torch.randn(4), torch.randn(4))
        gm = torch.export.export(MulAddAfterChain(), inputs).module(check_guards=False)
        partitioner = ResourcePartitioner(
            gm, cpu_memory_budget=None, submodule_name="m"
        )
        nodes = partitioner.put_nodes_into_subgraphs()[0].nodes
        # Split right after the mul, which breaks the fusion.
        subgraphs = [
            Subgraph(is_acc=True, nodes=nodes[:1]),
            Subgraph(is_acc=True, nodes=nodes[1:]),
        ]
        leaf_node = partitioner.get_leaf_node(subgraphs[0].nodes)
        assert partitioner.step_if_break_fusion(
            subgraphs,
            leaf_node,
            set(subgraphs[0].nodes),
            set(subgraphs[1].nodes),
        )
        assert subgraphs[0].nodes == nodes
        assert subgraphs[1].nodes == []

    def test_getitems_stay_with_the_node_they_unpack(self):
        # After decomposition, batch norm returns three values and only the first is read.
        gm = (
            torch.export.export(nn.BatchNorm1d(4).eval(), (torch.randn(2, 4),))
            .run_decompositions()
            .module(check_guards=False)
        )
        partitioner = ResourcePartitioner(
            gm, cpu_memory_budget=None, submodule_name="m"
        )
        nodes = partitioner.put_nodes_into_subgraphs()[0].nodes
        (batch_norm,) = [
            n for n in nodes if any(u.target == operator.getitem for u in n.users)
        ]
        # One copy of the group, so memory grows with the number of outputs, not its square.
        assert partitioner.fusion_patterns == {
            batch_norm: {batch_norm, *batch_norm.users}
        }
        # Split right after the batch norm, which would make the first engine output a tuple.
        cut = nodes.index(batch_norm) + 1
        subgraphs = [
            Subgraph(is_acc=True, nodes=nodes[:cut]),
            Subgraph(is_acc=True, nodes=nodes[cut:]),
        ]
        leaf_node = partitioner.get_leaf_node(subgraphs[0].nodes)
        assert partitioner.step_if_break_fusion(
            subgraphs,
            leaf_node,
            set(subgraphs[0].nodes),
            set(subgraphs[1].nodes),
        )
        assert set(batch_norm.users) <= set(subgraphs[0].nodes)

    def test_split_subgraphs_fit_the_budget(self):
        class TwoBranches(nn.Module):
            def __init__(self):
                super().__init__()
                # 3 MiB, read at both ends of the large branch.
                self.scale = nn.Parameter(torch.ones(1536, 512))
                self.layers = nn.Sequential(
                    *[nn.Linear(512, 512, bias=False) for _ in range(4)]
                )

            def forward(self, a, x):
                # A branch without weights, which must not become an engine of its own.
                small = a.sin().sum()
                x = self.layers(x * self.scale)
                # Enough nodes for the split to take steps of several nodes.
                for _ in range(300):
                    x = torch.sin(x)
                return small + (x * self.scale).sum()

        inputs = (torch.randn(2, 4), torch.randn(1536, 512))
        gm = torch.export.export(TwoBranches(), inputs).module(check_guards=False)
        weights = gm.state_dict()
        # The second budget is smaller than the scale.
        for budget in (5 << 20, 2 << 20):
            partitioner = ResourcePartitioner(
                gm, cpu_memory_budget=None, submodule_name="m"
            )
            subgraphs = partitioner.break_subgraphs(
                partitioner.put_nodes_into_subgraphs(), subgraph_size_budget=budget
            )
            for subgraph in subgraphs:
                # Every engine embeds each weight it reads, even one that another engine also reads.
                read = {
                    i.target
                    for n in subgraph.nodes
                    for i in n.all_input_nodes
                    if i.op == "get_attr"
                }
                size = sum(weights[t].nbytes for t in read)
                assert read and (
                    size <= budget or read == {"scale"}
                ), f"{size} bytes in a subgraph, budget {budget}"
            partitioner.tag(subgraphs)
            torch.testing.assert_close(partitioner.split()(*inputs), gm(*inputs))

    def test_split_takes_large_steps_after_an_oversized_node(self):
        class OversizedThenTail(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.ones(256))

            def forward(self, x):
                for _ in range(100):
                    x = torch.sin(x)
                x = x * self.w
                for _ in range(1000):
                    x = torch.cos(x)
                return x

        gm = torch.export.export(OversizedThenTail(), (torch.randn(256),)).module(
            check_guards=False
        )
        partitioner = ResourcePartitioner(
            gm, cpu_memory_budget=None, submodule_name="m"
        )
        subgraphs = partitioner.put_nodes_into_subgraphs()
        num_nodes = len(subgraphs[0].nodes)
        with mock.patch.object(
            partitioner, "size_of_subgraphs", wraps=partitioner.size_of_subgraphs
        ) as size_of_subgraphs:
            # The weight alone is larger than the budget.
            partitioner.break_subgraphs(subgraphs, subgraph_size_budget=512)
        # Each size estimate walks the graph, so one per node would make the split quadratic.
        assert size_of_subgraphs.call_count < num_nodes // 2


if __name__ == "__main__":
    run_tests()
