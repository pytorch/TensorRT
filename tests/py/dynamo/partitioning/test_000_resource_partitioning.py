# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import operator
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

    def test_split_keeps_getitem_with_multi_output_producer(self):
        # LayerNorm affine weights are the only weights here, so every size-driven cut
        # lands right after a native_layer_norm, whose tuple output only its getitem reads.
        class net(nn.Module):
            def __init__(self):
                super().__init__()
                self.norms = nn.ModuleList(nn.LayerNorm(4096) for _ in range(6))

            def forward(self, x):
                for norm in self.norms:
                    x = torch.relu(norm(x))
                return x

        model = net().eval().cuda()
        inputs = [torch.randn((8, 4096)).cuda()]
        settings = CompilationSettings(
            min_block_size=1,
            immutable_weights=True,
            reuse_cached_engines=False,
            enable_resource_partitioning=True,
        )
        exported_program = pre_export_lowering(
            torch.export.export(model, tuple(inputs)), settings
        ).run_decompositions(get_decompositions(False))
        gm = post_lowering(exported_program.module(), settings)
        partitioned_module, _ = partitioning.fast_partition(
            gm, min_block_size=1, skip_fusion=True
        )
        name, submodule = next(
            (n, m) for n, m in partitioned_module.named_children() if "_run_on_acc" in n
        )

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
        norm_bytes = 2 * 4096 * 4
        subgraphs = partitioner.break_subgraphs(
            subgraphs, subgraph_size_budget=norm_bytes * 3 // 2
        )

        self.assertGreater(len(subgraphs), 1)
        subgraph_of = {n: i for i, s in enumerate(subgraphs) for n in s.nodes}
        getitems = [n for n in subgraph_of if n.target is operator.getitem]
        self.assertTrue(getitems)
        for getitem in getitems:
            self.assertEqual(
                subgraph_of[getitem],
                subgraph_of[getitem.args[0]],
                f"{getitem.name} was split from its producer {getitem.args[0].name}",
            )


if __name__ == "__main__":
    run_tests()
