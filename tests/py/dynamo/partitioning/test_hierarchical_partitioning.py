from copy import deepcopy

import numpy as np
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo import partitioning


class TestHierarchicalAdjacencyPartitioning(TestCase):
    def test_hierarchical_adjacency_partition_fully_supported_one_op(self):
        class FullySupportedOneOp(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, y):
                return torch.ops.aten.add.Tensor(x, y)

        fx_graph = torch.fx.symbolic_trace(FullySupportedOneOp())
        partitioned_graph, _ = partitioning.hierarchical_adjacency_partition(
            deepcopy(fx_graph),
        )
        self.assertEqual(
            len(
                [
                    1
                    for submod in list(partitioned_graph.named_children())
                    if "_run_on_acc" in submod[0]
                ]
            ),
            0,
            "Single operators should not be segmented",
        )

    def test_hierarchical_adjacency_partition_fully_supported_one_op_require_full_compilation(
        self,
    ):
        class FullySupportedOneOp(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, y):
                return torch.ops.aten.add.Tensor(x, y)

        fx_graph = torch.fx.symbolic_trace(FullySupportedOneOp())
        partitioned_graph, _ = partitioning.hierarchical_adjacency_partition(
            deepcopy(fx_graph), require_full_compilation=True
        )
        self.assertEqual(
            len(
                [
                    1
                    for submod in list(partitioned_graph.named_children())
                    if "_run_on_acc" in submod[0]
                ]
            ),
            1,
            "Single operators can be segmented if full compilation is required",
        )

    def test_hierarchical_adjacency_partition_fully_supported_multi_op(self):
        class FullySupportedMultiOp(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, y):
                sum_ = torch.ops.aten.sub.Tensor(x, y)
                concat_ = torch.ops.aten.cat.default(x, sum_)
                relu_ = torch.ops.aten.relu.default(concat_)
                pow_ = torch.ops.aten.pow.Tensor_Scalar(relu_, 2)
                return pow_

        fx_graph = torch.fx.symbolic_trace(FullySupportedMultiOp())
        partitioned_graph, _ = partitioning.hierarchical_adjacency_partition(
            deepcopy(fx_graph), min_block_size=2
        )
        self.assertEqual(
            len(
                [
                    1
                    for submod in list(partitioned_graph.named_children())
                    if "_run_on_acc" in submod[0]
                ]
            ),
            1,
            "All operators are supported, there should be one segment",
        )

    def test_hierarchical_adjacency_partition_partially_supported_multi_op(self):
        class PartiallySupportedMultiOp(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, y):
                sum_1 = torch.ops.aten.add.Tensor(x, y)
                sum_2 = torch.ops.aten.add.Tensor(x, sum_1)
                sum_ = np.sum(sum_1) + np.sum(sum_2)
                relu_ = torch.ops.aten.relu.default(sum_)
                pow_ = torch.ops.aten.pow.Tensor_Scalar(relu_, 2)
                return pow_

        fx_graph = torch.fx.symbolic_trace(PartiallySupportedMultiOp())
        partitioned_graph, _ = partitioning.hierarchical_adjacency_partition(
            deepcopy(fx_graph), min_block_size=2
        )
        self.assertEqual(
            len(
                [
                    1
                    for submod in list(partitioned_graph.named_children())
                    if "_run_on_acc" in submod[0]
                ]
            ),
            2,
            "Unsupported operators interleave supported ones, expected 2 segments",
        )

    def test_hierarchical_adjacency_partition_partially_supported_with_torch_executed_ops(
        self,
    ):
        class PartiallySupportedMultiOp(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, y):
                sum_1 = torch.ops.aten.add.Tensor(x, y)
                sum_2 = torch.ops.aten.add.Tensor(x, sum_1)
                sum_ = torch.ops.aten.add.Tensor(sum_1, sum_2)
                relu_ = torch.ops.aten.relu.default(sum_)
                pow_ = torch.ops.aten.pow.Tensor_Scalar(relu_, 2)
                return pow_

        torch_executed_ops = {torch.ops.aten.add.Tensor}

        fx_graph = torch.fx.symbolic_trace(PartiallySupportedMultiOp())
        partitioned_graph, _ = partitioning.hierarchical_adjacency_partition(
            deepcopy(fx_graph),
            min_block_size=1,
            torch_executed_ops=torch_executed_ops,
        )

        unexpected_ops = torch_executed_ops
        expected_ops = {torch.ops.aten.relu.default, torch.ops.aten.pow.Tensor_Scalar}

        unexpected_ops_seen = set()
        expected_ops_seen = set()

        for name, gm in partitioned_graph.named_children():
            if "_run_on_acc" in name:
                for node in gm.graph.nodes:
                    if node.op == "call_function":
                        if node.target in unexpected_ops:
                            unexpected_ops_seen.add(node.target)
                        elif node.target in expected_ops:
                            expected_ops_seen.add(node.target)

        expected_ops_unseen = expected_ops.difference(expected_ops_seen)

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )
        self.assertEqual(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn1 = torch.nn.BatchNorm2d(64)
            self.bn2 = torch.nn.BatchNorm2d(128)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = torch.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = torch.relu(x)
            return x

    def test_hierarchical_adjacency_partition_with_two_backends(self):
        from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
            DYNAMO_CONVERTERS as CONVERTERS,
        )
        from torch_tensorrt.dynamo.lowering import (
            get_decompositions,
            pre_export_lowering,
        )

        model = self.SimpleModel().cuda().eval()
        example_input = torch.randn(1, 3, 224, 224).cuda()

        exported_program = torch.export.export(model, (example_input,))
        exported_program = pre_export_lowering(exported_program)
        exported_program = exported_program.run_decompositions(get_decompositions())
        gm = exported_program.module()

        partitioned_graph, _ = partitioning.hierarchical_adjacency_partition(
            gm,
            min_block_size=1,
            backend_priority=["inductor", "tensorrt"],
            backend_support_map={
                "inductor": {
                    "torch.ops.aten.convolution.default",
                },
                "tensorrt": CONVERTERS.keys(),
            },
        )

        inductor_subgraphs_num = 0
        tensorrt_subgraphs_num = 0

        for name, gm in partitioned_graph.named_children():
            if "_run_on_acc_inductor" in name:
                inductor_subgraphs_num += 1
            elif "_run_on_acc_tensorrt" in name:
                tensorrt_subgraphs_num += 1
            else:
                raise ValueError(f"Unknown backend: {name}")

        self.assertEqual(
            inductor_subgraphs_num,
            2,
            "There should be 2 subgraphs running on inductor backend",
        )
        self.assertEqual(
            tensorrt_subgraphs_num,
            2,
            "There should be 2 subgraph running on tensorrt backend",
        )

    def test_hierarchical_adjacency_partition_with_two_backends_with_torch_executed_ops(
        self,
    ):
        from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
            DYNAMO_CONVERTERS as CONVERTERS,
        )
        from torch_tensorrt.dynamo.lowering import (
            get_decompositions,
            post_lowering,
            pre_export_lowering,
        )

        model = self.SimpleModel().cuda().eval()
        example_input = torch.randn(1, 3, 224, 224).cuda()
        exported_program = torch.export.export(model, (example_input,))
        exported_program = pre_export_lowering(exported_program)
        exported_program = exported_program.run_decompositions(get_decompositions())
        gm = exported_program.module()
        gm = post_lowering(gm)
        partitioned_graph, _ = partitioning.hierarchical_adjacency_partition(
            gm,
            min_block_size=1,
            backend_priority=["inductor", "tensorrt"],
            backend_support_map={
                "inductor": {
                    "torch.ops.aten.convolution.default",
                },
                "tensorrt": CONVERTERS.keys(),
            },
            torch_executed_ops={
                "torch.ops.aten._native_batch_norm_legit_no_training.default"
            },
        )

        inductor_subgraphs_num = 0
        tensorrt_subgraphs_num = 0
        torch_gpu_subgraphs_num = 0

        for name, gm in partitioned_graph.named_children():
            if "_run_on_acc_inductor" in name:
                inductor_subgraphs_num += 1
            elif "_run_on_acc_tensorrt" in name:
                tensorrt_subgraphs_num += 1
            elif "_run_on_gpu" in name:
                torch_gpu_subgraphs_num += 1
            else:
                raise ValueError(f"Unknown backend: {name}")

        self.assertEqual(
            torch_gpu_subgraphs_num,
            2,
            "There should be 2 subgraphs running on torch gpu backend",
        )
        self.assertEqual(
            inductor_subgraphs_num,
            2,
            "There should be 2 subgraphs running on inductor backend",
        )
        self.assertEqual(
            tensorrt_subgraphs_num,
            2,
            "There should be 2 subgraph running on tensorrt backend",
        )


if __name__ == "__main__":
    run_tests()
