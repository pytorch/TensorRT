from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo import partitioning

from ..testing_utilities import lower_graph_testing


class TestGlobalPartitioning(TestCase):
    def test_end2end_global_partition(self):
        class SimpleCNN(torch.nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
                self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
                self.fc1 = torch.nn.Linear(32 * 134 * 134, 10)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, kernel_size=2, stride=2)
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, kernel_size=2, stride=2)
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                return x

        mod = SimpleCNN().to(dtype=torch.float16, device=torch.device("cuda"))
        mod.eval()
        batch_size, tile_size = 1, 538
        with torch.no_grad():
            inputs = torch.randn(
                batch_size, 3, tile_size, tile_size, device="cuda", dtype=torch.float16
            )
            try:
                torch_tensorrt.compile(
                    mod,
                    ir="dynamo",
                    inputs=[inputs],
                    enabled_precisions={torch.float16},
                    use_fast_partitioner=False,
                )
            except Exception as e:
                pytest.fail(f"unexpected exception raised: {e}")

    def test_partition_fully_supported_one_op(self):
        class FullySupportedOneOp(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, y):
                return torch.ops.aten.add.Tensor(x, y)

        fx_graph = torch.fx.symbolic_trace(FullySupportedOneOp())
        partitioned_graph, _ = partitioning.global_partition(deepcopy(fx_graph))
        self.assertEqual(
            len(list(partitioned_graph.named_children())),
            0,
            "Single operators should not be segmented",
        )

    def test_partition_fully_supported_one_op_require_full_compilation(self):
        class FullySupportedOneOp(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, y):
                return torch.ops.aten.add.Tensor(x, y)

        fx_graph = torch.fx.symbolic_trace(FullySupportedOneOp())
        partitioned_graph, _ = partitioning.global_partition(
            deepcopy(fx_graph), require_full_compilation=True
        )
        self.assertEqual(
            len(list(partitioned_graph.named_children())),
            1,
            "Single operators can be segmented if full compilation is required",
        )

    def test_partition_fully_supported_multi_op(self):
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
        partitioned_graph, _ = partitioning.global_partition(
            deepcopy(fx_graph), min_block_size=2
        )
        self.assertEqual(
            len(list(partitioned_graph.named_children())),
            1,
            "All operators are supported, there should be one segment",
        )

    def test_partition_partially_supported_multi_op(self):
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
        partitioned_graph, _ = partitioning.global_partition(
            deepcopy(fx_graph), min_block_size=2
        )
        self.assertEqual(
            len(list(partitioned_graph.named_children())),
            2,
            "Unsupported operators interleave supported ones, expected 2 segments",
        )

    def test_partition_partially_supported_with_torch_executed_ops(self):
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

        unexpected_ops = {torch.ops.aten.add.Tensor}

        inputs = [
            torch.randint(
                1,
                10,
                (5,),
            ),
            torch.randint(
                1,
                10,
                (5,),
            ),
        ]

        fx_graph = torch.fx.symbolic_trace(PartiallySupportedMultiOp())
        (
            unexpected_ops_seen,
            _,
            partitioned_graphs,
        ) = lower_graph_testing(
            fx_graph,
            inputs,
            unexpected_ops=unexpected_ops,
            min_block_size=2,
            torch_executed_ops={"torch.ops.aten.add.Tensor"},
            testing_partitioning=True,
            use_fast_partitioner=False,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
            len(partitioned_graphs),
            1,
            "Without control flow breaks, there should only be a single graph",
        )
        self.assertEqual(
            len(list(partitioned_graphs[0].named_children())),
            1,
            "Certain operators are set to run in Torch, expected 1 segment",
        )


if __name__ == "__main__":
    run_tests()
