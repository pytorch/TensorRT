from torch_tensorrt.dynamo.backend.lowering import partition
from torch.testing._internal.common_utils import run_tests, TestCase
from utils import lower_graph_testing
import torch
from copy import deepcopy
import numpy as np


class TestPartitioning(TestCase):
    def test_partition_fully_supported_one_op(self):
        class FullySupportedOneOp(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, y):
                return torch.ops.aten.add.Tensor(x, y)

        fx_graph = torch.fx.symbolic_trace(FullySupportedOneOp())
        partitioned_graph = partition(deepcopy(fx_graph))
        self.assertEquals(
            len(list(partitioned_graph.named_children())),
            0,
            "Single operators should not be segmented",
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
        partitioned_graph = partition(deepcopy(fx_graph))
        self.assertEquals(
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
        partitioned_graph = partition(deepcopy(fx_graph), min_block_size=2)
        self.assertEquals(
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
        (unexpected_ops_seen, _, partitioned_graphs,) = lower_graph_testing(
            fx_graph,
            inputs,
            unexpected_ops=unexpected_ops,
            min_block_size=2,
            torch_executed_ops={"torch.ops.aten.add.Tensor"},
            testing_partitioning=True,
        )

        self.assertEquals(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEquals(
            len(partitioned_graphs),
            1,
            "Without control flow breaks, there should only be a single graph",
        )
        self.assertEquals(
            len(list(partitioned_graphs[0].named_children())),
            1,
            "Certain operators are set to run in Torch, expected 1 segment",
        )


if __name__ == "__main__":
    run_tests()
