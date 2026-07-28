from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo import partitioning

from ..testing_utilities import lower_graph_testing

# Note: the following tests were a part of test_global_partitioning.py and were flaky when
# we ran all the tests. So, the following test cases were separated out in this test_flaky_global_partitioning.py
# The partitioned graphs were different when you ran the graph as a part of test_global_partitioning.py vs when you
# run these tests independently. pytest by default doesn't use parallel execution, so we are not sure why this behavior occurs
# currently. When you run these tests independently, the partitioned graph is structurally correct and is similar to fast partitioning.


class TestGlobalPartitioning(TestCase):
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
        # breakpoint()
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
