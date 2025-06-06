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


class TestGlobalPartitioning(TestCase):
    @parameterized.expand(
        [
            ({}, 1),
            ({"torch.ops.aten.relu.default"}, 3),
        ]
    )
    def test_end2end_global_partition(self, torch_executed_ops, trt_mod_cnt):
        class SimpleCNN(torch.nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 12, 3, padding=1)
                self.bn = torch.nn.BatchNorm2d(12)
                self.conv2 = torch.nn.Conv2d(12, 12, 3, padding=1)
                self.fc1 = torch.nn.Linear(12 * 56 * 56, 10)

            def forward(self, x, b=5):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.bn(x)
                x = F.max_pool2d(x, (2, 2))
                x = self.conv2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, (2, 2))
                x = torch.flatten(x, 1)
                x = x + b
                return self.fc1(x)

        mod = SimpleCNN().to("cuda")
        mod.eval()
        with torch.no_grad():
            inputs = torch.rand((1, 3, 224, 224)).to("cuda")
            try:
                trt_mod = torch_tensorrt.compile(
                    mod,
                    ir="dynamo",
                    inputs=[inputs],
                    min_block_size=1,
                    torch_executed_ops=torch_executed_ops,
                    use_fast_partitioner=False,
                )
                cnt = 0
                for name, _ in trt_mod.named_children():
                    if "_run_on_acc" in name:
                        cnt += 1
                self.assertEqual(cnt, trt_mod_cnt)
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

    @pytest.mark.critical
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


if __name__ == "__main__":
    run_tests()
