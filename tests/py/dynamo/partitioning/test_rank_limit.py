import unittest

import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests

RANK_10_SHAPE = (1, 2, 1, 1, 1, 1, 1, 1, 4, 8)
RANK_8_SHAPE = (1, 2, 1, 1, 1, 1, 4, 8)


@torch.library.custom_op("torchtrt_partition_test::high_rank", mutates_args=())
def _high_rank(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(RANK_10_SHAPE).clone()


@_high_rank.register_fake
def _high_rank_fake(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(RANK_10_SHAPE)


@torch.library.custom_op("torchtrt_partition_test::low_rank", mutates_args=())
def _low_rank(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(RANK_8_SHAPE).clone()


@_low_rank.register_fake
def _low_rank_fake(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(RANK_8_SHAPE)


class HighRankBoundary(torch.nn.Module):
    def forward(self, x):
        return torch.ops.torchtrt_partition_test.high_rank(x) * 2.0


class LowRankBoundary(torch.nn.Module):
    def forward(self, x):
        return torch.ops.torchtrt_partition_test.low_rank(x) * 2.0


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTensorRankPartitioning(TestCase):
    @parameterized.expand([("fast", True), ("global", False)])
    def test_rank_above_trt_limit_falls_back(self, _, use_fast_partitioner):
        model = HighRankBoundary().eval().cuda()
        x = torch.rand(2, 4, 8, device="cuda")

        compiled = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=[x],
            min_block_size=1,
            use_fast_partitioner=use_fast_partitioner,
        )

        accelerated = [
            name for name, _ in compiled.named_children() if "_run_on_acc" in name
        ]
        self.assertEqual(accelerated, [])
        self.assertEqual(compiled(x), model(x))

    @parameterized.expand([("fast", True), ("global", False)])
    def test_rank_at_trt_limit_converts(self, _, use_fast_partitioner):
        model = LowRankBoundary().eval().cuda()
        x = torch.rand(2, 4, 8, device="cuda")

        compiled = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=[x],
            min_block_size=1,
            use_fast_partitioner=use_fast_partitioner,
        )

        accelerated = [
            name for name, _ in compiled.named_children() if "_run_on_acc" in name
        ]
        self.assertEqual(len(accelerated), 1)
        self.assertEqual(compiled(x), model(x))


if __name__ == "__main__":
    run_tests()
