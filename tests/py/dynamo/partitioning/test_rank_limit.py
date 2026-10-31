# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.lowering.passes.restore_reshape import restore_reshape

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


class HighRankView(torch.nn.Module):
    def forward(self, x):
        return x.view(RANK_10_SHAPE) * 2.0


class HighRankViewAfterConvertibleOp(torch.nn.Module):
    def forward(self, x):
        return (x + 1.0).view(RANK_10_SHAPE) * 2.0


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

    @parameterized.expand([("fast", True), ("global", False)])
    def test_view_left_to_pytorch_is_core_aten(self, _, use_fast_partitioner):
        # aten._reshape_copy is not a core ATen op, so AOTInductor has no kernel for it.
        x = torch.rand(2, 4, 8, device="cuda")
        for model in (HighRankView(), HighRankViewAfterConvertibleOp()):
            with self.subTest(type(model).__name__):
                compiled = torch_tensorrt.compile(
                    model.eval().cuda(),
                    ir="dynamo",
                    inputs=[x],
                    min_block_size=1,
                    use_fast_partitioner=use_fast_partitioner,
                )
                targets = {
                    node.target
                    for module in compiled.modules()
                    if isinstance(module, torch.fx.GraphModule)
                    for node in module.graph.nodes
                }
                self.assertNotIn(torch.ops.aten._reshape_copy.default, targets)
                self.assertIn(torch.ops.aten.view.default, targets)
                self.assertEqual(compiled(x), model(x))

    def test_view_left_to_pytorch_does_not_alias_the_input(self):
        # torch.compile drops the clone of an input, so the reshape must copy.
        class CloneThenHighRankView(torch.nn.Module):
            def forward(self, x):
                return x.clone().view(RANK_10_SHAPE)

        x = torch.rand(2, 4, 8, device="cuda")
        compiled = torch.compile(
            CloneThenHighRankView(),
            backend="tensorrt",
            options={"min_block_size": 1, "pass_through_build_failures": True},
        )
        out = compiled(x)
        self.assertEqual(out, x.view(RANK_10_SHAPE))
        self.assertNotEqual(
            out.untyped_storage().data_ptr(), x.untyped_storage().data_ptr()
        )

    def test_restored_reshape_is_contiguous(self):
        # aten._reshape_copy returns a contiguous tensor, and so must its replacement.
        gm = torch.fx.symbolic_trace(
            lambda x: torch.ops.aten._reshape_copy.default(x.t(), [6, 4, 1])
        )
        restore_reshape(gm)
        self.assertTrue(gm(torch.rand(4, 6)).is_contiguous())


if __name__ == "__main__":
    run_tests()
