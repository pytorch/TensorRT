# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.lowering.passes.restore_reshape import restore_reshape

# TensorRT tensors have at most 8 dims, so a view to this shape stays in PyTorch.
RANK_10_SHAPE = (1, 2, 1, 1, 1, 1, 1, 1, 4, 8)


class HighRankView(torch.nn.Module):
    def forward(self, x):
        return x.view(RANK_10_SHAPE) * 2.0


class HighRankViewAfterConvertibleOp(torch.nn.Module):
    def forward(self, x):
        return (x + 1.0).view(RANK_10_SHAPE) * 2.0


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestViewLeftToPyTorch(TestCase):
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
