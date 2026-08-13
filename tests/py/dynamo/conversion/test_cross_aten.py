import unittest

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestCrossConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3,), (3,), -1),
            ((4, 3), (4, 3), -1),
            ((4, 3), (4, 3), 1),
            ((3, 4, 5), (3, 4, 5), 0),
            ((2, 3, 4), (2, 3, 4), 1),
        ]
    )
    def test_linalg_cross(self, a_shape, b_shape, dim):
        class Cross(nn.Module):
            def forward(self, a, b):
                return torch.ops.aten.linalg_cross.default(a, b, dim=dim)

        inputs = [torch.randn(a_shape), torch.randn(b_shape)]
        self.run_test(
            Cross(),
            inputs,
        )

    def test_linalg_cross_broadcast(self):
        class Cross(nn.Module):
            def forward(self, a, b):
                return torch.ops.aten.linalg_cross.default(a, b, dim=-1)

        inputs = [torch.randn(5, 3), torch.randn(1, 3)]
        self.run_test(
            Cross(),
            inputs,
        )


class TestCrossConverterDynamicShape(DispatchTestCase):
    @parameterized.expand(
        [
            ((2, 3), (4, 3), (6, 3), -1),
            ((2, 3, 4), (4, 3, 4), (6, 3, 4), 1),
        ]
    )
    def test_linalg_cross_dynamic(self, min_shape, opt_shape, max_shape, dim):
        class Cross(nn.Module):
            def forward(self, a, b):
                return torch.ops.aten.linalg_cross.default(a, b, dim=dim)

        input_specs = [
            Input(min_shape=min_shape, opt_shape=opt_shape, max_shape=max_shape),
            Input(min_shape=min_shape, opt_shape=opt_shape, max_shape=max_shape),
        ]
        self.run_test_with_dynamic_shape(
            Cross(),
            input_specs,
        )


@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TestCrossEndToEnd(TestCase):
    """torch.cross and torch.linalg.cross both funnel into
    aten.linalg_cross.default after decomposition (torch.cross resolves its
    optional dim and delegates to linalg_cross), so one converter handles
    both public APIs. These tests go through the full compile() pipeline to
    confirm that end to end.
    """

    def test_torch_linalg_cross_compiles(self):
        class Cross(nn.Module):
            def forward(self, a, b):
                return torch.linalg.cross(a, b, dim=-1)

        mod = Cross().eval().cuda()
        inputs = [torch.randn(4, 5, 3).cuda(), torch.randn(4, 5, 3).cuda()]
        trt_mod = torch_tensorrt.compile(
            mod, ir="dynamo", inputs=inputs, min_block_size=1
        )
        acc_count = sum(
            1 for name, _ in trt_mod.named_children() if "_run_on_acc" in name
        )
        self.assertEqual(acc_count, 1)
        torch.testing.assert_close(trt_mod(*inputs), mod(*inputs))

    def test_torch_cross_compiles(self):
        class Cross(nn.Module):
            def forward(self, a, b):
                return torch.cross(a, b)

        mod = Cross().eval().cuda()
        inputs = [torch.randn(5, 3, 2).cuda(), torch.randn(5, 3, 2).cuda()]
        trt_mod = torch_tensorrt.compile(
            mod, ir="dynamo", inputs=inputs, min_block_size=1
        )
        acc_count = sum(
            1 for name, _ in trt_mod.named_children() if "_run_on_acc" in name
        )
        self.assertEqual(acc_count, 1)
        torch.testing.assert_close(trt_mod(*inputs), mod(*inputs))


if __name__ == "__main__":
    run_tests()
