import unittest

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestRepeatConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3,), (1,)),
            ((3,), (0,)),
            ((3,), (2,)),
            ((2,), (2, 2)),
            ((2,), (0, 2)),
        ]
    )
    def test_repeat_1D(self, shape, repeats):
        class Repeat(nn.Module):
            def forward(self, x):
                return torch.ops.aten.repeat.default(x, repeats)

        inputs = [torch.randn(shape)]
        self.run_test(
            Repeat(),
            inputs,
        )

    @parameterized.expand(
        [
            # Unlike aten.tile, aten.repeat requires len(repeats) >= input.dim() —
            # it does not pad `repeats` with leading 1s.
            ((2, 3), (2, 2)),
            ((2, 3), (1, 0)),
            ((2, 3), (0, 2)),
            ((2, 3), (4, 2, 3)),
            ((2, 3), (0, 0, 3)),
            ((2, 3), (4, 2, 3, 1, 2)),
        ]
    )
    def test_repeat_2D(self, shape, repeats):
        class Repeat(nn.Module):
            def forward(self, x):
                return torch.ops.aten.repeat.default(x, repeats)

        inputs = [torch.randn(shape)]
        self.run_test(
            Repeat(),
            inputs,
        )

    @parameterized.expand(
        [
            ((1, 2, 3), (2, 3, 4)),
            ((1, 2, 3), (2, 3, 4, 5)),
        ]
    )
    def test_repeat_3D(self, shape, repeats):
        class Repeat(nn.Module):
            def forward(self, x):
                return torch.ops.aten.repeat.default(x, repeats)

        inputs = [torch.randn(shape)]
        self.run_test(
            Repeat(),
            inputs,
        )

    @parameterized.expand(
        [
            # Regression test for https://github.com/pytorch/TensorRT/issues/3172
            # and #3974: aten.repeat used to decompose into
            # unsqueeze -> expand -> reshape, which doubles the rank to 2N and
            # broke for any input of rank >= 5 since TensorRT tensors support
            # at most 8 dims. The direct converter builds the output at its
            # native rank, so these now succeed.
            ((1, 3, 4, 8, 8), (1, 1, 1, 1, 1)),
            ((1, 3, 4, 8, 8), (2, 1, 1, 1, 1)),
            ((2, 1, 3, 8, 8), (1, 1, 2, 1, 1)),
        ]
    )
    def test_repeat_5D(self, shape, repeats):
        class Repeat(nn.Module):
            def forward(self, x):
                return torch.ops.aten.repeat.default(x, repeats)

        inputs = [torch.randn(shape)]
        self.run_test(
            Repeat(),
            inputs,
        )


class TestRepeatConverterDynamicShape(DispatchTestCase):
    @parameterized.expand(
        [
            ((3,), (3,), (6,), (1,)),
            ((3,), (3,), (6,), (0,)),
            ((3,), (3,), (6,), (2,)),
            ((2,), (3,), (6,), (2, 2)),
            ((2,), (3,), (6,), (0, 2)),
            # 2d cases (aten.repeat requires len(repeats) >= input.dim())
            ((2, 3), (2, 3), (4, 3), (2, 2)),
            ((2, 3), (2, 3), (4, 3), (1, 0)),
            ((2, 3), (2, 3), (4, 3), (0, 2)),
            ((2, 3), (2, 3), (4, 3), (4, 2, 3)),
            ((2, 3), (2, 3), (4, 3), (0, 0, 3)),
            ((2, 3), (2, 3), (4, 3), (4, 2, 3, 1, 2)),
            # 3d cases
            ((1, 2, 3), (1, 2, 3), (6, 2, 3), (2, 3, 4)),
            ((1, 2, 3), (1, 2, 3), (6, 2, 3), (2, 3, 4, 5)),
            # 5d case from #3974
            (
                (1, 3, 4, 8, 8),
                (2, 3, 4, 8, 8),
                (4, 3, 4, 8, 8),
                (1, 1, 1, 1, 1),
            ),
        ]
    )
    def test_repeat_input_dynamic(self, min_shape, opt_shape, max_shape, repeats):
        class Repeat(nn.Module):
            def forward(self, x):
                return torch.ops.aten.repeat.default(x, repeats)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=torch.float32,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Repeat(),
            input_specs,
        )


@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TestRepeatValidator(TestCase):
    """repeat_validator is only consulted by the partitioner, which
    DispatchTestCase bypasses (it interprets the traced graph directly).
    These tests go through the full torch_tensorrt.compile() pipeline so the
    validator actually runs.
    """

    def test_repeat_rank_exceeding_trt_max_falls_back(self):
        # https://github.com/pytorch/TensorRT/issues/3172: a repeat call
        # producing a rank > 8 output can't be represented by a TensorRT
        # tensor. repeat_validator should reject it so the node runs in
        # PyTorch instead of crashing at engine build time.
        class Repeat(nn.Module):
            def forward(self, x):
                return x.repeat(1, 1, 1, 1, 1, 1, 1, 1, 2)  # output rank 9

        mod = Repeat().eval().cuda()
        inputs = [torch.randn(1, 1, 1, 1, 1, 1, 1, 3).cuda()]
        trt_mod = torch_tensorrt.compile(
            mod, ir="dynamo", inputs=inputs, min_block_size=1
        )
        acc_count = sum(
            1 for name, _ in trt_mod.named_children() if "_run_on_acc" in name
        )
        self.assertEqual(acc_count, 0)
        torch.testing.assert_close(trt_mod(*inputs), mod(*inputs))

    def test_repeat_rank_within_trt_max_converts(self):
        class Repeat(nn.Module):
            def forward(self, x):
                return x.repeat(1, 1, 1, 1, 1, 1, 1, 2)  # output rank 8

        mod = Repeat().eval().cuda()
        inputs = [torch.randn(1, 1, 1, 1, 1, 1, 1, 3).cuda()]
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
