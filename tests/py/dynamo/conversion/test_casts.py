import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo.conversion import UnsupportedOperatorException

from .harness import DispatchTestCase


class TestCloneConverter(DispatchTestCase):
    def test_clone_contiguous(self):
        class Clone(nn.Module):
            def forward(self, x):
                y = torch.clone(x, memory_format=torch.contiguous_format)
                return y + 1

        inputs = [torch.randn((1, 3, 10))]
        self.run_test(
            Clone(),
            inputs,
            expected_ops={torch.ops.aten.clone.default},
            disable_passes=True,
        )

    def test_clone_regular(self):
        class Clone(nn.Module):
            def forward(self, x):
                y = torch.clone(x)
                return y + 1

        inputs = [torch.randn((8, 2, 10))]
        self.run_test(
            Clone(),
            inputs,
            expected_ops={torch.ops.aten.clone.default},
            disable_passes=True,
        )

    def test_clone_direct(self):
        class Clone(nn.Module):
            def forward(self, x):
                return x.clone()

        inputs = [torch.randn((8, 2, 10))]
        self.run_test(
            Clone(),
            inputs,
            expected_ops={torch.ops.aten.clone.default},
            disable_passes=True,
        )


class TestToCopyConverter(DispatchTestCase):
    def test_to_copy_half(self):
        class ToCopyHalf(nn.Module):
            def forward(self, x):
                y = x.to(dtype=torch.half)
                return y

        inputs = [torch.rand((1, 3, 10))]
        self.run_test(
            ToCopyHalf(),
            inputs,
            expected_ops={torch.ops.aten._to_copy.default},
            precision=torch.half,
            disable_passes=True,
        )

    def test_to_copy_float(self):
        class ToCopyFloat(nn.Module):
            def forward(self, x):
                y = x.to(dtype=torch.float)
                return y

        inputs = [torch.rand((1, 3, 10)).half()]
        self.run_test(
            ToCopyFloat(),
            inputs,
            expected_ops={torch.ops.aten._to_copy.default},
            precision=torch.float,
            disable_passes=True,
        )

    def test_to_copy_unsupported(self):
        class ToCopy64Bit(nn.Module):
            def forward(self, x):
                y = x.to(dtype=torch.int64)
                return y

        inputs = [torch.randn((1, 3, 10)).int()]

        with self.assertRaises(UnsupportedOperatorException):
            self.run_test(
                ToCopy64Bit(),
                inputs,
                expected_ops={torch.ops.aten._to_copy.default},
                disable_passes=True,
            )

    def test_to_copy_direct(self):
        class ToCopyFloat(nn.Module):
            def forward(self, x):
                return x.to(dtype=torch.float, copy=True)

        inputs = [torch.rand((1, 3, 10)).float()]
        self.run_test(
            ToCopyFloat(),
            inputs,
            expected_ops={torch.ops.aten._to_copy.default},
            precision=torch.float,
            disable_passes=True,
        )


if __name__ == "__main__":
    run_tests()
