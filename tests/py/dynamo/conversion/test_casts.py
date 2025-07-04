# type: ignore

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import dtype
from torch_tensorrt.dynamo.conversion import UnsupportedOperatorException

from .harness import DispatchTestCase


class TestCloneConverter(DispatchTestCase):
    def test_clone_contiguous(self):
        class Clone(nn.Module):
            def forward(self, x):
                y = torch.ops.aten.clone.default(
                    x, memory_format=torch.contiguous_format
                )
                return y + 1

        inputs = [torch.randn((1, 3, 10))]
        self.run_test(
            Clone(),
            inputs,
        )

    def test_clone_regular(self):
        class Clone(nn.Module):
            def forward(self, x):
                y = torch.ops.aten.clone.default(x)
                return y + 1

        inputs = [torch.randn((8, 2, 10))]
        self.run_test(
            Clone(),
            inputs,
        )


class TestToCopyConverter(DispatchTestCase):
    def test_to_copy_half(self):
        class ToCopyHalf(nn.Module):
            def forward(self, x):
                y = torch.ops.aten._to_copy.default(x, dtype=torch.half)
                return y

        inputs = [torch.rand((1, 3, 10))]
        self.run_test(
            ToCopyHalf(),
            inputs,
            precision=torch.half,
        )

    def test_to_copy_float(self):
        class ToCopyFloat(nn.Module):
            def forward(self, x):
                y = torch.ops.aten._to_copy.default(x, dtype=torch.float)
                return y

        inputs = [torch.rand((1, 3, 10)).half()]
        self.run_test(
            ToCopyFloat(),
            inputs,
            precision=torch.float,
        )

    def test_to_copy_bfloat16(self):
        class ToCopyBFloat16(nn.Module):
            def forward(self, x):
                y = torch.ops.aten._to_copy.default(x, dtype=torch.bfloat16)
                y = y**2
                return y

        inputs = [torch.rand((1, 3, 10), dtype=torch.float32)]
        self.run_test(
            ToCopyBFloat16(),
            inputs,
            precision=torch.float,
            use_explicit_typing=True,
        )

    def test_to_copy_i64b(self):
        class ToCopy64Bit(nn.Module):
            def forward(self, x):
                y = torch.ops.aten._to_copy.default(x, dtype=torch.int64)
                return y

        inputs = [torch.randn((1, 3, 10)).int()]

        self.run_test(
            ToCopy64Bit(),
            inputs,
        )

    def test_to_copy_multiple_returns(self):
        class ToCopyReturns(nn.Module):
            def forward(self, x):
                x_1 = x + 1
                y = torch.ops.aten._to_copy.default(x_1, dtype=torch.float)
                z = torch.ops.aten._to_copy.default(x_1, dtype=torch.float)
                return y, z

        inputs = [torch.rand((1, 3, 10))]
        self.run_test(
            ToCopyReturns(),
            inputs,
            precision=torch.float,
        )


if __name__ == "__main__":
    run_tests()
