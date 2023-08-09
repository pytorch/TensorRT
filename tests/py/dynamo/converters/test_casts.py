import torch
import torch.nn as nn
from harness import DispatchTestCase
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo.conversion import UnsupportedOperatorException


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


if __name__ == "__main__":
    run_tests()
