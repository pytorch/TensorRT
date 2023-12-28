import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestRemainderConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("1d", (5,), 3),
            ("2d", (2, 1), 1.0),
            ("3d", (2, 1, 2), 2),
        ]
    )
    def test_remainder_scalar(self, _, shape, scalar):
        class Remainder(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.remainder.Scalar(lhs_val, scalar)

        inputs = [torch.randn(shape)]
        self.run_test(
            Remainder(),
            inputs,
        )

    def test_remainder_scalar_int(self, scalar=3):
        class Remainder(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.remainder.Scalar(lhs_val, scalar)

        inputs = [torch.tensor([0, 1, 2, 3, 4, -1, -2, -3, -4], dtype=torch.float32)]
        self.run_test(
            Remainder(),
            inputs,
        )

    @parameterized.expand(
        [
            ("1d", (5,)),
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_remainder_tensor(self, _, shape):
        class Remainder(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.remainder.Tensor(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            Remainder(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
